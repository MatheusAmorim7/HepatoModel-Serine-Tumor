"""Matched hepatic capability and GLPK throughput tests, not a spatial run.

Fluxes are normalized diagnostic capacities. Input SBMLs remain untouched.
"""
import argparse
import gc
import hashlib
import json
import logging
import platform
import resource
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cobra
from cobra.flux_analysis import pfba
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'config/hepatonet_original_13_08.xml'
REDUCED = ROOT / 'notebooks/serine_coculture_compartment_pilot/models/curated_transport_local_cap0.2.xml'
ATP, SER = 'CULTURE_ATPM', 'EX_HC00068_s'
GLC, O2, GLY, GLN = 'EX_HC00040_s', 'EX_HC00017_s', 'EX_HC00045_s', 'EX_HC00067_s'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rss_mb():
    # Current resident set, not accumulated process peak.
    return int(Path('/proc/self/statm').read_text().split()[1]) * 4096 / 2**20

def load(label):
    start = time.perf_counter()
    cobra.Configuration().solver = 'glpk'
    path = REDUCED if label == 'reduced_curated' else ORIGINAL
    m = cobra.io.read_sbml_model(str(path))
    corrections, additions = [], []
    if label != 'reduced_curated':
        for node in ET.parse(ORIGINAL).getroot().iter():
            if node.tag.split('}')[-1] != 'reaction':
                continue
            rid = node.get('id')
            if node.get('reversible') == 'false' and rid in m.reactions:
                r = m.reactions.get_by_id(rid)
                if r.lower_bound < 0:
                    corrections.append(dict(reaction=rid, before=r.lower_bound, after=0))
                    r.lower_bound = 0
        template = cobra.io.read_sbml_model(str(REDUCED))
        # Same assay exchanges and ATP hydrolysis, plus optional effective carriers.
        candidates = list(template.boundary) + [template.reactions.get_by_id(ATP)]
        if label == 'full_with_effective_carriers':
            candidates += [r for r in template.reactions if r.id.startswith('CUR_')]
            if 'r1435' in m.reactions:
                corrections.append(dict(reaction='r1435', before=m.reactions.r1435.lower_bound, after=-1000))
                m.reactions.r1435.bounds = (-1000, 1000)
        for r in candidates:
            if r.id not in m.reactions:
                m.add_reactions([r.copy()])
                additions.append(r.id)
    m.solver = 'glpk'
    m.solver.configuration.timeout = 30
    # All diagnostic variants use identical tracked exchange limits.
    template = cobra.io.read_sbml_model(str(REDUCED))
    for r in m.boundary:
        r.bounds = (0, 0)
    for r in template.boundary:
        if r.id in m.reactions:
            m.reactions.get_by_id(r.id).bounds = r.bounds
    m.reactions.get_by_id(SER).bounds = (0, 1000)
    m.reactions.get_by_id(ATP).bounds = (0, 1000)
    return m, dict(model=label, input=str(path.relative_to(ROOT)), sha256=sha(path),
        reactions=len(m.reactions), metabolites=len(m.metabolites),
        load_and_prepare_seconds=time.perf_counter()-start,
        corrections=corrections, added_reactions=additions,
        open_boundaries={r.id:list(r.bounds) for r in m.boundary if r.bounds != (0, 0)})

def apply(m, medium, arm):
    if medium == 'no_organic':
        # Only water, inorganic phosphate, bicarbonate, oxygen and CO2 remain.
        inorganic = {'EX_HC00011_s','EX_HC00019_s','EX_HC00253_s',O2,'EX_HC00021_s','EX_HC00024_s'}
        for r in m.boundary:
            if r.id not in inorganic:
                r.bounds = (0, 0)
    elif medium == 'glucose_glutamine':
        for r in m.boundary:
            if r.id not in {GLC, GLN, O2, 'EX_HC00011_s', 'EX_HC00019_s'}:
                r.lower_bound = max(0, r.lower_bound)
        m.reactions.get_by_id(GLN).lower_bound = -11.9047619047619
    elif medium == 'candidate_AA_no_glycine':
        m.reactions.get_by_id(GLY).lower_bound = 0
    elif medium != 'candidate_AA':
        raise ValueError(medium)
    for rid in {'none':[], 'PHGDH':['r0338'], 'PHGDH_SHMT':['r0338','r0230','r0231'],
                'export':[SER]}[arm]:
        if rid in m.reactions:
            m.reactions.get_by_id(rid).bounds = (0, 0)

def capabilities(m, label):
    rows, vectors = [], []
    for medium in ['no_organic','glucose_glutamine','candidate_AA','candidate_AA_no_glycine']:
        for arm in ['none','PHGDH','PHGDH_SHMT','export']:
            for task in ['ATP_capacity','serine_capacity','maintenance_min_flux']:
                with m:
                    apply(m, medium, arm)
                    # 5 is inherited exploratory ATP demand, not a fitted PHH rate.
                    m.reactions.get_by_id(ATP).bounds = (0, 1000) if task == 'ATP_capacity' else (5, 5)
                    m.objective = ATP if task != 'serine_capacity' else SER
                    m.objective_direction = 'max'
                    t = time.perf_counter()
                    sol = m.optimize()
                    row = dict(model=label, medium=medium, arm=arm, task=task,
                               status=sol.status, primary_value=None, seconds=time.perf_counter()-t)
                    if sol.status == 'optimal':
                        row['primary_value'] = float(sol.objective_value)
                        # pFBA here is a witness, not HepatoNet's thermodynamic protocol.
                        sol = pfba(m)
                        sv = np.zeros(len(m.metabolites))
                        indexes = {met.id:i for i,met in enumerate(m.metabolites)}
                        for r in m.reactions:
                            for met, coef in r.metabolites.items():
                                sv[indexes[met.id]] += coef * sol.fluxes[r.id]
                        violation = max(max(r.lower_bound-sol.fluxes[r.id], sol.fluxes[r.id]-r.upper_bound, 0) for r in m.reactions)
                        row.update(serine=float(sol.fluxes[SER]), ATP=float(sol.fluxes[ATP]),
                                   oxygen=float(sol.fluxes[O2]), glucose=float(sol.fluxes[GLC]),
                                   max_Sv=float(abs(sv).max()), max_bound_violation=float(violation))
                        assert row['max_Sv'] < 1e-6 and violation < 1e-6
                        vectors += [dict(model=label,medium=medium,arm=arm,task=task,
                                        reaction=rid,flux=float(v)) for rid,v in sol.fluxes.items() if abs(v)>1e-9]
                    rows.append(row)
        print(label, medium, 'complete', flush=True)
    return rows, vectors

def throughput(m, label, sizes):
    # One persistent solver, changing local bounds. This measures LP throughput;
    # it does not simulate N cells, BioFVM or independent solver memory per cell.
    rows = []
    m.reactions.get_by_id(ATP).bounds = (5, 5)
    m.objective = SER
    m.objective_direction = 'max'
    for _ in range(3):
        m.slim_optimize()
    for n in sizes:
        before = rss_mb()
        t = time.perf_counter()
        successes = 0
        for i in range(n):
            m.reactions.get_by_id(O2).lower_bound = -(1 + 5*(i % 17)/16)
            m.reactions.get_by_id(GLC).lower_bound = -(10 + 20*(i % 13)/12)
            m.slim_optimize()
            successes += m.solver.status == 'optimal'
        duration = time.perf_counter()-t
        rows.append(dict(model=label, lp_updates=n, optimal=successes, seconds=duration,
                         ms_per_update=1000*duration/n, projected_360_updates_seconds=duration*360,
                         rss_before_mb=before,rss_after_mb=rss_mb(),
                         scope='serial persistent COBRApy GLPK solver; not agents or C++ runtime'))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes',nargs='+',type=int,default=[100,500,1000])
    args = parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings('ignore',message='Solver status is.*')
    out = HERE/'outputs'
    out.mkdir(exist_ok=True)
    protected = {str(p.relative_to(ROOT)):sha(p) for p in [ORIGINAL,REDUCED,ROOT/'config/PhysiCell_settings.xml',ROOT/'custom_modules/custom.cpp']}
    inventory, rows, vectors, timing = [], [], [], []
    for label in ['reduced_curated','full_native','full_with_effective_carriers']:
        m, info = load(label)
        inventory.append(info)
        r,v = capabilities(m,label)
        rows += r
        vectors += v
        timing += throughput(m,label,args.sizes)
        for name, data in [('capabilities',rows),('active_fluxes',vectors),('throughput',timing)]:
            pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
        del m
        gc.collect()
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    manifest = dict(inputs=protected,models=inventory,python=platform.python_version(),
                    cobra=cobra.__version__,solver='GLPK',cpu=platform.processor(),
                    peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    inherited_medium='candidate snapshot; normalized stock ceilings, not blood or calibrated culture',
                    no_thermodynamic_constraints=True,no_spatial_simulation=True,
                    no_per_agent_memory_measurement=True,protected_inputs_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).query("medium == 'candidate_AA' and task == 'serine_capacity'")[['model','arm','status','primary_value']].to_string(index=False))
    print(pd.DataFrame(timing).to_string(index=False))

if __name__ == '__main__':
    main()
