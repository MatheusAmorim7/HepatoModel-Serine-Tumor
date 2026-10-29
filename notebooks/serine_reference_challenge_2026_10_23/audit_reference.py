"""Matched reference preparation, energy gate and native mitochondrial C1 inventory."""
import hashlib
import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cobra
from cobra.flux_analysis import pfba
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REF=ROOT/'notebooks/serine_full_network_comparison_2026_10_22'
sys.path.insert(0,str(REF))
from curate_directions import apply_published

NATIVE=ROOT/'config/hepatonet_original_13_08.xml'
CANDIDATE=ROOT/'notebooks/serine_evidence_supply_map/config/map_14_local.xml'
ATP='CULTURE_ATPM'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR);cobra.Configuration().solver='glpk'
    inputs=[NATIVE,CANDIDATE,Path(__file__),ROOT/'config/PhysiCell_settings.xml',
            ROOT/'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
            REF/'sources/44320_2010_BFMSB201062_MOESM4_ESM.txt',
            ROOT/'notebooks/serine_compartment_route_blocks_2026_10_23/results.csv']
    protected={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    (HERE/'protocol.json').write_text(json.dumps(dict(protected=protected,declared_before_execution=True,
        gates=['all_exchanges_closed','no_organic_inputs'],
        biological_retest='Only if both ATP gates optimal and zero; otherwise no quantitative reference claim',
        matching='Supplement directions, then candidate bounds for shared reactions; candidate-only helpers and ports copied'),indent=2)+'\n')
    native=cobra.io.read_sbml_model(str(NATIVE));candidate=cobra.io.read_sbml_model(str(CANDIDATE))
    full=native.copy();directions=apply_published(full)
    additions=[];overrides=[]
    for r in candidate.reactions:
        if r.id not in full.reactions:
            full.add_reactions([r.copy()]);additions.append(r.id)
        else:
            old=full.reactions.get_by_id(r.id).bounds
            if old!=r.bounds:
                overrides.append(dict(reaction=r.id,old_lb=old[0],old_ub=old[1],lb=r.lower_bound,ub=r.upper_bound))
            full.reactions.get_by_id(r.id).bounds=r.bounds
    for r in full.boundary:
        if r.id not in candidate.reactions:
            r.bounds=(0.,0.)
    inventory=[]
    c1=['HC00140_m','HC00102_m','HC00360_m','HC00212_m','HC00340_m','HC00061_m']
    for rid in sorted({r.id for mid in c1 for r in native.metabolites.get_by_id(mid).reactions}):
        r=native.reactions.get_by_id(rid)
        inventory.append(dict(reaction=rid,equation=r.reaction,boundary=r.boundary,
            present_in_candidate=rid in candidate.reactions,native_lb=r.lower_bound,native_ub=r.upper_bound,
            candidate_lb=candidate.reactions.get_by_id(rid).lower_bound if rid in candidate.reactions else None,
            candidate_ub=candidate.reactions.get_by_id(rid).upper_bound if rid in candidate.reactions else None))
    pd.DataFrame(inventory).to_csv(HERE/'mitochondrial_C1_inventory.csv',index=False)
    rows=[];witnesses=[];positive=[]
    for label,model in [('candidate_177',candidate),('full_matched',full)]:
        model.solver.configuration.timeout=20
        model.solver.configuration.tolerances.feasibility=1e-9
        for gate in ['all_exchanges_closed','no_organic_inputs']:
            with model:
                inorganic={'EX_HC00011_s','EX_HC00019_s','EX_HC00253_s','EX_HC00017_s','EX_HC00021_s','EX_HC00024_s'}
                for r in model.boundary:
                    if gate=='all_exchanges_closed' or r.id not in inorganic:
                        r.bounds=(0.,0.)
                model.reactions.get_by_id(ATP).bounds=(0.,1000.)
                model.objective=ATP;model.objective_direction='max'
                value=model.slim_optimize();status=model.solver.status
                passed=status=='optimal' and abs(value)<1e-7
                rows.append(dict(model=label,gate=gate,status=status,
                    ATP_capacity=value if status=='optimal' else None,passed=passed,
                    reactions=len(model.reactions),metabolites=len(model.metabolites)))
                if gate=='all_exchanges_closed' and status=='optimal' and value>1e-7:
                    model.reactions.get_by_id(ATP).bounds=(1.,1.)
                    sol=pfba(model)
                    assert sol.status=='optimal'
                    assert max(abs(sol.fluxes[r.id]) for r in model.boundary)<1e-8
                    residual=np.max(abs(cobra.util.array.create_stoichiometric_matrix(model)@sol.fluxes.to_numpy()))
                    assert residual<1e-7
                    for r in model.reactions:
                        if abs(sol.fluxes[r.id])>1e-8:
                            witnesses.append(dict(model=label,reaction=r.id,flux=sol.fluxes[r.id],
                                equation=r.reaction,boundary=r.boundary,max_Sv=residual))
                print(label,gate,status,value,'passed',passed,flush=True)
        good=all(r['passed'] for r in rows if r['model']==label)
        # A failed reference remains excluded from biological capacity comparison.
        if good:
            for blocks in [[],['r0338'],['r0338','r0295'],['r0338','r0231']]:
                with model:
                    for rid in blocks:model.reactions.get_by_id(rid).bounds=(0.,0.)
                    model.reactions.get_by_id(ATP).bounds=(5.,5.)
                    model.reactions.get_by_id('EX_HC00068_s').bounds=(0.,.2)
                    model.objective='EX_HC00068_s';model.objective_direction='max'
                    value=model.slim_optimize();status=model.solver.status
                    positive.append(dict(model=label,blocks=','.join(blocks) or 'none',status=status,
                                         Ser_capacity=value if status=='optimal' else None))
    pd.DataFrame(rows).to_csv(HERE/'energy_gates.csv',index=False)
    pd.DataFrame(witnesses).to_csv(HERE/'closed_ATP_witness.csv',index=False)
    pd.DataFrame(positive).to_csv(HERE/'eligible_capacity_tests.csv',index=False)
    pd.DataFrame(overrides).to_csv(HERE/'matching_bound_overrides.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    meta=dict(original_reactions=len(native.reactions),candidate_reactions=len(candidate.reactions),
        prepared_full_reactions=len(full.reactions),copied_reactions=additions,
        published_directions_matched=len(directions),common_bounds_overridden=len(overrides),
        protected_unchanged=True,full_reference_ready=all(r['passed'] for r in rows if r['model']=='full_matched'),
        full_biological_capacity_interpreted=False)
    (HERE/'verification.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    main()
