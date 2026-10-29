"""Paired demand-composition sensitivity, not biological demand calibration."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import cobra
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'notebooks/poster_result_closure_2026_10_25'))
import run_closure as prior

HOURS, TOL = 12., 1e-7
SER, IMP, C1 = .056, .014, .014
WORK, MAINT = 2.3912, .0035
FACTORS_S, FACTORS_P, FACTORS_C = [.5, 1., 2.], [.5, 1., 2.], [0., .5, 1., 2.]
TARGETS = [.25, .55, .85]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compositions():
    return [dict(composition=f'S{s:g}_P{p:g}_C{c:g}', ser_factor=s,
                 imp_factor=p, c1_factor=c, ser_rate=SER*s, imp_rate=IMP*p, c1_rate=C1*c)
            for s, p, c in itertools.product(FACTORS_S, FACTORS_P, FACTORS_C)]


def plan():
    rows = []
    for comp in compositions():
        for energy in ['fixed_ATP', 'Ser_linked_ATP']:
            services = [None]+TARGETS if energy == 'fixed_ATP' else [None]
            for label, hx, hprt in prior.SALVAGE:
                for channel in ['closed', 'serine_only']:
                    for target in services:
                        rows.append(comp | dict(energy_mode=energy, HX_pmol=hx, HPRT=hprt,
                            salvage_condition=label, channel=channel, target=target,
                            work_rate=WORK if energy == 'fixed_ATP' else WORK*comp['ser_factor']))
    assert len(rows) == 1080
    return [r | dict(case=f'demand_{i:04d}') for i, r in enumerate(rows)]


def load_inputs():
    cobra.Configuration().solver = 'glpk'
    logging.getLogger('cobra').setLevel(logging.ERROR)
    native = cobra.io.read_sbml_model(str(ROOT/'config/hepatonet_original_13_08.xml'))
    template = prior.community.shared.engine.build(native)
    curated = native.copy()
    prior.community.shared.donor.apply_published(curated)
    q = json.loads(prior.community.shared.engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    fields = pd.read_csv(prior.HERE/'fields.csv').fillna({'cobra_exchange': ''})
    return native, template, curated, q, fields


def templates(inputs):
    native, template, curated, q, fields = inputs
    cache = {}
    for label, hx, hprt in prior.SALVAGE:
        for channel in ['closed', 'serine_only']:
            spec = prior.community_plan()[0] | dict(panel='demand_sensitivity',
                precursor_access='external_ribose', salvage_condition=label,
                HX_pmol=hx, HPRT=hprt, channel=channel, hepatic_PHGDH='available',
                fixed_service=None, selection='resource_sparing', policy='glucose_sparing')
            joint, initial, idx = prior.prepare_community(spec, fields, template, native, curated, q)
            cache[(label, channel)] = (joint, initial, idx, spec)
    return cache


def clone(base, fields):
    original, initial, idx, spec = base
    m = original.model.copy()
    m.solver.configuration.tolerances.feasibility = 1e-9
    m.solver.configuration.timeout = 45
    joint = SimpleNamespace(model=m, factor=original.factor, ids=original.ids,
        groups=original.groups, S=original.S,
        theta=m.solver.variables[original.theta.name],
        ha=m.solver.variables[original.ha.name], ta=m.solver.variables[original.ta.name])
    joint.r = {g:{rid:m.reactions.get_by_id(r.id) for rid,r in group.items()}
               for g,group in original.r.items()}
    joint.gross, joint.net, joint.budget = {}, {}, {}
    for field in fields.itertuples():
        gross, net = 0., 0.
        for g in joint.r:
            rid = prior.community.ROLES.get(field.substrate) if g == 'tumor' else field.cobra_exchange
            if rid in joint.r[g]:
                r = joint.r[g][rid]
                gross += joint.factor[g]*r.reverse_variable
                net += joint.factor[g]*r.flux_expression
        joint.gross[field.substrate], joint.net[field.substrate] = gross, net
        joint.budget[field.substrate] = m.solver.constraints[original.budget[field.substrate].name]
    joint.faces = {k:m.solver.constraints[c.name] for k,c in original.faces.items()}
    joint.l1 = {var:joint.factor[g] for g in joint.r for r in joint.r[g].values()
                for var in [r.forward_variable,r.reverse_variable]}
    return joint, initial.copy(), idx, spec.copy()


def configure(joint, row):
    m = joint.model
    for name, rate in [('minimum_DM_SER_c',row['ser_rate']),
                       ('community_IMP_material_task',row['imp_rate']),
                       ('minimum_DM_BOOKKEEP_C1',row['c1_rate']),
                       ('paid_energy_requirement',row['work_rate'])]:
        con = m.solver.constraints[name]
        con.set_linear_coefficients({joint.theta:-rate})
        con.lb = con.ub = 0.
    if row['target'] is not None:
        joint.theta.lb = joint.theta.ub = row['target']
    # Only task coefficients change. No medium, pathway or maintenance relaxation.
    assert abs(float(joint.ha.lb)-1.) < TOL and abs(float(joint.ta.lb)-1.) < TOL


def solve(joint, fixed):
    m, p = joint.model, joint.model.problem
    m.objective = p.Objective(joint.theta, direction='max')
    value = m.slim_optimize()
    if m.solver.status != 'optimal':
        assert m.solver.status == 'infeasible', m.solver.status
        return None, None
    theta = float(value)
    joint.faces['tasks'].lb = joint.faces['tasks'].ub = theta
    if fixed:
        source = joint.factor['local']*joint.r['local'][prior.community.shared.SER].flux_expression
        m.objective = p.Objective(source, direction='min')
        value = float(m.slim_optimize(error_value=None))
        m.add_cons_vars(p.Constraint(source, lb=max(0.,value-1e-10), ub=max(0.,value)+1e-10,
                                    name='sensitivity_minimum_source'))
    for key in ['glucose','oxygen']:
        m.objective = p.Objective(joint.gross[key], direction='min')
        value = float(m.slim_optimize(error_value=None))
        joint.faces[key].lb, joint.faces[key].ub = max(0.,value-1e-9), max(0.,value)+1e-9
    m.objective = p.Objective(0., direction='min')
    m.objective.set_linear_coefficients(joint.l1)
    m.slim_optimize(error_value=None)
    flux = {g:{rid:float(r.flux) for rid,r in group.items()} for g,group in joint.r.items()}
    return theta, flux


def validate(joint, row, fields, initial, idx, theta, flux):
    sv = bound = 0.
    for g in joint.r:
        v = np.array([flux[g][rid] for rid in joint.ids[g]])
        sv = max(sv, float(abs(joint.S[g]@v).max()))
        for rid, r in joint.r[g].items():
            bound = max(bound, r.lower_bound-flux[g][rid], flux[g][rid]-r.upper_bound)
    t = flux['tumor']
    model = joint.groups['tumor']
    cnp = sum((sum((prior.community.purine.atoms(met)*coef for met,coef in r.metabolites.items()),
        start=np.zeros(3))*t[r.id] for r in model.boundary),start=np.zeros(3))
    made = 2*t['PAID_GLYCOLYSIS']+24*t['PAID_OXIDATION']
    used = sum(-r.metabolites.get(model.metabolites.HC00012_c,0.)*t[r.id]
               for r in model.reactions if r.id not in ['PAID_GLYCOLYSIS','PAID_OXIDATION'])
    eq = max(abs(t['DM_SER_c']-row['ser_rate']*theta),
        abs(t['DM_IMP_MATERIAL']-row['imp_rate']*theta),
        abs(t['DM_BOOKKEEP_C1']-row['c1_rate']*theta),
        abs(t['TASK_ATP']-MAINT-row['work_rate']*theta),abs(made-used),
        abs(-t['r0273']-t['r0275']-t['DM_IMP_MATERIAL']))
    assert max(sv,bound,eq,float(abs(cnp).max())) < TOL, (row['case'],sv,bound,eq,cnp)
    assert max(abs(t[rid]) for rid in prior.community.purine.redox.BACKGROUND) < TOL
    assert all(abs(flux[g]['CULTURE_ATPM']-5.) < TOL for g in ['basal','local'])
    net, gross = np.zeros(len(fields)), np.zeros(len(fields))
    group_net = {g:np.zeros(len(fields)) for g in flux}
    for j, f in enumerate(fields.itertuples()):
        for g in flux:
            rid = prior.community.ROLES.get(f.substrate) if g == 'tumor' else f.cobra_exchange
            if rid in flux[g]:
                value = joint.factor[g]*flux[g][rid]
                group_net[g][j] = value
                net[j] += value
                gross[j] += max(0.,-value)
    credit = np.zeros(len(fields))
    for role, rid in [('serine',prior.community.shared.SER),('formate',prior.community.shared.FOR)]:
        credit[idx[role]] = joint.factor['local']*flux['local'][rid]
        assert credit[idx[role]] >= -TOL
    sink = np.array([3*t['DM_SER_c']+t['DM_BOOKKEEP_C1']+10*t['DM_IMP_MATERIAL'],
                     t['DM_SER_c']+4*t['DM_IMP_MATERIAL']])
    max_cn = max_inventory = max_pool = 0.
    endpoints = []
    for dt in [.1,.05]:
        stock, external, reservoir = initial.copy(), np.zeros(len(fields)), 32.
        for step in range(round(HOURS/dt)):
            pool = float(np.maximum(0.,gross*dt-stock-credit*dt).max())
            stock += dt*net
            feed = min(max(0.,4.-stock[idx['oxygen']]),3.*dt,reservoir)
            stock[idx['oxygen']] += feed
            external[idx['oxygen']] += feed
            reservoir -= feed
            time = (step+1)*dt
            inv = stock-initial-external-time*net
            cn = (stock-initial-external)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()+time*sink
            max_cn = max(max_cn,float(abs(cn).max()))
            max_inventory = max(max_inventory,float(abs(inv).max()))
            max_pool = max(max_pool,pool)
            assert stock.min() >= -TOL and max(pool,float(abs(inv).max()),float(abs(cn).max())) < TOL
        endpoints.append(stock)
    assert float(abs(endpoints[0]-endpoints[1]).max()) < TOL
    metrics = dict(theta=theta, task_h=HOURS*theta,
        hepatic_Ser_export_pmol=HOURS*credit[idx['serine']],
        tumor_Ser_uptake_pmol=-HOURS*t['EX_SER_c'],
        tumor_Ser_material_pmol=HOURS*t['DM_SER_c'],
        tumor_IMP_material_pmol=HOURS*t['DM_IMP_MATERIAL'],
        tumor_C1_material_pmol=HOURS*t['DM_BOOKKEEP_C1'],
        IMP_de_novo_pmol=-HOURS*t['r0273'], IMP_salvage_pmol=-HOURS*t['r0275'],
        glucose_uptake_pmol=HOURS*gross[idx['glucose']], oxygen_uptake_pmol=HOURS*gross[idx['oxygen']],
        ATP_paid_pmol=HOURS*made, max_Sv=sv,max_bound_error=max(0.,bound),
        max_equation_error=eq,max_CNP_error=float(abs(cnp).max()),max_CN_error=max_cn,
        max_inventory_error=max_inventory,max_pool_error=max_pool)
    inventory = [dict(substrate=f.substrate,initial_pmol=initial[j],external_pmol=external[j],
        final_pmol=endpoints[-1][j],basal_change_pmol=HOURS*group_net['basal'][j],
        local_change_pmol=HOURS*group_net['local'][j],tumor_change_pmol=HOURS*group_net['tumor'][j])
        for j,f in enumerate(fields.itertuples())]
    return metrics, inventory


def protected():
    names = ['config/hepatonet_original_13_08.xml','config/PhysiCell_settings.xml',
        'custom_modules/custom.cpp','addons/dFBA/src/dfba_Model.cpp','addons/dFBA/src/dfba_intracellular.cpp']
    for folder in [prior.HERE,ROOT/'notebooks/serine_purine_community_2026_10_25',
                   ROOT/'notebooks/serine_shared_Ser_formate_2026_10_24']:
        names += [p.relative_to(ROOT).as_posix() for p in folder.iterdir() if p.is_file()]
    return {name:sha(ROOT/name) for name in names}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    out = HERE/('smoke' if args.smoke else 'outputs')
    out.mkdir(exist_ok=True)
    assert not (out/'results.csv').exists(), 'Use a new output folder; previous results are preserved'
    cases = plan()
    if args.smoke:
        cases = [r for r in cases if r['composition']=='S1_P1_C1' and
                 (r['target'] in [None,.55])]
        assert len(cases)==18
    frozen = protected()
    protocol = dict(declared_before_execution=True,cases=cases,script_sha256=sha(__file__),protected=frozen,
        composition_factors=dict(serine=FACTORS_S,IMP=FACTORS_P,residual_C1=FACTORS_C),
        reference_rates_pmol_h=dict(serine=SER,IMP=IMP,C1=C1,work_ATP=WORK,maintenance_ATP=MAINT),
        targets=TARGETS,energy_modes=dict(fixed_ATP='Same task-ATP coefficient across compositions; material sensitivity',
            Ser_linked_ATP='Additional sensitivity: work ATP scales with Ser factor; no calibrated translation'),
        primary='fixed_ATP; 36 compositions, 3 salvage conditions, 2 export channels, max and three fixed targets',
        zero_C1='Diagnostic omission, not a complete biological tumour',
        fixed_background='External ribose access, same medium, 252 basal+12 local+7 tumour equivalents,20nL,12h',
        hepatic_PHGDH='Available in all paired cases; same permitted diagnostic DMG0.144pmol; source is not induced secretion',
        no_calibration=True,no_growth=True,cooperative_capacity_bound=True,active_PhysiCell_changed=False,
        negative_results='All infeasible targets retained; no nutrient additions or maintenance relaxation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs=load_inputs()
    fields=inputs[-1]
    fields.to_csv(out/'fields.csv',index=False)
    cache=templates(inputs)
    results=[]
    with gzip.open(out/'fluxes.csv.gz','wt',newline='') as vf, (out/'stocks.csv').open('w',newline='') as sf:
        fw=csv.DictWriter(vf,fieldnames=['case','group','reaction','flux']);fw.writeheader()
        iw=None
        for i,row in enumerate(cases):
            joint,initial,idx,spec=clone(cache[(row['salvage_condition'],row['channel'])],fields)
            configure(joint,row)
            theta,flux=solve(joint,row['target'] is not None)
            if flux is None:
                results.append(row|dict(status='infeasible_fixed_target' if row['target'] is not None else 'infeasible_maintenance'))
            else:
                metrics,inventory=validate(joint,row,fields,initial,idx,theta,flux)
                results.append(row|dict(status='optimal')|metrics)
                for g in flux:
                    fw.writerows(dict(case=row['case'],group=g,reaction=rid,flux=v) for rid,v in flux[g].items())
                if iw is None:
                    iw=csv.DictWriter(sf,fieldnames=['case']+list(inventory[0]));iw.writeheader()
                iw.writerows(dict(case=row['case'])|v for v in inventory)
            if (i+1)%18==0 or i==len(cases)-1:
                pd.DataFrame(results).to_csv(out/'results.csv',index=False)
                print(f'{i+1}/{len(cases)} {row["composition"]} {results[-1]["status"]}',flush=True)
    assert all(sha(ROOT/name)==h for name,h in frozen.items())
    verification=dict(cases=len(results),status_counts=pd.DataFrame(results).status.value_counts().to_dict(),
        protected_files_unchanged=True,all_saved_feasible_vectors_checked=True,stock_dt_checks=[.1,.05],
        no_biological_validation=True,task_coefficients_not_fitted=True)
    (out/'run_checks.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()
