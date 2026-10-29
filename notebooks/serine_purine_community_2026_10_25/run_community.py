"""Paired finite-stock community with an explicit tumour IMP material task.

This is a conditional capacity experiment, not induced secretion, calibrated
PATU metabolism, proliferation, or a deployment into the active PhysiCell run.
"""
import argparse
import json
import logging
from pathlib import Path
import sys

import cobra
from cobra.flux_analysis import flux_variability_analysis
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/serine_shared_Ser_formate_2026_10_24'))
import run_shared as shared
sys.path.insert(0, str(ROOT / 'notebooks/serine_purine_material_2026_10_25'))
import run_purine as purine

HOURS = 12.
TOL = 1e-7
ROLES = purine.PORTS


def fields_table():
    f = shared.fields_table()
    extra = pd.DataFrame([
        dict(substrate='ribose', cobra_exchange='', carbon_atoms=5, nitrogen_atoms=0, initial_pmol=.3),
        dict(substrate='hypoxanthine', cobra_exchange='', carbon_atoms=5, nitrogen_atoms=4, initial_pmol=0.),
        dict(substrate='fumarate', cobra_exchange='', carbon_atoms=4, nitrogen_atoms=0, initial_pmol=0.)])
    return pd.concat([f, extra], ignore_index=True).fillna({'cobra_exchange': ''})


def plan():
    base = shared.plan()[0] | dict(panel='primary', capacity_fraction=1.,
        tumor_assembly='closed', hepatic_PHGDH='blocked', fixed_C1_pmol_h=0.,
        residual_C1=0., HX_pmol=0., HPRT='available', HPRT_cap_pmol_h=.14,
        ribose_source='finite_ribose', redox_disposal=True, formate_oxidation=False,
        fixed_service=None, selection='resource_sparing', tumor_gly_routes='closed')
    cases = []
    for residual in [0., .014]:
        for hx in [0., .2]:
            for phgdh in ['available', 'blocked']:
                for channel in shared.CHANNELS:
                    cases.append(base | dict(residual_C1=residual, HX_pmol=hx,
                        hepatic_PHGDH=phgdh, channel=channel))
    for phgdh in ['available', 'blocked']:
        for channel in shared.CHANNELS:
            cases.append(base | dict(panel='HPRT_closed', residual_C1=.014,
                HX_pmol=.2, HPRT='closed', hepatic_PHGDH=phgdh, channel=channel))
    # This target is above some no-source capacities; infeasibility is retained,
    # not repaired by extra nutrients or relaxing maintenance.
    for residual in [0., .014]:
        for hx in [0., .2]:
            for hprt in ['available', 'closed']:
                for channel in ['serine_only', 'formate_only', 'closed']:
                    cases.append(base | dict(panel='matched_demand', residual_C1=residual,
                        HX_pmol=hx, HPRT=hprt, channel=channel,
                        fixed_service=.55, selection='source_sparing'))
    for hx in [0., .2]:
        for channel in shared.CHANNELS:
            cases.append(base | dict(panel='tumour_Gly_routes', residual_C1=.014,
                HX_pmol=hx, channel=channel, tumor_gly_routes='available'))
            cases.append(base | dict(panel='glucose_PPP_precursor', residual_C1=.014,
                HX_pmol=hx, channel=channel, ribose_source='glucose_PPP'))
    for hx in [0., .2]:
        for channel in ['both', 'closed']:
            cases.append(base | dict(panel='no_DMG', residual_C1=.014,
                HX_pmol=hx, channel=channel, donor_stock_pmol=0.))
    for cap in [.001, .005]:
        for channel in ['both', 'closed']:
            cases.append(base | dict(panel='HPRT_capacity', residual_C1=.014,
                HX_pmol=.2, channel=channel, HPRT_cap_pmol_h=cap))
    assert len(cases) == 88
    return [s | dict(case=f'community_{i:03d}') for i, s in enumerate(cases)]


def prepare(spec, fields, template, native, curated, q):
    joint, initial, idx, _ = shared.prepare(spec, fields, template, native, curated, q)
    initial[idx['hypoxanthine']] = spec['HX_pmol']
    initial[idx['ribose']] = .3 if spec['ribose_source'] == 'finite_ribose' else 0.
    total = initial.copy()
    total[idx['oxygen']] = 36.
    standalone_spec = purine.plan()[0] | {key: spec[key] for key in [
        'residual_C1', 'ribose_source', 'redox_disposal', 'formate_oxidation', 'policy']}
    standalone_spec['blocks'] = ['r0275'] if spec['HPRT'] == 'closed' else []
    tumor = purine.build(native, template, standalone_spec, q).model
    # Transfer chemistry and bounds only. Keep the community's own task and
    # maintenance variables; standalone optlang constraints are not imported.
    m = joint.model
    assert set(joint.r['tumor']).issubset(set(tumor.reactions.list_attr('id')))
    for rxn in tumor.reactions:
        if rxn.id not in m.reactions:
            m.add_reactions([rxn.copy()])
        else:
            assert {met.id: v for met, v in rxn.metabolites.items()} == {
                met.id: v for met, v in m.reactions.get_by_id(rxn.id).metabolites.items()}
        m.reactions.get_by_id(rxn.id).bounds = rxn.bounds
    joint.r['tumor'] = {r.id: m.reactions.get_by_id(r.id) for r in tumor.reactions}
    t = joint.r['tumor']
    for role, rid in ROLES.items():
        cap = .14
        if role == 'serine':
            cap = q * 7 / 1000.  # inherited net-uptake reference, not measured Vmax
        if role in ['glycine', 'formate']:
            cap *= spec['capacity_fraction']
        if role == 'glucose':
            cap = spec['energy_glucose_cap']
        if role == 'oxygen':
            cap = 3.
        if role in ['glutamate', 'fumarate', 'lactate', 'CO2', 'ammonia', 'pentose_carbon']:
            cap = 0.
        t[rid].bounds = (-cap, 0. if role in [
            'glucose', 'oxygen', 'ribose', 'hypoxanthine', 'glutamine', 'aspartate'] else 140.)
    t['r0275'].bounds = (-spec['HPRT_cap_pmol_h'], 0.) if spec['HPRT'] == 'available' else (0., 0.)
    if spec['tumor_gly_routes'] == 'available':
        t['r0230'].upper_bound = .14
        for rid in ['CAP_SER_m_c', 'CAP_GLY_c_m', 'r0231', 'r0295', 'PAID_NADH_OX_m']:
            t[rid].bounds = (0., .14)
    tumor_copy = cobra.Model('tumour_purine_chemistry_only')
    tumor_copy.solver = 'glpk'
    tumor_copy.add_reactions([r.copy() for r in t.values()])
    joint.groups['tumor'] = tumor_copy
    c1 = m.solver.constraints['minimum_DM_BOOKKEEP_C1']
    c1.set_linear_coefficients({joint.theta: -spec['residual_C1']})
    c1.lb = c1.ub = 0.
    m.add_cons_vars(m.problem.Constraint(t['DM_IMP_MATERIAL'].flux_expression -
        purine.PURINE_RATE*joint.theta, lb=0, ub=0, name='community_IMP_material_task'))
    if spec['fixed_service'] is not None:
        joint.theta.lb = joint.theta.ub = spec['fixed_service']

    m.remove_cons_vars(list(joint.budget.values()))
    joint.gross, joint.net, joint.budget = {}, {}, {}
    for f in fields.itertuples():
        gross, net = 0., 0.
        for g in joint.r:
            rid = ROLES.get(f.substrate) if g == 'tumor' else f.cobra_exchange
            if rid in joint.r[g]:
                r = joint.r[g][rid]
                gross += joint.factor[g]*r.reverse_variable
                net += joint.factor[g]*r.flux_expression
        credit = 0.
        if f.substrate in ['serine', 'formate']:
            credit = joint.factor['local']*joint.r['local'][f.cobra_exchange].flux_expression
        con = m.problem.Constraint(gross-credit, ub=total[idx[f.substrate]]/HOURS,
            name='purine_shared_'+f.substrate)
        m.add_cons_vars(con)
        joint.gross[f.substrate], joint.net[f.substrate], joint.budget[f.substrate] = gross, net, con
    for face in joint.faces.values():
        face.lb = face.ub = None
    joint.ids = {g: list(joint.r[g]) for g in joint.r}
    joint.S = {g: create_stoichiometric_matrix(joint.groups[g]) for g in joint.r}
    joint.l1 = {var: joint.factor[g] for g in joint.r for r in joint.r[g].values()
        for var in [r.forward_variable, r.reverse_variable]}
    return joint, initial, idx


def solve(joint, spec):
    m, p = joint.model, joint.model.problem
    m.objective = p.Objective(joint.theta, direction='max')
    value = m.slim_optimize()
    if m.solver.status != 'optimal':
        assert m.solver.status == 'infeasible', m.solver.status
        return None, []
    joint.faces['tasks'].lb = joint.faces['tasks'].ub = float(value)
    ranges = []
    if (spec['panel'] == 'primary' and spec['residual_C1'] == .014 and
            spec['hepatic_PHGDH'] == 'blocked' and spec['channel'] in ['both', 'closed']):
        ids = [joint.r['local'][shared.SER].id, joint.r['local'][shared.FOR].id,
            'r0273', 'r0275', 'EX_SER_c', 'EX_FOR_c']
        fva = flux_variability_analysis(m, ids, fraction_of_optimum=1., processes=1)
        for rid, vals in fva.iterrows():
            factor = joint.factor['local'] if rid.startswith('local__') else 1.
            ranges.append(dict(case=spec['case'], reaction=rid,
                minimum_pmol=HOURS*factor*vals['minimum'], maximum_pmol=HOURS*factor*vals['maximum']))
    if spec['selection'] == 'source_sparing':
        # The matched protocol opens at most one source channel. Minimizing
        # this carbon expression is therefore minimizing that channel's amount.
        source = joint.factor['local']*(3*joint.r['local'][shared.SER].flux_expression+
            joint.r['local'][shared.FOR].flux_expression)
        m.objective = p.Objective(source, direction='min')
        value = float(m.slim_optimize(error_value=None))
        m.add_cons_vars(p.Constraint(source, lb=max(0., value-1e-10), ub=value+1e-10,
            name='matched_minimal_source_carbon'))
    for key in (['glucose', 'oxygen'] if spec['policy'] == 'glucose_sparing' else ['oxygen', 'glucose']):
        m.objective = p.Objective(joint.gross[key], direction='min')
        value = float(m.slim_optimize(error_value=None))
        joint.faces[key].lb, joint.faces[key].ub = max(0., value-1e-9), max(0., value)+1e-9
    m.objective = p.Objective(0., direction='min')
    m.objective.set_linear_coefficients(joint.l1)
    m.slim_optimize(error_value=None)
    return {g: {rid: float(r.flux) for rid, r in joint.r[g].items()} for g in joint.r}, ranges


def validate(joint, spec, fields, initial, idx, fluxes, theta, dt=.1):
    residual = bound_error = 0.
    for g in joint.r:
        v = np.array([fluxes[g][rid] for rid in joint.ids[g]])
        residual = max(residual, float(abs(joint.S[g]@v).max()))
        for rid, value in zip(joint.ids[g], v):
            r = joint.r[g][rid]
            bound_error = max(bound_error, r.lower_bound-value, value-r.upper_bound)
    t = fluxes['tumor']
    model = joint.groups['tumor']
    cn_p = sum((sum((purine.atoms(met)*coef for met, coef in r.metabolites.items()),
        start=np.zeros(3))*t[r.id] for r in model.boundary), start=np.zeros(3))
    atp_source = 2*t['PAID_GLYCOLYSIS']+spec['yield_ATP']*t['PAID_OXIDATION']
    atp_cost = sum(-r.metabolites.get(model.metabolites.HC00012_c, 0.)*t[r.id]
        for r in model.reactions if r.id not in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION'])
    eq = max(abs(t['DM_SER_c']-.056*theta), abs(t['DM_BOOKKEEP_C1']-spec['residual_C1']*theta),
        abs(t['DM_IMP_MATERIAL']-purine.PURINE_RATE*theta),
        abs(t['TASK_ATP']-.0035-spec['work_ATP_at_full_service_pmol_h']*theta),
        abs(atp_source-atp_cost), abs(-t['r0273']-t['r0275']-t['DM_IMP_MATERIAL']),
        abs(2*t['r0199']+t['r0293']+t['r0227']-t['ASSAY_NADPH_OX']))
    assert max(residual, bound_error, eq, abs(cn_p).max()) < TOL
    assert max(abs(t[rid]) for rid in purine.redox.BACKGROUND) < TOL
    assert all(abs(fluxes[g]['CULTURE_ATPM']-5.) < TOL for g in ['basal', 'local'])
    known = set(ROLES.values()) | {'EX_WATER_c', 'EX_WATER_m', 'DM_SER_c', 'DM_BOOKKEEP_C1', 'DM_IMP_MATERIAL'}
    assert all(abs(t[r.id]) < TOL or r.id in known for r in model.boundary)
    net, gross = np.zeros(len(fields)), np.zeros(len(fields))
    group_net = {g: np.zeros(len(fields)) for g in fluxes}
    for j, f in enumerate(fields.itertuples()):
        for g in fluxes:
            rid = ROLES.get(f.substrate) if g == 'tumor' else f.cobra_exchange
            if rid in fluxes[g]:
                amount = joint.factor[g]*fluxes[g][rid]
                group_net[g][j] = amount
                net[j] += amount
                gross[j] += max(0., -amount)
    credit = np.zeros(len(fields))
    for name, rid in [('serine', shared.SER), ('formate', shared.FOR)]:
        credit[idx[name]] = joint.factor['local']*fluxes['local'][rid]
        assert credit[idx[name]] >= -TOL
    stock, external = initial.copy(), np.zeros(len(fields))
    oxygen_reservoir = 32.
    trace = []
    errors = dict(max_CN_error=0., max_inventory_error=0., max_pool_excess_pmol=0.)
    sink_cn = np.array([3*t['DM_SER_c']+t['DM_BOOKKEEP_C1']+10*t['DM_IMP_MATERIAL'],
        t['DM_SER_c']+4*t['DM_IMP_MATERIAL']])
    for k in range(round(HOURS/dt)):
        pool = float(np.maximum(0., gross*dt-stock-credit*dt).max())
        stock += dt*net
        feed = min(max(0., 4.-stock[idx['oxygen']]), 3.*dt, oxygen_reservoir)
        oxygen_reservoir -= feed
        stock[idx['oxygen']] += feed
        external[idx['oxygen']] += feed
        time = (k+1)*dt
        inv = stock-initial-external-time*net
        cn = (stock-initial-external)@fields[['carbon_atoms', 'nitrogen_atoms']].to_numpy()+time*sink_cn
        assert stock.min() >= -TOL and max(pool, abs(inv).max(), abs(cn).max()) < TOL, (
            spec['case'], stock.min(), pool, abs(inv).max(), cn)
        errors['max_CN_error'] = max(errors['max_CN_error'], float(abs(cn).max()))
        errors['max_inventory_error'] = max(errors['max_inventory_error'], float(abs(inv).max()))
        errors['max_pool_excess_pmol'] = max(errors['max_pool_excess_pmol'], pool)
        trace.append(dict(time_h=time, task_h=time*theta,
            serine_pmol=stock[idx['serine']], formate_pmol=stock[idx['formate']],
            HX_pmol=stock[idx['hypoxanthine']], oxygen_pmol=stock[idx['oxygen']],
            CN_error=float(abs(cn).max()), inventory_error=float(abs(inv).max())))
    inventory = [dict(substrate=f.substrate, initial_pmol=initial[j], external_pmol=external[j],
        final_pmol=stock[j], basal_change_pmol=HOURS*group_net['basal'][j],
        local_change_pmol=HOURS*group_net['local'][j], tumor_change_pmol=HOURS*group_net['tumor'][j])
        for j, f in enumerate(fields.itertuples())]
    metrics = dict(task_h=HOURS*theta, hepatic_Ser_export_pmol=HOURS*credit[idx['serine']],
        hepatic_formate_export_pmol=HOURS*credit[idx['formate']],
        tumor_Ser_uptake_pmol=-HOURS*t['EX_SER_c'], tumor_formate_uptake_pmol=-HOURS*t['EX_FOR_c'],
        tumor_IMP_de_novo_pmol=-HOURS*t['r0273'], tumor_IMP_salvage_pmol=-HOURS*t['r0275'],
        tumor_IMP_material_pmol=HOURS*t['DM_IMP_MATERIAL'],
        tumor_Ser_task_pmol=HOURS*t['DM_SER_c'], tumor_C1_task_pmol=HOURS*t['DM_BOOKKEEP_C1'],
        tumor_ATP_paid_pmol=HOURS*atp_source,
        tumor_net_Ser_assembly_pmol=HOURS*(t['r0230']+t['CAP_SER_m_c']),
        glucose_uptake_pmol=HOURS*gross[idx['glucose']], oxygen_uptake_pmol=HOURS*gross[idx['oxygen']],
        supplied_oxygen_pmol=initial[idx['oxygen']]+external[idx['oxygen']],
        final_serine_pmol=stock[idx['serine']], final_HX_pmol=stock[idx['hypoxanthine']],
        max_Sv=residual, max_bound_error=max(0., bound_error), max_tumor_CNP_error=float(abs(cn_p).max()),
        max_equation_error=eq) | errors
    return metrics, trace, inventory


def execute(spec, fields, template, native, curated, q):
    joint, initial, idx = prepare(spec, fields, template, native, curated, q)
    flux, ranges = solve(joint, spec)
    if flux is None:
        return spec | dict(status='infeasible_fixed_demand'), []
    theta = float(joint.theta.primal)
    metrics, trace, inventory = validate(joint, spec, fields, initial, idx, flux, theta)
    refined, _, _ = validate(joint, spec, fields, initial, idx, flux, theta, dt=.05)
    assert abs(metrics['final_serine_pmol']-refined['final_serine_pmol']) < TOL
    out = HERE / 'outputs' / spec['case']
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(out/'trajectory.csv', index=False)
    pd.DataFrame(inventory).to_csv(out/'stock_balances.csv', index=False)
    pd.DataFrame([dict(group=g, reaction=rid, flux=value,
        lb=joint.r[g][rid].lower_bound, ub=joint.r[g][rid].upper_bound,
        population_factor=joint.factor[g]) for g in flux for rid, value in flux[g].items()]).to_csv(out/'fluxes.csv', index=False)
    return spec | dict(status='optimal') | metrics, ranges


def controls(spec, fields, template, native, curated, q):
    rows = []
    for mode in ['native_Q', 'terminal_noATP']:
        joint, _, _ = prepare(spec | dict(electron_mode=mode), fields, template, native, curated, q)
        joint.ha.lb, joint.ha.ub = 0., 1.
        joint.ta.lb, joint.ta.ub = 0., 1.
        joint.theta.lb = joint.theta.ub = 0.
        for con in joint.budget.values():
            con.ub = 0.
        m = joint.model
        m.objective = m.problem.Objective(joint.ha+joint.ta, direction='max')
        value = float(m.slim_optimize(error_value=None))
        assert value < TOL
        rows.append(dict(control='all_resources_closed', electron_mode=mode,
            maintenance_capacity=value))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    specs = plan()
    if args.smoke:
        specs = [specs[i] for i in [0, 3, 8, 11]]
    previous = json.loads((ROOT/'notebooks/serine_public_discriminator_audit_2026_10_24/frozen_model_contract.json').read_text())['protected']
    assert all(shared.sha(ROOT/path.replace('\\', '/')) == h for path, h in previous.items())
    paths = [Path(shared.__file__), Path(purine.__file__)]
    paths += list((ROOT/'notebooks/serine_purine_material_2026_10_25').glob('*.csv'))
    paths += [ROOT/'notebooks/serine_purine_material_2026_10_25/verification.json']
    protected = previous | {str(path.relative_to(ROOT)): shared.sha(path) for path in paths}
    fields = fields_table()
    fields.to_csv(HERE/'fields.csv', index=False)
    suffix = '_smoke' if args.smoke else ''
    protocol = dict(cases=specs, protected=protected, populations=dict(basal=252, local=12, tumor=7),
        volume_nL=20, horizon_h=12, matched_service=.55,
        scope='Fixed populations, well-mixed common finite pools, constant 12h allocation; no growth or signalling',
        demand='Same .014 IMP and .056 Ser per hour at unit service; residual non-purine C1 0 or .014 * service',
        matched='Same absolute demands at .55 service; infeasibility retained. Minimize the single open local source channel before glucose/O2/L1',
        source='Local Ser/formate credited once in their own pools; no prior maximization or forced secretion',
        new_fields='Ribose and HX access restricted to tumour assay ports; no assertion of biological tumour specificity',
        oxygen='4 initially + finite 32 reservoir; same target and feed ceiling; actual delivery tracked',
        caveats=['Inherited placeholder amino acids and hypothetical ATP demands',
            'Ribose access and DMG matrix access uncalibrated',
            'Effective NADPH oxidation, not a specific human enzyme',
            'Purine-material task is IMP, not complete ATP/GTP, DNA/RNA synthesis',
            'CN checked for community; CNP moiety balance checked for tumour, not a complete hepatic H/O/charge audit'],
        experimental_fit=False, active_PhysiCell_modified=False)
    (HERE/('protocol'+suffix+'.json')).write_text(json.dumps(protocol, indent=2)+'\n')
    native = cobra.io.read_sbml_model(str(shared.engine.previous.NATIVE))
    template = shared.engine.build(native)
    curated = native.copy()
    shared.donor.apply_published(curated)
    q = json.loads(shared.engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    pd.DataFrame(controls(specs[0], fields, template, native, curated, q)).to_csv(HERE/('controls'+suffix+'.csv'), index=False)
    rows, ranges = [], []
    for spec in specs:
        row, own_ranges = execute(spec, fields, template, native, curated, q)
        rows.append(row)
        ranges += own_ranges
        pd.DataFrame(rows).to_csv(HERE/('results'+suffix+'.csv'), index=False)
        print(spec['case'], spec['panel'], spec['channel'], spec['HX_pmol'], row['status'],
            round(row.get('task_h', 0.), 6), flush=True)
    pd.DataFrame(ranges).to_csv(HERE/('route_FVA'+suffix+'.csv'), index=False)
    assert all(shared.sha(ROOT/path.replace('\\', '/')) == h for path, h in protected.items())
    data = pd.DataFrame(rows)
    good = data[data.status == 'optimal']
    result = dict(cases=len(data), optimal=len(good), fixed_demand_infeasible=len(data)-len(good),
        max_CN_error=float(good.max_CN_error.max()), max_inventory_error=float(good.max_inventory_error.max()),
        max_pool_excess_pmol=float(good.max_pool_excess_pmol.max()),
        max_tumor_CNP_error=float(good.max_tumor_CNP_error.max()), protected_unchanged=True,
        no_experimental_fit=True, no_growth_inference=True)
    (HERE/('verification'+suffix+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
