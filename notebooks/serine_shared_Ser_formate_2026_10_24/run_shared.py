"""Separate Ser and formate support in a finite, well-mixed community.

Task maximization is a community capacity bound, not induced secretion,
growth, a measured allocation policy, or a calibrated tumour reconstruction.
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
sys.path.insert(0, str(ROOT / 'notebooks/serine_local_glycine_limit_2026_10_23'))
from run_limit import setup, engine
sys.path.insert(0, str(ROOT / 'notebooks/serine_methyl_donor_curation_2026_10_24'))
import run_downstream as donor

HOURS = 12.0
CHANNELS = ['both', 'serine_only', 'formate_only', 'closed']
SER = 'EX_HC00068_s'
FOR = 'EX_HC00061_s'


def sha(path):
    return donor.sha(path)


def fields_table():
    fields = pd.read_csv(ROOT / 'notebooks/serine_redox_coculture_2026_10_22/fields.csv')
    fields = fields.fillna({'cobra_exchange': ''})
    fields.loc[fields.substrate == 'formate', 'cobra_exchange'] = FOR
    additions = pd.DataFrame([
        dict(substrate='diagnostic_DMG', cobra_exchange='ASSAY_DMG_m', carbon_atoms=4,
             nitrogen_atoms=1, initial_pmol=0.144),
        dict(substrate='diagnostic_SAR', cobra_exchange='ASSAY_SAR_m', carbon_atoms=3,
             nitrogen_atoms=1, initial_pmol=0.0)])
    return pd.concat([fields, additions], ignore_index=True)


def plan():
    prior = json.loads((ROOT / 'notebooks/serine_hepatic_intervention_comparison_2026_10_23/protocol.json').read_text())
    base = prior['cases'][0] | dict(panel='primary', priority='task_first', arm='funded',
        tumor_alternatives='open', capacity_fraction=1.0, formate_uM=0.0,
        local_Gly_ceiling_pmol_h=0.048, hepatic_background='DMG_GCS_closed',
        donor_stock_pmol=0.144, electron_mode='native_Q', tumor_formate_closed=False,
        policy='glucose_sparing', energy_glucose_cap=0.5, fixed_C1_pmol_h=0.028,
        ser_uM=25.0, gly_uM=100.0, external_rate_pmol_h=0.0,
        hepatic_source_rate_pmol_h=0.0, initial_formate_pmol=0.0,
        tumor_assembly='open', hepatic_PHGDH='available', channel='both')
    rows = []
    for capacity in [0.1, 1.0]:
        for assembly in ['closed', 'open']:
            for phgdh in ['available', 'blocked']:
                for channel in CHANNELS:
                    rows.append(base | dict(capacity_fraction=capacity, tumor_assembly=assembly,
                        hepatic_PHGDH=phgdh, channel=channel))
    # Perturb source architecture, not just the favourable source condition.
    for assembly in ['closed', 'open']:
        for phgdh in ['available', 'blocked']:
            for channel in CHANNELS:
                rows.append(base | dict(panel='GCS_background', hepatic_background='GCS_no_DMG_route',
                    tumor_assembly=assembly, hepatic_PHGDH=phgdh, channel=channel))
    for panel in ['no_DMG', 'C1_zero']:
        for assembly in ['closed', 'open']:
            for channel in CHANNELS:
                rows.append(base | dict(panel=panel, hepatic_PHGDH='blocked',
                    tumor_assembly=assembly, channel=channel,
                    donor_stock_pmol=0.0 if panel == 'no_DMG' else 0.144,
                    fixed_C1_pmol_h=0.0 if panel == 'C1_zero' else 0.028))
    for panel in ['oxygen_sparing', 'low_tumor_glucose']:
        for assembly in ['closed', 'open']:
            for channel in ['both', 'closed']:
                rows.append(base | dict(panel=panel, hepatic_PHGDH='blocked',
                    tumor_assembly=assembly, channel=channel,
                    policy='oxygen_sparing' if panel == 'oxygen_sparing' else 'glucose_sparing',
                    energy_glucose_cap=0.02 if panel == 'low_tumor_glucose' else 0.5))
    for panel, serine, formate, assembly in [
            ('external_formate_control', 25.0, 0.336, 'closed'),
            ('external_serine_control', 100.0, 0.0, 'closed'),
            ('no_Ser_formate_no_assembly', 0.0, 0.336, 'closed'),
            ('no_Ser_formate_assembly', 0.0, 0.336, 'open')]:
        rows.append(base | dict(panel=panel, hepatic_PHGDH='blocked', channel='closed',
            ser_uM=serine, initial_formate_pmol=formate, tumor_assembly=assembly))
    for cap in [0.1, 1.0]:
        rows.append(base | dict(panel='tumor_formate_uptake_closed', hepatic_PHGDH='blocked',
            channel='formate_only', tumor_assembly='closed', capacity_fraction=cap,
            tumor_formate_closed=True))
    for nutrient in ['serine', 'formate']:
        rows.append(base | dict(panel='equal_amount_' + nutrient, hepatic_PHGDH='blocked',
            channel='closed', tumor_assembly='closed',
            ser_uM=32.2 if nutrient == 'serine' else 25.0,
            initial_formate_pmol=0.144 if nutrient == 'formate' else 0.0))
    for assembly in ['closed', 'open']:
        for channel in ['both', 'closed']:
            rows.append(base | dict(panel='electron_closure', hepatic_PHGDH='blocked',
                tumor_assembly=assembly, channel=channel, electron_mode='terminal_noATP'))
    assert len(rows) == 84
    return [row | dict(case=f'shared_{i:03d}') for i, row in enumerate(rows)]


def prepare(spec, fields, template, native, native_curated, q):
    # Reuse paid tumour demands, population factors and maintenance constraints.
    s = spec | dict(arm='PHGDH' if spec['hepatic_PHGDH'] == 'blocked' else 'funded')
    joint, initial, idx, total = setup(s, fields, template, native, q)
    initial[idx['diagnostic_DMG']] = spec['donor_stock_pmol']
    initial[idx['formate']] = spec['initial_formate_pmol']
    total = initial.copy()
    total[idx['oxygen']] = 36.0  # 4 initially + at most 32 from a finite reservoir.
    local = joint.groups['local'].copy()
    for rid, rxn in joint.r['local'].items():
        local.reactions.get_by_id(rid).bounds = rxn.bounds
    local = donor.build_disposal(local, native_curated, spec['electron_mode'])
    local.reactions.get_by_id(SER).bounds = (0.0, 0.2 if spec['channel'] in ['both', 'serine_only'] else 0.0)
    local.reactions.get_by_id(FOR).bounds = (0.0, 1000.0 if spec['channel'] in ['both', 'formate_only'] else 0.0)
    local.reactions.get_by_id('ASSAY_DMG_m').bounds = (-spec['donor_stock_pmol'] / (HOURS * joint.factor['local']), 0.0)
    local.reactions.get_by_id('ASSAY_SAR_m').bounds = (0.0, 1000.0)
    if spec['hepatic_background'] == 'DMG_GCS_closed':
        local.reactions.get_by_id('r0295').bounds = (0.0, 0.0)
    else:
        for rid in ['CUR_DMGDH', 'CUR_SARDH']:
            local.reactions.get_by_id(rid).bounds = (0.0, 0.0)
    joint.groups['local'] = local.copy()
    ids = local.reactions.list_attr('id')
    for met in local.metabolites:
        met.id = 'local__' + met.id
    for rxn in local.reactions:
        rxn.id = 'local__' + rxn.id
    local.repair()
    m = joint.model
    m.add_reactions([r.copy() for r in local.reactions if r.id not in m.reactions])
    for original in ids:
        m.reactions.get_by_id('local__' + original).bounds = local.reactions.get_by_id('local__' + original).bounds
    joint.r['local'] = {rid: m.reactions.get_by_id('local__' + rid) for rid in ids}
    tumor = joint.r['tumor']
    for rid in ['r0353', 'r0199', 'r0479', 'r0343', 'r0227']:
        tumor[rid].bounds = (0.0, 0.14 * spec['capacity_fraction'])
    tumor['EX_E_GLC'].bounds = (-spec['energy_glucose_cap'], 0.0)
    if spec['tumor_assembly'] == 'closed':
        tumor['r0230'].upper_bound = 0.0  # Ser consumption for C1 remains allowed.
        tumor['CAP_SER_m_c'].bounds = (0.0, 0.0)
    if spec['tumor_formate_closed']:
        tumor['EX_FOR_c'].bounds = (0.0, 0.0)

    # Credit each hepatic secretion exactly once in its own shared resource pool.
    m.remove_cons_vars(list(joint.budget.values()))
    joint.gross, joint.net, joint.budget = {}, {}, {}
    for field in fields.itertuples():
        gross, net = 0.0, 0.0
        for group in joint.r:
            rid = engine.ROLES.get(field.substrate) if group == 'tumor' else field.cobra_exchange
            if rid in joint.r[group]:
                rxn = joint.r[group][rid]
                gross += joint.factor[group] * rxn.reverse_variable
                net += joint.factor[group] * rxn.flux_expression
        credit = 0.0
        if field.substrate in ['serine', 'formate']:
            credit = joint.factor['local'] * joint.r['local'][field.cobra_exchange].flux_expression
        con = m.problem.Constraint(gross - credit, ub=float(total[idx[field.substrate]]) / HOURS,
                                   name='separate_shared_' + field.substrate)
        m.add_cons_vars(con)
        joint.gross[field.substrate], joint.net[field.substrate], joint.budget[field.substrate] = gross, net, con
    for face in joint.faces.values():
        face.lb = face.ub = None
    joint.ids = {g: list(joint.r[g]) for g in joint.r}
    joint.S = {g: create_stoichiometric_matrix(joint.groups[g]) for g in joint.r}
    joint.l1 = {var: joint.factor[g] for g in joint.r for r in joint.r[g].values()
                for var in [r.forward_variable, r.reverse_variable]}
    m.solver.configuration.tolerances.feasibility = 1e-9
    m.solver.configuration.timeout = 45
    return joint, initial, idx, total


def solve(joint, spec):
    m, p = joint.model, joint.model.problem
    m.objective = p.Objective(joint.theta, direction='max')
    value = m.slim_optimize()
    if m.solver.status != 'optimal':
        return None, []
    joint.faces['tasks'].lb = joint.faces['tasks'].ub = float(value)
    variability = []
    if (spec['panel'] == 'primary' and spec['capacity_fraction'] == 1.0
            and spec['hepatic_PHGDH'] == 'blocked'):
        reactions = [joint.r['local'][SER].id, joint.r['local'][FOR].id,
                     joint.r['tumor']['EX_FOR_c'].id, joint.r['tumor']['EX_SER_c'].id]
        ranges = flux_variability_analysis(m, reactions, fraction_of_optimum=1.0, processes=1)
        for rid, values in ranges.iterrows():
            group = 'local' if rid.startswith('local__') else 'tumor'
            variability.append(dict(case=spec['case'], reaction=rid,
                minimum_pmol=float(values['minimum'] * joint.factor[group] * HOURS),
                maximum_pmol=float(values['maximum'] * joint.factor[group] * HOURS)))
    order = ['glucose', 'oxygen'] if spec['policy'] == 'glucose_sparing' else ['oxygen', 'glucose']
    for key in order:
        m.objective = p.Objective(joint.gross[key], direction='min')
        amount = float(m.slim_optimize(error_value=None))
        joint.faces[key].lb = max(0.0, amount - 1e-9)
        joint.faces[key].ub = max(0.0, amount) + 1e-9
    m.objective = p.Objective(0.0, direction='min')
    m.objective.set_linear_coefficients(joint.l1)
    m.slim_optimize(error_value=None)
    return {g: {rid: float(r.flux) for rid, r in joint.r[g].items()} for g in joint.r}, variability


def validate_and_replay(joint, spec, fields, initial, idx, fluxes, dt=0.1, theta=None):
    theta = float(joint.theta.primal) if theta is None else theta
    residual, bound_error = 0.0, 0.0
    gross, net = np.zeros(len(fields)), np.zeros(len(fields))
    group_net = {g: np.zeros(len(fields)) for g in joint.r}
    for g in joint.r:
        v = np.array([fluxes[g][rid] for rid in joint.ids[g]])
        residual = max(residual, float(abs(joint.S[g] @ v).max()))
        for rid, value in zip(joint.ids[g], v):
            rxn = joint.r[g][rid]
            bound_error = max(bound_error, rxn.lower_bound - value, value - rxn.upper_bound)
    for j, field in enumerate(fields.itertuples()):
        for g in joint.r:
            rid = engine.ROLES.get(field.substrate) if g == 'tumor' else field.cobra_exchange
            if rid in fluxes[g]:
                amount = joint.factor[g] * fluxes[g][rid]
                group_net[g][j] = amount
                net[j] += amount
                gross[j] += max(0.0, -amount)
    t, local = fluxes['tumor'], fluxes['local']
    task = t['DM_SER_c']
    c1 = t['DM_BOOKKEEP_C1']
    equation_error = max(abs(task - 0.056 * theta), abs(c1 - spec['fixed_C1_pmol_h']),
        abs(t['TASK_ATP'] - 0.0035 - spec['work_ATP_at_full_service_pmol_h'] * theta),
        abs(2 * t['PAID_GLYCOLYSIS'] + spec['yield_ATP'] * t['PAID_OXIDATION'] - t['TASK_ATP'] - t['r0228'] - t['r0353']),
        abs(2 * t['r0199'] + t['r0293'] + t['r0227']), abs(t['r0295'] - t['PAID_NADH_OX_m']),
        abs(-t['EX_SER_c'] + t['r0230'] + t['CAP_SER_m_c'] - task),
        abs(local['CUR_DMGDH'] + local['CUR_SARDH'] - local['CUR_ETF_REOX']))
    assert max(residual, bound_error, equation_error) < 1e-7
    assert max(abs(t[r]) for r in ['BG_ATP', 'BG_NADPH', 'BG_NADH_c', 'BG_NADH_m']) < 1e-9
    assert local[SER] > -1e-8 and local[FOR] > -1e-8
    assert all(abs(fluxes[g]['CULTURE_ATPM'] - 5.0) < 1e-7 for g in ['basal', 'local'])
    credit = np.zeros(len(fields))
    for name, rid in [('serine', SER), ('formate', FOR)]:
        credit[idx[name]] = joint.factor['local'] * local[rid]
    stock, external = initial.copy(), np.zeros(len(fields))
    oxygen_reservoir = 32.0
    traces = []
    max_pool = max_inventory = max_cn = 0.0
    for k in range(round(HOURS / dt)):
        # O2 feed is applied after each interval, as in the previous finite-reservoir protocol.
        # Compare resource amounts (pmol), not a rate error amplified by 1/dt.
        pool = float(np.maximum(0.0, gross * dt - stock - credit * dt).max())
        stock += dt * net
        feed = min(max(0.0, 4.0 - stock[idx['oxygen']]), 3.0 * dt, oxygen_reservoir)
        oxygen_reservoir -= feed
        stock[idx['oxygen']] += feed
        external[idx['oxygen']] += feed
        time = (k + 1) * dt
        balance = stock - initial - external - time * net
        cn = (stock - initial - external) @ fields[['carbon_atoms', 'nitrogen_atoms']].to_numpy()
        cn += time * np.array([3 * task + c1, task])
        assert stock.min() > -1e-7 and max(pool, abs(balance).max(), abs(cn).max()) < 1e-7, (
            spec['case'], dt, k, fields.iloc[int(stock.argmin())].substrate,
            float(stock.min()), pool, float(abs(balance).max()), float(abs(cn).max()))
        max_pool, max_inventory, max_cn = max(max_pool, pool), max(max_inventory, float(abs(balance).max())), max(max_cn, float(abs(cn).max()))
        traces.append(dict(time_h=time, task_h=time * theta, serine_pmol=stock[idx['serine']],
            formate_pmol=stock[idx['formate']], glycine_pmol=stock[idx['glycine']], oxygen_pmol=stock[idx['oxygen']],
            hepatic_Ser_export_pmol=time * credit[idx['serine']],
            hepatic_formate_export_pmol=time * credit[idx['formate']],
            tumor_formate_net_uptake_pmol=-time * t['EX_FOR_c'],
            oxygen_reservoir_remaining_pmol=oxygen_reservoir,
            CN_error=float(abs(cn).max()), inventory_error=float(abs(balance).max()), pool_excess_pmol=pool))
    inventory = [dict(substrate=f.substrate, initial_pmol=initial[j],
        external_pmol=external[j], final_pmol=stock[j],
        basal_change_pmol=HOURS * group_net['basal'][j],
        local_change_pmol=HOURS * group_net['local'][j],
        tumor_change_pmol=HOURS * group_net['tumor'][j]) for j, f in enumerate(fields.itertuples())]
    metrics = dict(task_h=HOURS * theta, hepatic_Ser_export_pmol=HOURS * credit[idx['serine']],
        hepatic_formate_export_pmol=HOURS * credit[idx['formate']],
        basal_Ser_uptake_pmol=-HOURS * min(0.0, group_net['basal'][idx['serine']]),
        tumor_Ser_net_uptake_pmol=-HOURS * t['EX_SER_c'],
        tumor_formate_net_uptake_pmol=-HOURS * t['EX_FOR_c'],
        tumor_net_Ser_assembly_pmol=HOURS * (t['r0230'] + t['CAP_SER_m_c']),
        tumor_Ser_task_pmol=HOURS * task, tumor_C1_task_pmol=HOURS * c1,
        local_DMG_uptake_pmol=-HOURS * joint.factor['local'] * local['ASSAY_DMG_m'],
        final_Ser_pmol=stock[idx['serine']], final_formate_pmol=stock[idx['formate']],
        total_glucose_uptake_pmol=HOURS * gross[idx['glucose']],
        total_oxygen_uptake_pmol=HOURS * gross[idx['oxygen']],
        supplied_oxygen_pmol=initial[idx['oxygen']] + external[idx['oxygen']],
        max_Sv=residual, max_bound_error=max(0.0, bound_error), max_equation_error=equation_error,
        max_pool_excess_pmol=max_pool, max_inventory_error=max_inventory, max_CN_error=max_cn)
    return metrics, traces, inventory


def community_controls(spec, fields, template, native, curated, q):
    rows = []
    for mode in ['native_Q', 'terminal_noATP']:
        control = spec | dict(electron_mode=mode, fixed_C1_pmol_h=0.0, donor_stock_pmol=0.0)
        joint, _, _, _ = prepare(control, fields, template, native, curated, q)
        joint.ha.lb = joint.ta.lb = 0.0
        joint.theta.lb = joint.theta.ub = 0.0
        for con in joint.budget.values():
            con.ub = 0.0
        m = joint.model
        m.objective = m.problem.Objective(joint.ha + joint.ta, direction='max')
        value = m.slim_optimize()
        assert m.solver.status == 'optimal' and abs(value) < 1e-7
        rows.append(dict(electron_mode=mode, control='no_resources_optional_maintenance',
            status=m.solver.status, total_maintenance_capacity=float(value)))
    return rows


def execute(spec, fields, template, native, curated, q):
    joint, initial, idx, total = prepare(spec, fields, template, native, curated, q)
    fluxes, ranges = solve(joint, spec)
    if fluxes is None:
        return spec | dict(status=joint.model.solver.status), []
    metrics, trace, inventory = validate_and_replay(joint, spec, fields, initial, idx, fluxes)
    # Check the same constant allocation with half the replay step, not a kinetic convergence claim.
    refined, _, _ = validate_and_replay(joint, spec, fields, initial, idx, fluxes, dt=0.05)
    assert abs(metrics['final_Ser_pmol'] - refined['final_Ser_pmol']) < 1e-7
    assert abs(metrics['final_formate_pmol'] - refined['final_formate_pmol']) < 1e-7
    folder = HERE / 'outputs' / spec['case']
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(folder / 'trajectory.csv', index=False)
    pd.DataFrame(inventory).to_csv(folder / 'stock_balances.csv', index=False)
    pd.DataFrame([dict(group=g, reaction=rid, flux=value,
        lb=joint.r[g][rid].lower_bound, ub=joint.r[g][rid].upper_bound,
        population_factor=joint.factor[g], equation=joint.r[g][rid].reaction)
        for g in fluxes for rid, value in fluxes[g].items()]).to_csv(folder / 'fluxes.csv', index=False)
    return spec | dict(status='optimal') | metrics, ranges


def contrasts(data):
    rows = []
    keys = ['panel', 'capacity_fraction', 'tumor_assembly', 'hepatic_PHGDH',
            'hepatic_background', 'fixed_C1_pmol_h', 'policy']
    for key, part in data.groupby(keys):
        if set(part.channel) != set(CHANNELS) or len(part) != 4 or not part.status.eq('optimal').all():
            continue
        p = part.set_index('channel')
        baseline, both = p.loc['closed'].task_h, p.loc['both'].task_h
        rows.append(dict(zip(keys, key)) | dict(closed_task_h=baseline, both_task_h=both,
            serine_only_task_h=p.loc['serine_only'].task_h,
            formate_only_task_h=p.loc['formate_only'].task_h,
            total_channel_benefit_h=both - baseline,
            loss_closing_Ser_h=both - p.loc['formate_only'].task_h,
            loss_closing_formate_h=both - p.loc['serine_only'].task_h,
            nonadditivity_h=both - p.loc['serine_only'].task_h - p.loc['formate_only'].task_h + baseline))
    pd.DataFrame(rows).to_csv(HERE / 'channel_contrasts.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    fields = fields_table()
    fields.to_csv(HERE / 'fields.csv', index=False)
    specs = plan()
    if args.smoke:
        specs = [s for s in specs if s['panel'] == 'primary' and s['capacity_fraction'] == 1.0
                 and s['hepatic_PHGDH'] == 'blocked' and s['tumor_assembly'] == 'closed']
    paths = [Path(__file__), HERE / 'fields.csv',
        ROOT / 'config/PhysiCell_settings.xml', ROOT / 'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT / 'custom_modules/custom.cpp', ROOT / 'addons/dFBA/src/dfba_intracellular.cpp',
        ROOT / 'config/hepatonet_original_13_08.xml',
        ROOT / 'notebooks/serine_methyl_donor_curation_2026_10_24/run_downstream.py',
        ROOT / 'notebooks/serine_local_glycine_limit_2026_10_23/run_limit.py',
        ROOT / 'notebooks/serine_paid_redox_2026_10_22/redox_model.py']
    paths += [engine.previous.hepatic.MAP / ('map_14_' + group + '.xml') for group in ['basal', 'local']]
    paths.append(engine.previous.PARKER)
    # Capture the helper implementations actually imported by this workflow.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'notebooks'):
                paths.append(path)
    protected = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protocol = dict(cases=specs, protected=protected, declared_before_execution=True,
        populations=dict(basal=252, local=12, tumor=7), volume_nL=20.0,
        primary_medium='Ser 25 uM, Gly 100 uM, formate zero, matrix-accessible DMG 0.144 pmol; inherited other fields',
        oxygen='4 pmol initially plus up to 32 pmol from reservoir; target 4, feed ceiling 3 pmol/h',
        allocation='Maximize tumour task at fixed maintenance/C1; then minimize gross glucose, oxygen, L1. No prior maximization of hepatic export',
        shared_pools='Ser and formate gross uptake minus their own local secretion, credited once; other pools gross-budgeted conservatively',
        capacities='0.1 and 1 of inherited exploratory tumour capacities, not measured PATU ranges',
        scope='Fixed populations, constant 12h allocation with stock replay; no growth, signalling, isotope attribution or experimental calibration',
        caveats=['DMG matrix access assumed, not physiological plasma transport',
                 'Inherited medium placeholders and energy/C1 demands remain uncalibrated',
                 'FVA ranges are model alternatives, not experimental error bars'])
    suffix = '_smoke' if args.smoke else ''
    (HERE / ('protocol' + suffix + '.json')).write_text(json.dumps(protocol, indent=2) + '\n')
    native = cobra.io.read_sbml_model(str(engine.previous.NATIVE))
    template = engine.build(native)
    curated = native.copy()
    donor.apply_published(curated)
    q = json.loads(engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    controls = community_controls(specs[0] | dict(channel='both'), fields, template, native, curated, q)
    pd.DataFrame(controls).to_csv(HERE / ('community_controls' + suffix + '.csv'), index=False)
    rows, ranges = [], []
    for spec in specs:
        row, own_ranges = execute(spec, fields, template, native, curated, q)
        rows.append(row)
        ranges += own_ranges
        pd.DataFrame(rows).to_csv(HERE / ('results' + suffix + '.csv'), index=False)
        print(spec['case'], spec['panel'], spec['channel'], spec['hepatic_PHGDH'],
              row['status'], round(row.get('task_h', 0.0), 6), flush=True)
    data = pd.DataFrame(rows)
    pd.DataFrame(ranges).to_csv(HERE / ('route_FVA' + suffix + '.csv'), index=False)
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    good = data[data.status == 'optimal']
    if not args.smoke:
        contrasts(data)
    verification = dict(cases=len(data), optimal=len(good), community_controls=len(controls), protected_unchanged=True,
        max_CN_error=float(good.max_CN_error.max()), max_inventory_error=float(good.max_inventory_error.max()),
        max_pool_excess_pmol=float(good.max_pool_excess_pmol.max()), max_Sv=float(good.max_Sv.max()),
        growth_inferred=False, experimental_rates_fitted=False)
    (HERE / ('verification' + suffix + '.json')).write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
