"""Precursor-access and matched-support controls for poster result selection.

Separate diagnostic copies only. No active PhysiCell, native SBML, prior source
or parameter fit is modified. Same tasks are not interpreted as cell growth.
"""
import json
import logging
from pathlib import Path
import sys

import cobra
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'notebooks/serine_tumour_SSP_control_2026_10_25'))
import run_ssp as ssp
sys.path.insert(0, str(ROOT/'notebooks/serine_purine_community_2026_10_25'))
import run_community as community

HOURS, TOL = 12., 1e-7
SOURCES = ['external_ribose', 'glucose_PPP', 'combined']
SALVAGE = [('no_HX', 0., 'available'), ('HX_HPRT', .2, 'available'), ('HX_HPRT_blocked', .2, 'closed')]


def isolated_plan():
    base = ssp.plan()[0]
    rows = []
    salvage = [('no_HX', 0., 'available'), ('HX_HPRT', .1, 'available'), ('HX_HPRT_blocked', .1, 'blocked')]
    for access in SOURCES:
        for ser in [.5, 2.8]:
            for capacity in ['available', 'blocked']:
                for nitrogen in ['glutamate', 'glutamine']:
                    for label, hx, hprt in salvage:
                        rows.append(base | dict(panel='isolated_capacity', precursor_access=access,
                            external_Ser_pmol=ser, SSP=capacity, nitrogen=nitrogen,
                            salvage_condition=label, HX_pmol=hx, HPRT=hprt, fixed_service=None))
    for access in SOURCES:
        for capacity in ['available', 'blocked']:
            for nitrogen in ['glutamate', 'glutamine']:
                for label, hx, hprt in salvage:
                    rows.append(base | dict(panel='isolated_matched_tasks', precursor_access=access,
                        external_Ser_pmol=.5, SSP=capacity, nitrogen=nitrogen,
                        salvage_condition=label, HX_pmol=hx, HPRT=hprt, fixed_service=.4,
                        selection='serine_sparing'))
    assert len(rows) == 108
    return [s | dict(case=f'isolated_{i:03d}') for i, s in enumerate(rows)]


def community_plan():
    base = community.plan()[0] | dict(residual_C1=.014, channel='serine_only')
    rows = []
    for access in SOURCES:
        for label, hx, hprt in SALVAGE:
            for phgdh in ['available', 'blocked']:
                for channel in ['serine_only', 'closed']:
                    for policy in ['glucose_sparing', 'oxygen_sparing']:
                        for fixed in [None, .55]:
                            rows.append(base | dict(panel='paired_support', precursor_access=access,
                                salvage_condition=label, HX_pmol=hx, HPRT=hprt,
                                hepatic_PHGDH=phgdh, channel=channel, policy=policy,
                                fixed_service=fixed,
                                selection='resource_sparing' if fixed is None else 'source_sparing'))
    # Negative donor control under the same fixed target. Infeasibility is a
    # retained resource result, not repaired by relaxing maintenance or demand.
    for access in SOURCES:
        for label, hx, hprt in SALVAGE:
            for channel in ['serine_only', 'closed']:
                rows.append(base | dict(panel='donor_negative_control', precursor_access=access,
                    salvage_condition=label, HX_pmol=hx, HPRT=hprt, hepatic_PHGDH='blocked',
                    channel=channel, fixed_service=.55, selection='source_sparing', donor_stock_pmol=0.))
    assert len(rows) == 162
    return [s | dict(case=f'paired_{i:03d}') for i, s in enumerate(rows)]


def access_bounds(model, mode):
    assert mode in SOURCES
    ppp = mode != 'external_ribose'
    model.reactions.r0353.bounds = (0., .14)
    # HK also feeds native upper glycolysis in the SSP-positive control.
    # Do not close or double its inherited shared .14 capacity.
    for rid in ['r0199', 'r0479', 'r0343']:
        model.reactions.get_by_id(rid).bounds = (0., .14 if ppp else 0.)
    model.reactions.r0250.bounds = (-.14, 0.) if ppp else (0., 0.)
    model.reactions.r0248.bounds = (0., .14 if mode != 'glucose_PPP' else 0.)
    if mode == 'glucose_PPP':
        model.reactions.ASSAY_RIBOSE_c.bounds = (0., 0.)


def build_isolated(spec, native, template, q):
    selector = ssp.build(spec, native, template, q)
    initial = ssp.configure_ports(selector, spec)
    access_bounds(selector.model, spec['precursor_access'])
    if spec['fixed_service'] is not None:
        selector.theta.lb = selector.theta.ub = spec['fixed_service']
    return selector, initial


def solve_isolated(selector, spec):
    m, p = selector.model, selector.model.problem
    def optimize(expression, direction):
        m.objective = p.Objective(expression, direction=direction)
        return float(m.slim_optimize(error_value=None))
    def fix_nonnegative(expression, value, name):
        assert value >= -TOL, (name, value)
        value = max(0., value)
        m.add_cons_vars(p.Constraint(expression, lb=max(0., value-1e-9), ub=value+1e-9, name=name))
    alpha = optimize(selector.alpha, 'max')
    m.add_cons_vars(p.Constraint(selector.alpha, lb=alpha, ub=alpha, name='poster_fixed_maintenance'))
    theta = optimize(selector.theta, 'max')
    m.add_cons_vars(p.Constraint(selector.theta, lb=theta, ub=theta, name='poster_fixed_tasks'))
    if spec['fixed_service'] is not None:
        # Gross entry is nonnegative even if the SSP-positive copy can export.
        # Minimizing -net exchange would instead reward extra Ser secretion.
        expression = m.reactions.EX_SER_c.reverse_variable
        fix_nonnegative(expression, optimize(expression, 'min'), 'poster_minimal_gross_Ser_entry')
    order = ['glucose', 'oxygen'] if spec['policy'] == 'glucose_sparing' else ['oxygen', 'glucose']
    for role in order:
        expression = -m.reactions.get_by_id(ssp.PORTS[role]).flux_expression
        fix_nonnegative(expression, optimize(expression, 'min'), 'poster_resource_'+role)
    optimize(sum(r.forward_variable+r.reverse_variable for r in m.reactions), 'min')
    return alpha, theta, {r.id: float(r.flux) for r in m.reactions}


def validate_isolated(selector, spec, initial, flux, alpha, theta):
    m = selector.model
    sv = float(abs(create_stoichiometric_matrix(m)@np.array([flux[r.id] for r in m.reactions])).max())
    bounds = max(max(r.lower_bound-flux[r.id], flux[r.id]-r.upper_bound, 0.) for r in m.reactions)
    cnp = sum((sum((ssp.atoms(met)*coef for met, coef in r.metabolites.items()), start=np.zeros(3))*flux[r.id]
        for r in m.boundary), start=np.zeros(3))
    made = 2*flux['PAID_GLYCOLYSIS']+24*flux['PAID_OXIDATION']
    used = sum(-r.metabolites.get(m.metabolites.HC00012_c, 0.)*flux[r.id] for r in m.reactions
        if r.id not in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION'])
    synthesis, ppp = flux['r0159'], flux['r0199']
    ribose = -flux['ASSAY_RIBOSE_c']
    imp = flux['DM_IMP_MATERIAL']
    eqs = [flux['r0338']-synthesis, -flux['r0663']-synthesis,
        flux['r0253']-synthesis, -flux['r0337']-synthesis,
        flux['r0353']-.5*synthesis-ppp, flux['GLUCOSE_TO_PPP']-.5*synthesis-ppp,
        flux['r0736']-.5*synthesis, flux['r0256']-.5*synthesis,
        -flux['r0243']-.5*synthesis, flux['r0479']-ppp, flux['r0343']-ppp,
        -flux['r0250']-ppp, flux['r0248']-ribose, ppp+ribose-imp,
        flux['r0253']+flux['r0338']-flux['ASSAY_NADH_OX_c']+flux['r0171'],
        2*ppp+flux['r0293']+flux['r0227']-flux['ASSAY_NADPH_OX'],
        flux['EX_AKG_c']-synthesis, -flux['r0273']-flux['r0275']-imp,
        flux['DM_SER_c']-.056*theta, flux['DM_BOOKKEEP_C1']-.014*theta,
        imp-.014*theta, flux['TASK_ATP']-.0035*alpha-ssp.prior.WORK_ATP*theta,
        -flux['EX_E_GLC']-flux['PAID_GLYCOLYSIS']-flux['PAID_OXIDATION']-flux['GLUCOSE_TO_PPP'],
        -flux['ASSAY_GLN_c']-flux['r0078']+2*flux['r0273'], flux['EX_NH3_m']-flux['r0078']]
    eq = max(abs(v) for v in eqs)
    assert max(sv, bounds, abs(cnp).max(), abs(made-used), eq) < TOL, (spec['case'], sv, eq)
    assert max(abs(flux[rid]) for rid in ssp.prior.redox.BACKGROUND) < TOL
    inventory = []
    for dt in [.1, .05]:
        remaining = initial.copy()
        for _ in range(round(HOURS/dt)):
            for role, rid in ssp.PORTS.items():
                remaining[role] += dt*flux[rid]
        for role, value in remaining.items():
            assert value >= -TOL and abs(value-initial[role]-HOURS*flux[ssp.PORTS[role]]) < TOL
            if dt == .1:
                inventory.append(dict(case=spec['case'], substrate=role, initial_pmol=initial[role],
                    exchange_pmol=HOURS*flux[ssp.PORTS[role]], final_pmol=value))
    return dict(task_h=HOURS*theta, maintenance_fraction=alpha,
        serine_uptake_pmol=-HOURS*flux['EX_SER_c'], SSP_serine_pmol=HOURS*synthesis,
        ribose_uptake_pmol=HOURS*ribose, PPP_R5P_pmol=HOURS*ppp,
        glucose_uptake_pmol=-HOURS*flux['EX_E_GLC'], oxygen_uptake_pmol=-HOURS*flux['EX_E_O2'],
        IMP_material_pmol=HOURS*imp, AKG_export_pmol=HOURS*flux['EX_AKG_c'],
        ATP_paid_pmol=HOURS*made, max_Sv=sv, max_bound_error=bounds,
        max_CNP_error=float(abs(cnp).max()), max_equation_error=eq), inventory


def prepare_community(spec, fields, template, native, curated, q):
    mapped = spec | dict(ribose_source='glucose_PPP' if spec['precursor_access'] == 'glucose_PPP' else 'finite_ribose')
    joint, initial, idx = community.prepare(mapped, fields, template, native, curated, q)
    access_bounds(joint.model, spec['precursor_access'])
    # Identical finite medium inventory in all access conditions. In PPP-only,
    # external ribose is present but inaccessible to the diagnostic tumour port.
    initial[idx['ribose']] = .3
    joint.budget['ribose'].ub = .3/HOURS
    for rid, r in joint.r['tumor'].items():
        assert r is joint.model.reactions.get_by_id(rid)
    copy = cobra.Model('tumour_precursor_access_copy')
    copy.solver = 'glpk'
    copy.add_reactions([r.copy() for r in joint.r['tumor'].values()])
    joint.groups['tumor'] = copy
    joint.S['tumor'] = create_stoichiometric_matrix(copy)
    return joint, initial, idx


def gates(native, ssp_template, community_template, fields, curated, q):
    records = []
    for mode in SOURCES:
        for control in ['no_inputs', 'no_energy_routes']:
            spec = isolated_plan()[0] | dict(precursor_access=mode)
            selector, _ = build_isolated(spec, native, ssp_template, q)
            m = selector.model
            selector.theta.lb = selector.theta.ub = 0.
            selector.alpha.lb = selector.alpha.ub = 0.
            m.remove_cons_vars(m.solver.constraints['separate_energy_turnover'])
            if control == 'no_inputs':
                for rid in ssp.PORTS.values():
                    m.reactions.get_by_id(rid).lower_bound = 0.
            else:
                for rid in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION']:
                    m.reactions.get_by_id(rid).bounds = (0., 0.)
            m.objective = 'TASK_ATP'
            m.objective_direction = 'max'
            value = float(m.slim_optimize(error_value=None))
            assert value < TOL
            records.append(dict(module='isolated', precursor_access=mode, control=control, capacity=value))
        spec = community_plan()[0] | dict(precursor_access=mode)
        joint, _, _ = prepare_community(spec, fields, community_template, native, curated, q)
        joint.ha.lb, joint.ha.ub = 0., 1.
        joint.ta.lb, joint.ta.ub = 0., 1.
        joint.theta.lb = joint.theta.ub = 0.
        for con in joint.budget.values():
            con.ub = 0.
        m = joint.model
        m.objective = m.problem.Objective(joint.ha+joint.ta, direction='max')
        value = float(m.slim_optimize(error_value=None))
        assert value < TOL
        records.append(dict(module='community', precursor_access=mode, control='all_resources_closed', capacity=value))
    return records


def protected_inputs():
    protected = json.loads((ROOT/'notebooks/serine_central_carbon_evidence_2026_10_25/protected_hashes.json').read_text())
    folders = ['serine_central_carbon_evidence_2026_10_25', 'serine_purine_community_2026_10_25',
        'serine_support_domain_2026_10_25']
    for name in folders:
        folder = ROOT/'notebooks'/name
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py', '.csv', '.json', '.md']:
                protected[p.relative_to(ROOT).as_posix()] = ssp.prior.sha(p)
    return {p.replace('\\', '/'): h for p, h in protected.items()}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    frozen = protected_inputs()
    assert all(ssp.prior.sha(ROOT/p) == h for p, h in frozen.items())
    a, b = isolated_plan(), community_plan()
    protocol = dict(isolated_cases=a, community_cases=b, protected=frozen,
        script_sha256=ssp.prior.sha(Path(__file__)), declared_before_execution=True,
        scope='Diagnostic isolated SSP-positive control and separate SSP-low community; no growth or signalling',
        common_stock='Ribose0.3pmol in all arms; access changes, not medium carbon inventory; no external3PG',
        access='external_ribose:PPP closed; glucose_PPP:ribose uptake closed; combined:both allowed',
        bounds='Inherited0.14pmol/h branch capacities; HK shared, not doubled; finite stocks and paid ATP/redox',
        energy='Hypothetical2/24ATP per glucose; effective cofactor disposal, no new fit',
        fixed_tasks='isolated theta0.4; community theta0.55, same absolute material/energy tasks within comparisons',
        isolated_source_objective='For fixed tasks minimize gross Ser entry, not negative net exchange; secretion is not rewarded',
        populations='community252basal+12local+7tumour equivalent in20nL,12h; isolated7tumour equivalents',
        hepatic_source='Permitted capacity with diagnostic DMG matrix access; not measured basal or induced secretion',
        preserved_exits='AKG and fumarate exits retained, not closed to create PHGDH dependence',
        precursor_scope='One-pass oxidative PPP only, not complete PPP/TCA/pyrimidines',
        claim='Conditional capacity and minimum source for fixed tasks, not cell growth, viability or drug efficacy',
        active_PhysiCell_changed=False, parameter_fit=False)
    (HERE/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    native = cobra.io.read_sbml_model(str(ROOT/'config/hepatonet_original_13_08.xml'))
    ssp_template = ssp.prior.redox.build(native)
    community_template = community.shared.engine.build(native)
    curated = native.copy()
    community.shared.donor.apply_published(curated)
    q = json.loads((ROOT/'notebooks/serine_tumor_response_audit/uptake_evidence/patu8902_serine_net_flux.json').read_text())['positive_net_consumption_nmol_per_million_cells_h']
    fields = community.fields_table()
    fields.to_csv(HERE/'fields.csv', index=False)
    pd.DataFrame(gates(native, ssp_template, community_template, fields, curated, q)).to_csv(HERE/'energy_gates.csv', index=False)
    rows, vectors, inventories = [], [], []
    for i, spec in enumerate(a):
        selector, initial = build_isolated(spec, native, ssp_template, q)
        alpha, theta, flux = solve_isolated(selector, spec)
        metrics, inv = validate_isolated(selector, spec, initial, flux, alpha, theta)
        rows.append(spec | dict(status='optimal') | metrics)
        inventories.extend(inv)
        vectors.extend(dict(case=spec['case'], reaction=rid, flux=v) for rid, v in flux.items())
        if (i+1) % 18 == 0:
            print('isolated', i+1, '/', len(a), flush=True)
    pd.DataFrame(rows).to_csv(HERE/'isolated_results.csv', index=False)
    pd.DataFrame(vectors).to_csv(HERE/'isolated_fluxes.csv.gz', index=False, compression='gzip')
    pd.DataFrame(inventories).to_csv(HERE/'isolated_stocks.csv', index=False)
    rows, vectors, inventories = [], [], []
    for i, spec in enumerate(b):
        joint, initial, idx = prepare_community(spec, fields, community_template, native, curated, q)
        flux, _ = community.solve(joint, spec)
        if flux is None:
            assert spec['fixed_service'] is not None
            rows.append(spec | dict(status='infeasible_fixed_demand'))
        else:
            theta = float(joint.theta.primal)
            metrics, _, inv = community.validate(joint, spec, fields, initial, idx, flux, theta)
            refined, _, _ = community.validate(joint, spec, fields, initial, idx, flux, theta, dt=.05)
            assert abs(metrics['final_serine_pmol']-refined['final_serine_pmol']) < TOL
            t = flux['tumor']
            assert abs(-t['ASSAY_RIBOSE_c']+t['r0199']-t['DM_IMP_MATERIAL']) < TOL
            rows.append(spec | dict(status='optimal') | metrics)
            inventories.extend(dict(case=spec['case']) | r for r in inv)
            vectors.extend(dict(case=spec['case'], group=g, reaction=rid, flux=v) for g in flux for rid, v in flux[g].items())
        if (i+1) % 12 == 0 or i == len(b)-1:
            pd.DataFrame(rows).to_csv(HERE/'community_results.csv', index=False)
            print('community', i+1, '/', len(b), rows[-1]['status'], flush=True)
    pd.DataFrame(vectors).to_csv(HERE/'community_fluxes.csv.gz', index=False, compression='gzip')
    pd.DataFrame(inventories).to_csv(HERE/'community_stocks.csv', index=False)
    assert all(ssp.prior.sha(ROOT/p) == h for p, h in frozen.items())
    result = dict(isolated_cases=len(a), community_cases=len(b),
        isolated_all_optimal=True, community_status_counts=pd.DataFrame(rows).status.value_counts().to_dict(),
        finite_stock_energy_and_CNP_checks_passed=True, ATP_resource_gates=9,
        protected_files_unchanged=len(frozen), active_PhysiCell_changed=False, no_fit=True,
        no_growth_inference=True)
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
