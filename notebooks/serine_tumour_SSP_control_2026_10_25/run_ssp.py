"""Paid glucose-to-serine positive control; no PATU fit or active model edit.

Native C/N/P equations are unchanged. Capacities, access, electron disposal,
fixed tasks and ATP yield remain diagnostic assumptions, not phenotype data.
"""
import argparse
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
sys.path.insert(0, str(ROOT/'notebooks/serine_purine_material_2026_10_25'))
import run_purine as prior

HOURS, TOL = prior.HOURS, prior.TOL
UPPER = dict(r0192=1, r0736=1, r0256=1, r0243=-1, r0253=1, r0337=-1)
ADDED_NATIVE = UPPER | dict(r2522=1, r0078=1, r0171=-1)
EXTRA_CNP = {
    'HC00088_c': (6, 0, 1), 'HC00300_c': (6, 0, 2),
    'HC00109_c': (3, 0, 1), 'HC00116_c': (3, 0, 1), 'HC00214_c': (3, 0, 2),
    'HC00067_m': (5, 2, 0), 'HC00034_m': (5, 1, 0),
    'HC00032_c': (3, 0, 0), 'HC00177_c': (3, 0, 0)}
PORTS = prior.PORTS | dict(AKG='EX_AKG_c', pyruvate='ASSAY_PYR_c')


def atoms(met):
    return np.array(EXTRA_CNP[met.id], dtype=float) if met.id in EXTRA_CNP else prior.atoms(met)


def plan():
    base = prior.plan()[0] | dict(panel='factorial', residual_C1=.014,
        external_Ser_pmol=0., SSP='available', HPRT='available', nitrogen='glutamate',
        HX_pmol=0., glutamine_pmol=.6, glutamate_pmol=1., glycine_pmol=0.,
        cyto_redox='oxygen_linked', pyruvate_pmol=0., extra_blocks=[], stock_override={})
    rows = []
    for ser in [0., .5]:
        for ssp in ['available', 'blocked']:
            for hprt in ['available', 'blocked']:
                for hx in [0., .004, .1]:
                    for nitrogen in ['glutamate', 'glutamine']:
                        rows.append(base | dict(external_Ser_pmol=ser, SSP=ssp,
                            HPRT=hprt, HX_pmol=hx, nitrogen=nitrogen))
    for ssp in ['available', 'blocked']:
        for hprt in ['available', 'blocked']:
            for hx in [0., .1]:
                rows.append(base | dict(panel='high_external_Ser', external_Ser_pmol=2.8,
                    SSP=ssp, HPRT=hprt, HX_pmol=hx))
    for rid in ['r0353', 'r0192', 'r0736', 'r0256', 'r0243', 'r0253', 'r0337',
            'r0338', 'r0663', 'r0159', 'ASSAY_NADH_OX_c']:
        rows.append(base | dict(panel='route_closure', extra_blocks=[rid], HX_pmol=.1))
    for panel, override, blocks in [
            ('no_glucose', {'glucose': 0.}, []),
            ('no_inputs', dict.fromkeys(PORTS, 0.), []),
            ('energy_routes_closed', {}, ['PAID_GLYCOLYSIS', 'PAID_OXIDATION']),
            ('no_external_nitrogen', {'serine': 0., 'glycine': 0., 'glutamate': 0.,
                'glutamine': 0., 'aspartate': 0., 'hypoxanthine': 0.}, []),
            ('no_oxygen', {'oxygen': 0.}, [])]:
        rows.append(base | dict(panel=panel, stock_override=override, extra_blocks=blocks))
    for rid in ['r2522', 'r0078', 'ASSAY_GLU_m_c']:
        rows.append(base | dict(panel='GLS_access_closure', nitrogen='glutamine', extra_blocks=[rid]))
    # Isolate NADH reoxidation from non-purine C1/redox demand. Finite pyruvate
    # pays the electron-acceptor alternative; no O2 or free cofactor is added.
    for redox in ['oxygen_linked', 'LDH', 'closed']:
        for pyr in [0., 1.5]:
            rows.append(base | dict(panel='nonrespiratory_NADH_control', residual_C1=0.,
                HX_pmol=.2, cyto_redox=redox, pyruvate_pmol=pyr, stock_override={'oxygen': 0.}))
    for nitrogen in ['glutamate', 'glutamine']:
        for hx in [0., .1]:
            rows.append(base | dict(panel='oxygen_sparing', nitrogen=nitrogen,
                HX_pmol=hx, policy='oxygen_sparing'))
    assert len(rows) == 85
    return [s | dict(case=f'ssp_{i:03d}') for i, s in enumerate(rows)]


def build(spec, native, template, q):
    base = prior.plan()[0] | dict(residual_C1=spec['residual_C1'], policy=spec['policy'])
    selector = prior.build(native, template, base, q)
    m = selector.model
    for rid, direction in ADDED_NATIVE.items():
        r = native.reactions.get_by_id(rid).copy()
        cap = .28 if rid in ['r0253', 'r0337', 'r0171'] else .14
        r.bounds = (0., cap) if direction > 0 else (-cap, 0.)
        r.notes['provenance'] = 'HepatoNet1 chemical equation unchanged; direction/capacity diagnostic, not measured tumour activity'
        m.add_reactions([r])
    m.reactions.r0353.bounds = (0., .14)
    m.reactions.r0338.bounds = (0., .14 if spec['SSP'] == 'available' else 0.)
    m.reactions.r0663.bounds = (-.14, 0.)
    m.reactions.r0159.bounds = (0., .14)
    # Existing E_GLC -> native glucose bridge now feeds both HK/glycolysis
    # and the (closed here) PPP branch. No carbon is credited twice.
    prior.add(m, 'ASSAY_GLU_m_c', {'HC00034_m': -1, 'HC00034_c': 1}, (0., .14))
    m.reactions.ASSAY_GLU_m_c.notes['scope'] = 'Effective finite matrix glutamate delivery; not a calibrated carrier or proton/energy model'
    prior.add(m, 'ASSAY_NADH_OX_c', {'HC00014_c': -1, 'E_O2_c': -.5,
        'HC00013_c': 1, 'HC00011_c': 1}, (0., .28))
    m.reactions.ASSAY_NADH_OX_c.notes['scope'] = 'Effective oxygen-linked electron disposal, zero ATP credit; not a specific cytosolic oxidase'
    prior.add(m, 'ASSAY_PYR_c', {'HC00032_c': -1}, (0., 0.))
    prior.add(m, 'ASSAY_LACTATE_TO_POOL', {'HC00177_c': -1, 'E_LAC_c': 1})
    if spec['nitrogen'] == 'glutamate':
        for rid in ['r2522', 'r0078', 'ASSAY_GLU_m_c']:
            m.reactions.get_by_id(rid).bounds = (0., 0.)
    if spec['cyto_redox'] != 'oxygen_linked':
        m.reactions.ASSAY_NADH_OX_c.bounds = (0., 0.)
    if spec['cyto_redox'] != 'LDH':
        m.reactions.r0171.bounds = (0., 0.)
    if spec['HPRT'] == 'blocked':
        m.reactions.r0275.bounds = (0., 0.)
    for rid in spec['extra_blocks']:
        m.reactions.get_by_id(rid).bounds = (0., 0.)
    for r in m.reactions:
        if not r.boundary:
            balance = sum((atoms(met)*value for met, value in r.metabolites.items()), start=np.zeros(3))
            assert abs(balance).max() < 1e-10, (r.id, balance)
    return selector


def initial_stock(spec):
    stock = dict.fromkeys(PORTS, 0.)
    stock.update(glucose=100., oxygen=36., phosphate=20., ribose=.3,
        serine=spec['external_Ser_pmol'], glycine=spec['glycine_pmol'],
        hypoxanthine=spec['HX_pmol'], glutamine=spec['glutamine_pmol'], aspartate=.3,
        glutamate=spec['glutamate_pmol'] if spec['nitrogen'] == 'glutamate' else 0.,
        pyruvate=spec['pyruvate_pmol'])
    stock.update(spec['stock_override'])
    return stock


def configure_ports(selector, spec):
    m = selector.model
    stock = initial_stock(spec)
    for role, rid in PORTS.items():
        upper = 0. if role in ['glucose', 'oxygen', 'ribose', 'hypoxanthine',
            'glutamine', 'aspartate', 'pyruvate'] else 140.
        m.reactions.get_by_id(rid).bounds = (-stock[role]/HOURS, upper)
    assert m.reactions.EX_3PG_c.bounds == (0., 0.)
    assert all(m.reactions.get_by_id(rid).bounds == (0., 0.) for rid in prior.redox.BACKGROUND)
    return stock


def validate(selector, spec, stock, flux, alpha, theta):
    m = selector.model
    sv = float(abs(create_stoichiometric_matrix(m)@np.array([flux[r.id] for r in m.reactions])).max())
    bound = max(max(r.lower_bound-flux[r.id], flux[r.id]-r.upper_bound, 0.) for r in m.reactions)
    cnp = sum((sum((atoms(met)*coef for met, coef in r.metabolites.items()), start=np.zeros(3))*flux[r.id]
        for r in m.boundary), start=np.zeros(3))
    source = 2*flux['PAID_GLYCOLYSIS']+24*flux['PAID_OXIDATION']
    cost = sum(-r.metabolites.get(m.metabolites.HC00012_c, 0.)*flux[r.id] for r in m.reactions
        if r.id not in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION'])
    ssp = flux['r0159']
    equations = [flux['r0338']-ssp, -flux['r0663']-ssp,
        flux['r0253']-ssp, -flux['r0337']-ssp,
        flux['r0353']-.5*ssp, flux['r0736']-.5*ssp,
        flux['r0256']-.5*ssp, -flux['r0243']-.5*ssp,
        flux['r0253']+flux['r0338']-flux['ASSAY_NADH_OX_c']+flux['r0171'],
        flux['EX_AKG_c']-ssp, -flux['r0273']-flux['r0275']-flux['DM_IMP_MATERIAL'],
        flux['DM_SER_c']-.056*theta, flux['DM_BOOKKEEP_C1']-spec['residual_C1']*theta,
        flux['DM_IMP_MATERIAL']-prior.PURINE_RATE*theta,
        flux['TASK_ATP']-.0035*alpha-prior.WORK_ATP*theta,
        -flux['EX_E_GLC']-flux['PAID_GLYCOLYSIS']-flux['PAID_OXIDATION']-flux['GLUCOSE_TO_PPP'],
        -flux['ASSAY_GLN_c']-flux['r0078']+2*flux['r0273'],
        flux['EX_NH3_m']-flux['r0078']]
    eq = max(abs(value) for value in equations)
    assert max(sv, bound, abs(cnp).max(), abs(source-cost), eq) < TOL
    assert max(abs(flux[rid]) for rid in prior.redox.BACKGROUND) < TOL
    inventory = []
    for dt in [.1, .05]:
        remaining = stock.copy()
        for _ in range(round(HOURS/dt)):
            for role, rid in PORTS.items():
                remaining[role] += dt*flux[rid]
        for role, amount in remaining.items():
            assert amount >= -TOL and abs(amount-stock[role]-HOURS*flux[PORTS[role]]) < TOL
            if dt == .1:
                inventory.append(dict(case=spec['case'], substrate=role, initial_pmol=stock[role],
                    exchange_pmol=HOURS*flux[PORTS[role]], final_pmol=amount))
    known = set(PORTS.values()) | {'EX_WATER_c', 'EX_WATER_m', 'DM_SER_c', 'DM_BOOKKEEP_C1', 'DM_IMP_MATERIAL'}
    assert all(abs(flux[r.id]) < TOL or r.id in known for r in m.boundary)
    return dict(task_h=HOURS*theta, maintenance_fraction=alpha,
        SSP_serine_pmol=HOURS*ssp, SSP_glucose_carbon_pmol=.5*HOURS*ssp,
        SSP_NADH_generated_pmol=2*HOURS*ssp, AKG_released_pmol=HOURS*flux['EX_AKG_c'],
        serine_uptake_pmol=-HOURS*flux['EX_SER_c'], glutamate_net_uptake_pmol=-HOURS*flux['EX_GLU_c'],
        glutamine_uptake_pmol=-HOURS*flux['ASSAY_GLN_c'], GLS_pmol=HOURS*flux['r0078'],
        IMP_de_novo_pmol=-HOURS*flux['r0273'], IMP_salvage_pmol=-HOURS*flux['r0275'],
        cyto_redox_O2_pmol=.5*HOURS*flux['ASSAY_NADH_OX_c'],
        LDH_pyruvate_pmol=-HOURS*flux['r0171'], ATP_produced_pmol=HOURS*source,
        max_Sv=sv, max_bound_error=bound, max_CNP_error=float(abs(cnp).max()), max_equation_error=eq), inventory


def gates(native, template, q):
    rows = []
    for mode in ['all_inputs_closed', 'energy_routes_closed_with_SSP_substrates']:
        spec = plan()[0]
        selector = build(spec, native, template, q)
        configure_ports(selector, spec)
        m = selector.model
        selector.theta.lb = selector.theta.ub = 0.
        selector.alpha.lb = selector.alpha.ub = 0.
        m.remove_cons_vars(m.solver.constraints['separate_energy_turnover'])
        if mode == 'all_inputs_closed':
            for rid in PORTS.values():
                m.reactions.get_by_id(rid).lower_bound = 0.
        else:
            for rid in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION']:
                m.reactions.get_by_id(rid).bounds = (0., 0.)
        m.objective = 'TASK_ATP'
        m.objective_direction = 'max'
        value = float(m.slim_optimize(error_value=None))
        assert value < TOL
        rows.append(dict(control=mode, max_ATP_hydrolysis_pmol_h=value))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    specs = plan()
    if args.smoke:
        specs = [specs[i] for i in [0, 1, 6, 12, 24, 75, 78]]
    contract = json.loads((ROOT/'notebooks/serine_public_functional_challenge_2026_10_25/frozen_model_contract.json').read_text())['protected']
    paths = [Path(__file__), Path(prior.__file__), ROOT/'notebooks/serine_public_functional_challenge_2026_10_25/challenge_contract.json']
    paths += list((ROOT/'notebooks/serine_public_functional_challenge_2026_10_25').glob('*.csv'))
    protected = contract | {str(p.relative_to(ROOT)): prior.sha(p) for p in paths}
    assert all(prior.sha(ROOT/path.replace('\\', '/')) == h for path, h in protected.items())
    suffix = '_smoke' if args.smoke else ''
    protocol = dict(cases=specs, protected=protected, declared_before_execution=True,
        scope='Isolated SSP-positive chemistry control; no hepatocytes, growth or calibrated tumour line',
        native_equations=ADDED_NATIVE, SSP_native=['r0338', 'r0663', 'r0159'],
        carbon='Only native upper glycolysis supplies3PG; external3PG closed; existing glucose pool pays energy and material separately',
        net_upper_glycolysis='Per Ser:0.5Glc gives3PG and1NADH, zero net ATP; PHGDH adds1NADH; nitrogen from Glu releases1AKG',
        redox='O2-linked effective NADH disposal zero ATP credit; separate finite-pyruvate LDH alternative, not an enzyme or calibrated shuttle claim',
        nitrogen='Finite Glu or finite Gln through native matrix GLS plus effective glutamate delivery; purine-derived Glu may also contribute',
        maintenance='Prior .0035 ATP/h, work2.3912*service; hypothetical, not a measured translation/growth model',
        drug_scope='PHGDH reaction closure is a route test, not PH755 dose response; HPRT closure not quantitative siRNA or6MP',
        source_dose='0.004/0.1pmol correspond to0.2/5uM only in diagnostic20nL; not experimental culture volume',
        no_parameter_fit=True, preserve_SSP_low_base=True, active_PhysiCell_modified=False)
    (HERE/('protocol'+suffix+'.json')).write_text(json.dumps(protocol, indent=2)+'\n')
    native = cobra.io.read_sbml_model(str(ROOT/'config/hepatonet_original_13_08.xml'))
    template = prior.redox.build(native)
    q = json.loads((ROOT/'notebooks/serine_tumor_response_audit/uptake_evidence/patu8902_serine_net_flux.json').read_text())['positive_net_consumption_nmol_per_million_cells_h']
    rows, vectors, inventory = [], [], []
    for i, spec in enumerate(specs):
        selector = build(spec, native, template, q)
        stock = configure_ports(selector, spec)
        alpha, theta, flux = prior.solve(selector, spec)
        metrics, own_inventory = validate(selector, spec, stock, flux, alpha, theta)
        rows.append({key: value for key, value in spec.items() if key not in ['blocks', 'extra_blocks', 'stock_override']} |
            dict(status='optimal') | metrics)
        vectors.extend(dict(case=spec['case'], reaction=rid, flux=value) for rid, value in flux.items())
        inventory += own_inventory
        if i == 0:
            pd.DataFrame([dict(reaction=r.id, equation=r.reaction, lb=r.lower_bound, ub=r.upper_bound)
                for r in selector.model.reactions]).to_csv(HERE/('reaction_catalog'+suffix+'.csv'), index=False)
        if (i+1) % 10 == 0 or i == len(specs)-1:
            print(i+1, '/', len(specs), spec['case'], spec['panel'], round(metrics['task_h'], 6), flush=True)
    pd.DataFrame(rows).to_csv(HERE/('results'+suffix+'.csv'), index=False)
    pd.DataFrame(vectors).to_csv(HERE/('fluxes'+suffix+'.csv.gz'), index=False, compression='gzip')
    pd.DataFrame(inventory).to_csv(HERE/('stock_balances'+suffix+'.csv'), index=False)
    pd.DataFrame(gates(native, template, q)).to_csv(HERE/('energy_gates'+suffix+'.csv'), index=False)
    assert all(prior.sha(ROOT/path.replace('\\', '/')) == h for path, h in protected.items())
    result = dict(cases=len(rows), all_optimal=True, chemistry_CNP_energy_and_finite_stock_checks_passed=True,
        ATP_gates_passed=True, max_CNP_error=max(r['max_CNP_error'] for r in rows),
        max_equation_error=max(r['max_equation_error'] for r in rows),
        protected_hashes_unchanged=True, no_fit=True, growth_inferred=False)
    (HERE/('verification'+suffix+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
