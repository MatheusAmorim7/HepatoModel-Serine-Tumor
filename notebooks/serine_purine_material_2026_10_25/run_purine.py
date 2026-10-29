"""Finite-resource purine-material diagnostic, not a PATU reconstruction.

Native equations are copied unchanged. Directions, capacities, ribose access,
effective NADPH disposal and demand magnitudes are declared hypotheses.
"""
import hashlib
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
sys.path.insert(0, str(ROOT / 'notebooks/serine_paid_redox_2026_10_22'))
import redox_model as redox

HOURS = 12.0
PURINE_RATE = 0.014
WORK_ATP = 2.3912
TOL = 1e-7
# Sign is the allowed direction relative to the unmodified native equation.
NATIVE_DIRECTIONS = dict(r0248=1, r0247=1, r0030=1, r0007=1,
    r0257=-1, r0659=1, r0677=1, r0687=1, r0666=1, r0667=-1,
    r0705=1, r0699=1, r0700=1, r0273=-1, r0275=-1, r0816=-1,
    r0250=-1)
DE_NOVO = ['r0257', 'r0659', 'r0677', 'r0687', 'r0666', 'r0667',
           'r0705', 'r0699', 'r0700', 'r0273']
EXTRA_CNP = {
    'HC00119_c': (5, 0, 0), 'HC00115_c': (5, 0, 1),
    'HC00030_c': (10, 5, 1), 'HC00023_c': (0, 0, 2),
    'HC00117_c': (5, 0, 3), 'HC01053_c': (5, 1, 1),
    'HC01190_c': (7, 2, 1), 'HC01281_c': (8, 2, 1),
    'HC01329_c': (8, 3, 1), 'HC01108_c': (8, 3, 1),
    'HC01347_c': (9, 3, 1), 'HC01356_c': (13, 4, 1),
    'HC01334_c': (9, 4, 1), 'HC01344_c': (10, 4, 1),
    'HC00128_c': (10, 4, 1), 'HC00238_c': (5, 4, 0),
    'HC00238_s': (5, 4, 0), 'HC00067_c': (5, 2, 0),
    'HC00055_c': (4, 1, 0), 'HC00120_c': (4, 0, 0),
}
PHOSPHORUS = {'ATP': 3, 'ADP': 2, 'Pi': 1, 'NADP+': 3, 'NADPH': 3,
    'NAD+': 2, 'NADH': 2, '3PG': 1, '3-Phosphonooxypyruvate': 1,
    '3-Phosphoserine': 1, 'Glucose-6P': 1, 'Glucono-1,5-lactone-6P': 1,
    '6-Phospho-D-gluconate': 1, 'Ribulose-5P': 1}
PORTS = redox.ROLES | dict(ribose='ASSAY_RIBOSE_c',
    hypoxanthine='ASSAY_HX_s', glutamine='ASSAY_GLN_c',
    aspartate='ASSAY_ASP_c', glutamate='EX_GLU_c', fumarate='ASSAY_FUM_c')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atoms(met):
    if met.id in EXTRA_CNP:
        return np.array(EXTRA_CNP[met.id], dtype=float)
    cn = redox.atoms(met)
    return np.array([*cn, PHOSPHORUS.get(met.name.rsplit('(', 1)[0], 0)])


def add(model, rid, stoich, bounds=(0, 140)):
    r = cobra.Reaction(rid, lower_bound=bounds[0], upper_bound=bounds[1])
    r.add_metabolites({model.metabolites.get_by_id(k): v for k, v in stoich.items()})
    model.add_reactions([r])
    return r


def build(native, template, spec, q):
    selector = redox.RedoxSelector(template, native, dict(
        variant='mitochondrial_GCS', ATP_per_glucose_oxidative=24,
        policy=spec['policy']), q)
    m = selector.model
    for rid, sign in NATIVE_DIRECTIONS.items():
        r = native.reactions.get_by_id(rid).copy()
        r.bounds = (0, .14) if sign == 1 else (-.14, 0)
        r.notes['provenance'] = 'Unmodified HepatoNet1 chemical equation; diagnostic direction/capacity, not measured PATU activity'
        m.add_reactions([r])
    for rid in ['r0295', 'r0231', 'CAP_GLY_c_m', 'CAP_SER_m_c',
                'PENTOSE_ACCOUNTING', 'BOOKKEEP_PENTOSE_OUT']:
        m.reactions.get_by_id(rid).bounds = (0, 0)
    # Ser remains usable for folate C1 production, but assembly is closed here.
    m.reactions.r0230.bounds = (-.14, 0)
    m.reactions.r0227.bounds = (0, .14 if spec['formate_oxidation'] else 0)
    m.reactions.EX_NH3_m.bounds = (0, 0)
    for rid in redox.PPP:
        m.reactions.get_by_id(rid).bounds = (0, .14 if spec['ribose_source'] == 'glucose_PPP' else 0)
    m.reactions.r0250.bounds = (-.14, 0) if spec['ribose_source'] == 'glucose_PPP' else (0, 0)
    m.reactions.r0248.upper_bound = .14 if spec['ribose_source'] == 'finite_ribose' else 0
    for rid, mid in dict(ASSAY_RIBOSE_c='HC00119_c', ASSAY_HX_s='HC00238_s',
            ASSAY_GLN_c='HC00067_c', ASSAY_ASP_c='HC00055_c', ASSAY_FUM_c='HC00120_c').items():
        add(m, rid, {mid: -1})
    add(m, 'DM_IMP_MATERIAL', {'HC00128_c': -1})
    # Conservative effective electron disposal; not a specified human enzyme.
    r = add(m, 'ASSAY_NADPH_OX', {'HC00015_c': -1, 'E_O2_c': -.5,
        'HC00016_c': 1, 'HC00011_c': 1}, (0, .14 if spec['redox_disposal'] else 0))
    r.notes['scope'] = 'Hypothetical effective NADPH oxidation coupled to oxygen; no ATP credit, no enzyme/kinetics claim'
    p = m.problem
    m.remove_cons_vars(m.solver.constraints['minimum_DM_BOOKKEEP_C1'])
    m.add_cons_vars(p.Constraint(m.reactions.DM_BOOKKEEP_C1.flux_expression -
        spec['residual_C1'] * selector.theta, lb=0, ub=0, name='residual_C1_not_purine'))
    m.add_cons_vars(p.Constraint(m.reactions.DM_IMP_MATERIAL.flux_expression -
        PURINE_RATE * selector.theta, lb=0, ub=0, name='same_purine_material_task'))
    m.remove_cons_vars(m.solver.constraints['paid_energy_requirement'])
    m.add_cons_vars(p.Constraint(m.reactions.TASK_ATP.flux_expression -
        redox.MAINTENANCE * selector.alpha - WORK_ATP * selector.theta,
        lb=0, ub=0, name='separate_energy_turnover'))
    for rid in spec['blocks']:
        m.reactions.get_by_id(rid).bounds = (0, 0)
    if spec['fixed_service'] is not None:
        selector.theta.lb = spec['fixed_service']
        selector.theta.ub = spec['fixed_service']
    for r in m.reactions:
        if not r.boundary:
            residual = sum((v * atoms(met) for met, v in r.metabolites.items()), start=np.zeros(3))
            assert abs(residual).max() < 1e-10, (r.id, residual)
    m.notes['scope'] = 'Isolated finite-resource purine task. No hepatocytes, growth, isotope model or calibrated PATU capacities.'
    return selector


def stocks(spec):
    s = dict.fromkeys(PORTS, 0.0)
    s.update(serine=.5, glycine=2., formate=spec['formate_pmol'],
        hypoxanthine=spec['HX_pmol'], ribose=.3 if spec['ribose_source'] == 'finite_ribose' else 0,
        glutamine=.6, aspartate=.3, glucose=100., oxygen=36., phosphate=20.)
    s.update(spec.get('stock_override', {}))
    return s


def plan():
    base = dict(panel='primary', residual_C1=0., HX_pmol=0., formate_pmol=0.,
        ribose_source='finite_ribose', redox_disposal=True, formate_oxidation=False,
        fixed_service=None, selection='resource_sparing', blocks=[], policy='glucose_sparing')
    cases = []
    for residual in [0., .014]:
        for hx in [0., .05, .1, .2]:
            for formate in [0., .144]:
                for hprt in ['available', 'closed']:
                    cases.append(base | dict(residual_C1=residual, HX_pmol=hx,
                        formate_pmol=formate, HPRT=hprt, blocks=[] if hprt == 'available' else ['r0275']))
    for source in ['finite_ribose', 'glucose_PPP']:
        for redox_open in [False, True]:
            for hx in [0., .2]:
                cases.append(base | dict(panel='precursor_redox_sensitivity', ribose_source=source,
                    redox_disposal=redox_open, HX_pmol=hx, HPRT='available'))
    for label, blocks in [('PRPS_closed', ['r0247']), ('AK_closed', ['r0030']),
            ('PPase_closed', ['r0007']), ('ribose_kinase_closed', ['r0248']),
            ('HX_access_closed', ['r0816']), ('de_novo_closed', ['r0257']),
            ('both_routes_closed', ['r0257', 'r0275'])]:
        for hx in [0., .2]:
            cases.append(base | dict(panel=label, blocks=blocks, HX_pmol=hx, HPRT='available'))
    for label, override in [('no_serine', {'serine': 0.}), ('no_glutamine', {'glutamine': 0.}),
            ('no_energy_glucose', {'glucose': 0.}), ('no_oxygen', {'oxygen': 0.}),
            ('no_inputs', dict.fromkeys(PORTS, 0.))]:
        for hx in [0., .2]:
            cases.append(base | dict(panel=label, stock_override=override, HX_pmol=hx, HPRT='available'))
    for hx in [0., .2]:
        cases.append(base | dict(panel='allocation_policy', HX_pmol=hx,
            policy='oxygen_sparing', HPRT='available'))
    for hx in [0., .1, .2]:
        for oxidative_C1 in [False, True]:
            cases.append(base | dict(panel='formate_redox_source', HX_pmol=hx,
                residual_C1=.014, formate_pmol=.144, formate_oxidation=oxidative_C1,
                HPRT='available'))
    # Common feasible service below the lowest primary capacity. This compares
    # equal absolute IMP/Ser/C1/energy tasks, rather than different maxima.
    for residual in [0., .014]:
        for hx in [0., .2]:
            for hprt in ['available', 'closed']:
                cases.append(base | dict(panel='fixed_service_serine_budget',
                    residual_C1=residual, HX_pmol=hx, HPRT=hprt,
                    blocks=[] if hprt == 'available' else ['r0275'],
                    fixed_service=.4, selection='serine_sparing'))
    return [s | dict(case=f'purine_{i:03d}') for i, s in enumerate(cases)]


def configure_ports(selector, spec):
    m = selector.model
    initial = stocks(spec)
    for role, rid in PORTS.items():
        # Uniform budget cap: quantity / horizon, not a fitted transport rate.
        r = m.reactions.get_by_id(rid)
        upper = 0. if role in ['glucose', 'oxygen', 'ribose', 'hypoxanthine', 'glutamine', 'aspartate'] else 140.
        r.bounds = (-initial[role] / HOURS, upper)
    # All other entry ports are closed except solvent water, inherited as free solvent.
    known = set(PORTS.values()) | {'EX_WATER_c', 'EX_WATER_m',
        'DM_SER_c', 'DM_BOOKKEEP_C1', 'DM_IMP_MATERIAL'}
    assert all(r.lower_bound >= 0 or r.id in known for r in m.boundary)
    return initial


def solve(selector, spec):
    m, p = selector.model, selector.model.problem
    def opt(expr, direction):
        m.objective = p.Objective(expr, direction=direction)
        return float(m.slim_optimize(error_value=None))
    alpha = opt(selector.alpha, 'max')
    m.add_cons_vars(p.Constraint(selector.alpha, lb=alpha, ub=alpha, name='fixed_maintenance'))
    theta = opt(selector.theta, 'max')
    m.add_cons_vars(p.Constraint(selector.theta, lb=theta, ub=theta, name='fixed_tasks'))
    if spec['selection'] == 'serine_sparing':
        expr = -m.reactions.EX_SER_c.flux_expression
        value = opt(expr, 'min')
        m.add_cons_vars(p.Constraint(expr, lb=max(0., value-1e-10),
            ub=value+1e-10, name='fixed_serine_sparing'))
    order = ['glucose', 'oxygen'] if spec['policy'] == 'glucose_sparing' else ['oxygen', 'glucose']
    for role in order:
        expr = -m.reactions.get_by_id(PORTS[role]).flux_expression
        val = opt(expr, 'min')
        m.add_cons_vars(p.Constraint(expr, lb=max(0., val-1e-10), ub=val+1e-10, name='fixed_resource_'+role))
    opt(sum(r.forward_variable+r.reverse_variable for r in m.reactions), 'min')
    return alpha, theta, {r.id: float(r.flux) for r in m.reactions}


def check(selector, spec, initial, flux):
    m = selector.model
    v = np.array([flux[r.id] for r in m.reactions])
    sv = float(abs(create_stoichiometric_matrix(m) @ v).max())
    bounds = max(max(r.lower_bound-flux[r.id], flux[r.id]-r.upper_bound, 0.) for r in m.reactions)
    cnp = sum((sum((atoms(met)*coef for met, coef in r.metabolites.items()), start=np.zeros(3))*flux[r.id]
               for r in m.boundary), start=np.zeros(3))
    inventories = []
    for dt in [.1, .05]:
        stock = initial.copy()
        for step in range(round(HOURS/dt)):
            for role, rid in PORTS.items():
                stock[role] += dt*flux[rid]
        for role, amount in stock.items():
            expected = initial[role]+HOURS*flux[PORTS[role]]
            assert amount >= -TOL and abs(amount-expected) < TOL, (spec['case'], role, amount)
            if dt == .1:
                inventories.append(dict(case=spec['case'], substrate=role,
                    initial_pmol=initial[role], exchange_pmol=HOURS*flux[PORTS[role]], final_pmol=amount))
    # The energy identity counts AMP regeneration as well as ATP->ADP reactions.
    produced = 2*flux['PAID_GLYCOLYSIS']+24*flux['PAID_OXIDATION']
    consumed = sum(-r.metabolites.get(m.metabolites.HC00012_c, 0)*flux[r.id]
        for r in m.reactions if r.id not in ['PAID_GLYCOLYSIS', 'PAID_OXIDATION'])
    denovo, salvage = -flux['r0273'], -flux['r0275']
    assert abs(denovo+salvage-flux['DM_IMP_MATERIAL']) < TOL
    assert max(sv, bounds, abs(cnp).max(), abs(produced-consumed)) < TOL
    assert max(abs(flux[rid]) for rid in redox.BACKGROUND) < TOL
    return dict(max_Sv=sv, bound_error=bounds, max_CNP_error=float(abs(cnp).max()),
        ATP_energy_produced_pmol=produced*HOURS, ATP_energy_consumed_pmol=consumed*HOURS,
        IMP_de_novo_pmol=denovo*HOURS, IMP_salvage_pmol=salvage*HOURS,
        IMP_material_pmol=flux['DM_IMP_MATERIAL']*HOURS,
        serine_to_C1_pmol=-flux['r0230']*HOURS,
        NADPH_disposal_O2_pmol=.5*flux['ASSAY_NADPH_OX']*HOURS), inventories


def energy_gate(native, template, base, q):
    rows = []
    for label in ['all_nutrients_closed', 'glucose_closed_HX_ribose_available']:
        spec = base | dict(HX_pmol=.2)
        selector = build(native, template, spec, q)
        m = selector.model
        configure_ports(selector, spec)
        selector.theta.ub = 0
        selector.alpha.ub = 0
        m.remove_cons_vars(m.solver.constraints['separate_energy_turnover'])
        for role, rid in PORTS.items():
            if label == 'all_nutrients_closed' or role == 'glucose':
                m.reactions.get_by_id(rid).lower_bound = 0
        m.objective = 'TASK_ATP'
        m.objective_direction = 'max'
        value = float(m.slim_optimize(error_value=None))
        assert value < TOL
        rows.append(dict(control=label, max_ATP_hydrolysis_pmol_h=value))
    return rows


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    frozen_path = ROOT / 'notebooks/serine_public_discriminator_audit_2026_10_24/frozen_model_contract.json'
    protected = json.loads(frozen_path.read_text())['protected']
    assert all(sha(ROOT / p.replace('\\', '/')) == h for p, h in protected.items())
    cases = plan()
    protocol = dict(cases=cases, hours=HOURS, population_equivalent=7, volume_nL=20,
        purine_demand_pmol_h_at_full_service=PURINE_RATE, serine_material_demand_pmol_h=.056,
        residual_C1_demand='0 or .014 pmol/h * service; explicit non-purine sensitivity',
        energy_work_ATP_pmol_h=WORK_ATP, maintenance_ATP_pmol_h=redox.MAINTENANCE,
        C1_replacement='Generic .028 C1 demand replaced by .014 IMP using exactly two formyl-THF; separate residual C1 sensitivity retained',
        allocation='constant feasible 12h allocation; maintenance first, then joint task service, resource priorities and L1',
        common_service_control='theta .4, identical absolute tasks, minimize net Ser uptake before resources; not a measured allocation policy',
        stocks='finite external stocks; no replenishment; ribose access is a diagnostic precursor assumption, not a measured culture condition',
        redox='effective NADPH + 0.5 O2 disposal with zero ATP; separate closure and precursor-source sensitivity',
        units='pmol and hours; magnitudes and capacities hypothetical, not PATU rates',
        no_fit=True, growth_model=False, hepatocytes_present=False,
        CNP_scope='curated C/N/P moiety counts only; H/O/charge and thermodynamic validation not claimed',
        source_urls=['https://reactome.org/content/detail/R-HSA-74215',
            'https://reactome.org/content/detail/R-HSA-73817', 'https://reactome.org/content/detail/R-HSA-71732',
            'https://doi.org/10.1038/s41467-021-26395-5'],
        protected=protected)
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    native = cobra.io.read_sbml_model(str(ROOT / 'config/hepatonet_original_13_08.xml'))
    template = redox.build(native)
    q = json.loads((ROOT / 'notebooks/serine_tumor_response_audit/uptake_evidence/patu8902_serine_net_flux.json').read_text())['positive_net_consumption_nmol_per_million_cells_h']
    rows, vectors, balances = [], [], []
    for spec in cases:
        selector = build(native, template, spec, q)
        initial = configure_ports(selector, spec)
        alpha, theta, flux = solve(selector, spec)
        metrics, inventories = check(selector, spec, initial, flux)
        row = {k: v for k, v in spec.items() if k not in ['blocks', 'stock_override']}
        rows.append(row | dict(status='optimal', maintenance_fraction=alpha,
            task_service=theta, task_h=HOURS*theta) | metrics)
        balances.extend(inventories)
        vectors.extend(dict(case=spec['case'], reaction=rid, flux_pmol_h=value) for rid, value in flux.items())
        print(spec['case'], spec['panel'], round(HOURS*theta, 6), flush=True)
        if spec['case'] == cases[0]['case']:
            pd.DataFrame([dict(reaction=r.id, name=r.name, equation=r.reaction,
                lower_bound=r.lower_bound, upper_bound=r.upper_bound,
                native_equation_unchanged=r.id in NATIVE_DIRECTIONS)
                for r in selector.model.reactions]).to_csv(HERE / 'reaction_catalog.csv', index=False)
            pd.DataFrame([dict(metabolite=met.id, name=met.name,
                C=atoms(met)[0], N=atoms(met)[1], P=atoms(met)[2])
                for met in selector.model.metabolites]).to_csv(HERE / 'CNP_moiety_registry.csv', index=False)
    pd.DataFrame(rows).to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(vectors).to_csv(HERE / 'fluxes.csv', index=False)
    pd.DataFrame(balances).to_csv(HERE / 'stock_balances.csv', index=False)
    pd.DataFrame(energy_gate(native, template, cases[0], q)).to_csv(HERE / 'energy_gates.csv', index=False)
    assert all(sha(ROOT / p.replace('\\', '/')) == h for p, h in protected.items())
    result = dict(cases=len(rows), all_optimal=True, resource_and_CNP_checks_passed=True,
        closed_energy_gates_passed=True, model_and_prior_files_unchanged=True,
        max_CNP_error=max(r['max_CNP_error'] for r in rows),
        max_Sv=max(r['max_Sv'] for r in rows))
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
