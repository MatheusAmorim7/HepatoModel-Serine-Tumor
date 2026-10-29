"""Isolated, normalized Ser/Gly/C1 chemistry test using COBRApy and GLPK.

This is a capability module, not a reconstructed PATU-8902 metabolic model.
Native HepatoNet equations are chemical templates only. Background energy and
redox budgets are explicit assumptions, not reactions inferred in tumor cells.
"""
import hashlib
import itertools
import json
import logging
from pathlib import Path
import warnings

import cobra
from cobra.flux_analysis import flux_variability_analysis, pfba
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / 'config/hepatonet_original_13_08.xml'
OUT = HERE / 'outputs'
BIG = 1000.
CYTO = ['r0230', 'r0293', 'r0371', 'r0228']
MITO = ['r0231', 'r0295']
SSP = ['r0338', 'r0663', 'r0159']
OXI = ['r0227']
VARIANTS = ['uptake_only', 'cytosolic_C1', 'plus_mito_GCS', 'plus_formate_oxidation', 'SSP_restored_control']
CN = {
    'H2O': (0, 0), 'ATP': (10, 5), 'ADP': (10, 5), 'Pi': (0, 0),
    'NADP+': (21, 7), 'NADPH': (21, 7), 'NAD+': (21, 7), 'NADH': (21, 7),
    'Serine': (3, 1), 'Glycine': (2, 1), 'Formate': (1, 0), 'THF': (19, 7),
    '5,10-Methylene-THF': (20, 7), '5,10-Methenyl-THF': (20, 7),
    '10-Formyl-THF': (20, 7), 'CO2': (1, 0), 'NH3': (0, 1),
    '3PG': (3, 0), '3-Phosphonooxypyruvate': (3, 0), '3-Phosphoserine': (3, 1),
    'Glutamate': (5, 1), 'AKG': (5, 0),
}
PORTS = {
    'serine': ('EX_SER_c', 'HC00068_c'),
    'glycine': ('EX_GLY_c', 'HC00045_c'),
    'formate': ('EX_FOR_c', 'HC00061_c'),
    '3PG': ('EX_3PG_c', 'HC00186_c'),
    'glutamate': ('EX_GLU_c', 'HC00034_c'),
    'AKG': ('EX_AKG_c', 'HC00035_c'),
    'CO2': ('EX_CO2_m', 'HC00021_m'),
    'CO2_c': ('EX_CO2_c', 'HC00021_c'),
    'ammonia': ('EX_NH3_m', 'HC00024_m'),
    'phosphate': ('EX_PI_c', 'HC00019_c'),
    'water_c': ('EX_WATER_c', 'HC00011_c'),
    'water_m': ('EX_WATER_m', 'HC00011_m'),
}
SOURCES = {
    'observed_response': 'https://doi.org/10.1002/advs.202515740',
    'MTHFD1_formate_ligation': 'https://reactome.org/content/detail/R-HSA-200711',
    'SHMT1_reversible_chemistry': 'https://reactome.org/content/detail/R-HSA-200735',
    'GCS_chemistry': 'https://reactome.org/content/detail/R-HSA-6783984',
    'cytosolic_C1_reversal_primary': 'https://pubmed.ncbi.nlm.nih.gov/27211901/',
}
GENES = dict(r0230='SHMT1', r0293='MTHFD1', r0371='MTHFD1', r0228='MTHFD1',
             r0231='SHMT2', r0295='GLDC and AMT and GCSH and DLD',
             r0338='PHGDH', r0663='PSAT1', r0159='PSPH', r0227='ALDH1L1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cn(met):
    return CN[met.name.rsplit('(', 1)[0]]


def add(model, rid, stoichiometry, bounds, name, role):
    r = cobra.Reaction(rid, name=name, lower_bound=bounds[0], upper_bound=bounds[1])
    r.add_metabolites({model.metabolites.get_by_id(mid): value for mid, value in stoichiometry.items()})
    r.notes['diagnostic_role'] = role
    model.add_reactions([r])
    return r


def build(native):
    model = cobra.Model('Ser_Gly_C1_capability_DIAGNOSTIC_ONLY')
    model.solver = 'glpk'
    model.notes['scope'] = 'Normalized chemistry capability; not a PATU-8902 reconstruction, growth law or PhysiCell deployment.'
    model.notes['elemental_scope'] = 'Source has no full formulas. Only carbon/nitrogen balances are independently audited. No fabricated formulas assigned.'
    model.notes['background'] = 'ATP regeneration, NADPH regeneration and NADH disposal have explicit finite unit budgets; not glucose/O2-calibrated.'
    model.notes['folates'] = 'Pre-existing functional folate pools and holoenzymes assumed. Internal recycling only; no folate exchange. No folate pool kinetics.'
    for rid in CYTO + MITO + SSP + OXI:
        reaction = native.reactions.get_by_id(rid).copy()
        reaction.notes['chemical_template'] = 'HepatoNet1 equation; copied without stoichiometric changes.'
        reaction.notes['tumor_expression'] = 'Generic biochemical gene mapping; activity in PATU-8902 not inferred from hepatic origin.'
        reaction.gene_reaction_rule = GENES[rid]
        model.add_reactions([reaction])
    # Native AKG id is verified by its name before building any boundary port.
    for role, (rid, mid) in PORTS.items():
        if mid not in model.metabolites:
            source_met = native.metabolites.get_by_id(mid)
            model.add_metabolites([source_met.copy()])
        expected_names = {'AKG': 'AKG', 'glutamate': 'Glutamate'}
        if role in expected_names:
            assert model.metabolites.get_by_id(mid).name.startswith(expected_names[role]+'('), (role, mid)
        add(model, rid, {mid: -1}, (0, 0), 'Diagnostic boundary '+role,
            'Open boundary with explicit supply/export limit, not a measured transport capacity.')
    add(model, 'DM_SER_c', {'HC00068_c': -1}, (0, BIG), 'Serine availability sink, NOT biomass',
        'Serine withdrawal proxy only; no nucleotide, protein, redox or growth-demand vector.')
    add(model, 'CAP_GLY_c_m', {'HC00045_c': -1, 'HC00045_m': 1}, (0, 0),
        'Effective glycine matrix delivery', 'Conditional effective transport, no measured tumor rate or unique carrier assignment.')
    add(model, 'CAP_SER_m_c', {'HC00068_m': -1, 'HC00068_c': 1}, (0, 0),
        'Effective mitochondrial serine exit', 'Conditional effective transport, not a validated PATU-8902 carrier.')
    add(model, 'BG_ATP', {'HC00018_c': -1, 'HC00019_c': -1, 'HC00012_c': 1}, (0, 1),
        'Background ATP regeneration budget', 'Bookkeeping boundary for background metabolism; not a biochemical ATP-generating pathway.')
    add(model, 'BG_NADPH', {'HC00016_c': -1, 'HC00015_c': 1}, (0, 1),
        'Background NADPH regeneration budget', 'Bookkeeping boundary for reducing equivalents; no measured NADPH supply.')
    for suffix in ['c', 'm']:
        add(model, 'BG_NADH_'+suffix, {'HC00014_'+suffix: -1, 'HC00013_'+suffix: 1}, (0, 1),
            'Background NADH oxidation budget '+suffix,
            'Finite redox disposal, not an oxidative-phosphorylation or oxygen-consumption model.')
    model.objective = 'DM_SER_c'
    model.objective_direction = 'max'
    return model


def configure(template, variant, condition, gcs_cap=1.):
    model = template.copy()
    assert variant in VARIANTS
    for rid in CYTO:
        model.reactions.get_by_id(rid).bounds = (0, 0) if variant == 'uptake_only' else (-BIG, BIG)
    # ATP-dependent formate activation; reverse formate production is not tested here.
    model.reactions.get_by_id('r0228').lower_bound = 0
    for rid in MITO:
        model.reactions.get_by_id(rid).bounds = (0, 0)
    for rid in ['CAP_GLY_c_m', 'CAP_SER_m_c']:
        model.reactions.get_by_id(rid).bounds = (0, 0)
    if variant == 'plus_mito_GCS':
        model.reactions.get_by_id('r0295').bounds = (0, gcs_cap)
        model.reactions.get_by_id('r0231').bounds = (0, BIG)
        for rid in ['CAP_GLY_c_m', 'CAP_SER_m_c']:
            model.reactions.get_by_id(rid).bounds = (0, BIG)
    model.reactions.get_by_id('r0338').bounds = (0, 1 if variant == 'SSP_restored_control' else 0)
    model.reactions.get_by_id('r0663').bounds = (-BIG, 0)
    model.reactions.get_by_id('r0159').bounds = (0, BIG)
    model.reactions.get_by_id('r0227').bounds = (0, 1 if variant == 'plus_formate_oxidation' else 0)
    for role, (rid, _) in PORTS.items():
        bounds = (0, BIG)
        if role.startswith('water'):
            bounds = (-BIG, BIG)
        elif role in ['serine', 'glycine', 'formate']:
            nominal = condition[role+'_uM']
            bounds = (-1. if nominal > 0 else 0., 0.)
        elif role in ['3PG', 'glutamate']:
            bounds = (-1. if variant == 'SSP_restored_control' else 0., 0.)
        model.reactions.get_by_id(rid).bounds = bounds
    return model


def diagnostics(model, solution):
    v = solution.fluxes
    matrix = create_stoichiometric_matrix(model)
    sv = float(np.abs(matrix @ v.to_numpy()).max())
    violation = max(max(r.lower_bound-v[r.id], v[r.id]-r.upper_bound, 0) for r in model.reactions)
    carbon, nitrogen = 0., 0.
    for reaction in model.boundary:
        c, n = np.sum([np.array(cn(m))*coeff for m, coeff in reaction.metabolites.items()], axis=0)
        carbon += c*v[reaction.id]
        nitrogen += n*v[reaction.id]
    assert sv < 1e-7 and violation < 1e-7
    assert max(abs(carbon), abs(nitrogen)) < 1e-7
    return dict(max_Sv=sv, bound_violation=float(violation), carbon_residual=float(carbon),
                nitrogen_residual=float(nitrogen))


def solve(model, tag, rows, vectors):
    model.objective = 'DM_SER_c'
    model.objective_direction = 'max'
    solution = model.optimize(raise_error=True)
    maximum = float(solution.fluxes['DM_SER_c'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        parsimonious = pfba(model)
    row = dict(case_id=len(rows), **tag, status=solution.status, max_serine_availability=maximum,
               flux_units='normalized_capacity_not_mmol_gDW_h', **diagnostics(model, parsimonious))
    for role, (rid, _) in PORTS.items():
        row['flux_'+role] = float(parsimonious.fluxes[rid])
    for rid in CYTO + MITO + SSP + OXI + ['CAP_GLY_c_m', 'CAP_SER_m_c', 'BG_ATP', 'BG_NADPH', 'BG_NADH_c', 'BG_NADH_m']:
        row['flux_'+rid] = float(parsimonious.fluxes[rid])
    row['serine_withdrawal_flux'] = float(parsimonious.fluxes['DM_SER_c'])
    assert abs(maximum - row['serine_withdrawal_flux']) < 1e-7
    rows.append(row)
    vectors.extend(dict(case_id=row['case_id'], reaction=rid, flux=float(value))
                   for rid, value in parsimonious.fluxes.items())
    return row


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    OUT.mkdir(exist_ok=True)
    (HERE / 'models').mkdir(exist_ok=True)
    native = cobra.io.read_sbml_model(str(NATIVE))
    template = build(native)
    protocol_source = ROOT / 'notebooks/serine_shin2026_response_audit/external_challenge_protocol.json'
    protocol = json.loads(protocol_source.read_text())
    conditions = {c['id']: dict(id=c['id'], serine_uM=c['serine_uM'], glycine_uM=c['glycine_uM'], formate_uM=c['formate_uM'])
                  for c in protocol['conditions']}
    rows, vectors, catalog, fva_rows = [], [], [], []
    for r in template.reactions:
        internal = not r.boundary
        residual = np.sum([np.array(cn(m))*coeff for m, coeff in r.metabolites.items()], axis=0)
        if internal:
            assert np.max(np.abs(residual)) < 1e-9, (r.id, residual)
        native_match = r.id in native.reactions and {
            m.id:v for m,v in r.metabolites.items()} == {
            m.id:v for m,v in native.reactions.get_by_id(r.id).metabolites.items()}
        catalog.append(dict(reaction=r.id, name=r.name, equation=r.build_reaction_string(use_metabolite_names=True),
                            role='native_chemical_template' if native_match else r.notes.get('diagnostic_role'),
                            native_stoichiometry_identical=native_match, generic_gene_mapping=r.gene_reaction_rule,
                            carbon_residual=float(residual[0]), nitrogen_residual=float(residual[1]),
                            boundary=not internal, full_mass_charge_audited=False))
    pd.DataFrame(catalog).to_csv(OUT / 'reaction_catalog.csv', index=False)

    for variant, condition in itertools.product(VARIANTS, conditions.values()):
        model = configure(template, variant, condition)
        row = solve(model, dict(group='main', variant=variant, condition=condition['id'], intervention='none'), rows, vectors)
        if condition['id'] == 'EAA_GLY_FOR':
            model.notes['conditions'] = 'Ser input closed, Gly and formate available at separate normalized unit uptake ceilings.'
            model.notes['nominal_doses'] = 'Gly 100 uM, formate 500 uM are source labels only. No concentration-to-flux mapping is claimed.'
            cobra.io.write_sbml_model(model, str(HERE / 'models' / (variant+'_capability.xml')))
        representative = (variant, condition['id']) in [
            ('cytosolic_C1', 'EAA_GLY_FOR'), ('cytosolic_C1', 'EAA_GLY'),
            ('plus_mito_GCS', 'EAA_GLY'), ('SSP_restored_control', 'EAA')]
        if representative:
            frame = flux_variability_analysis(model, CYTO+MITO+SSP, processes=1, fraction_of_optimum=1.)
            for rid, values in frame.iterrows():
                fva_rows.append(dict(case_id=row['case_id'], variant=variant, condition=condition['id'],
                                     reaction=rid, minimum=float(values.minimum), maximum=float(values.maximum),
                                     fixed_primary_serine_optimum=row['max_serine_availability'],
                                     face='primary_optimal_not_pFBA_face'))

    tests = []
    # Route ablations must remove the claimed capacity, not produce infeasibility.
    ablations = {
        'cytosolic_C1': ('EAA_GLY_FOR', ['r0228', 'r0371', 'r0293', 'r0230', 'BG_ATP', 'BG_NADPH', 'EX_GLY_c', 'EX_FOR_c']),
        'plus_mito_GCS': ('EAA_GLY', ['r0295', 'r0231', 'CAP_GLY_c_m', 'CAP_SER_m_c', 'BG_NADH_m']),
        'SSP_restored_control': ('EAA', ['r0338', 'EX_3PG_c', 'EX_GLU_c', 'BG_NADH_c']),
    }
    for variant, (condition_id, reactions) in ablations.items():
        for rid in reactions:
            model = configure(template, variant, conditions[condition_id])
            model.reactions.get_by_id(rid).bounds = (0, 0)
            row = solve(model, dict(group='ablation', variant=variant, condition=condition_id,
                                   intervention='close_'+rid), rows, vectors)
            tests.append(dict(check='route_ablation_'+variant+'_'+rid,
                              passed=abs(row['max_serine_availability']) < 1e-7,
                              value=row['max_serine_availability']))
    for cap in [0., .001, .01, .1, .5, 1.]:
        model = configure(template, 'plus_mito_GCS', conditions['EAA_GLY'], gcs_cap=cap)
        row = solve(model, dict(group='gcs_sensitivity', variant='plus_mito_GCS', condition='EAA_GLY',
                               intervention='GCS_cap_'+str(cap), normalized_capacity=cap), rows, vectors)
        expected = min(cap, .5)
        tests.append(dict(check='GCS_capacity_yield_'+str(cap), passed=abs(row['max_serine_availability']-expected) < 1e-7,
                          value=row['max_serine_availability']))
    for budget, cap in itertools.product(['BG_ATP', 'BG_NADPH'], [0., .01, .1, .5, 1.]):
        model = configure(template, 'cytosolic_C1', conditions['EAA_GLY_FOR'])
        model.reactions.get_by_id(budget).upper_bound = cap
        row = solve(model, dict(group='cofactor_sensitivity', variant='cytosolic_C1', condition='EAA_GLY_FOR',
                               intervention=budget+'_cap_'+str(cap), normalized_capacity=cap), rows, vectors)
        tests.append(dict(check=budget+'_yield_'+str(cap), passed=abs(row['max_serine_availability']-cap) < 1e-7,
                          value=row['max_serine_availability']))

    # The NADPH requirement can be met by another carbon-consuming route.
    # Probe it explicitly rather than asserting external NADPH support is unique.
    for cap in [0., .001, .01, .1, .5, 1.]:
        model = configure(template, 'plus_formate_oxidation', conditions['EAA_GLY_FOR'])
        model.reactions.BG_NADPH.bounds = (0, 0)
        model.reactions.get_by_id('r0227').upper_bound = cap
        row = solve(model, dict(group='formate_redox_alternative', variant='plus_formate_oxidation',
                               condition='EAA_GLY_FOR', intervention='ALDH1L1_cap_'+str(cap), normalized_capacity=cap),
                    rows, vectors)
        tests.append(dict(check='formate_redox_yield_'+str(cap), passed=abs(row['max_serine_availability']-min(cap, .5)) < 1e-7,
                          value=row['max_serine_availability']))
        if cap == 1.:
            frame = flux_variability_analysis(model, CYTO+OXI, processes=1, fraction_of_optimum=1.)
            for rid, values in frame.iterrows():
                fva_rows.append(dict(case_id=row['case_id'], variant='plus_formate_oxidation',
                                     condition='EAA_GLY_FOR_BG_NADPH_closed', reaction=rid,
                                     minimum=float(values.minimum), maximum=float(values.maximum),
                                     fixed_primary_serine_optimum=row['max_serine_availability'],
                                     face='primary_optimal_not_pFBA_face'))
    model = configure(template, 'plus_formate_oxidation', conditions['EAA_GLY_FOR'])
    model.reactions.BG_NADPH.bounds = (0, 0)
    model.reactions.EX_CO2_c.bounds = (0, 0)
    row = solve(model, dict(group='ablation', variant='plus_formate_oxidation', condition='EAA_GLY_FOR',
                           intervention='close_cytosolic_CO2_exit_and_background_NADPH'), rows, vectors)
    tests.append(dict(check='formate_redox_requires_CO2_disposal', passed=row['max_serine_availability'] < 1e-7,
                      value=row['max_serine_availability']))

    # A background energy budget is not allowed to create organic material.
    model = configure(template, 'plus_mito_GCS', conditions['EAA'])
    free = solve(model, dict(group='negative_control', variant='plus_mito_GCS', condition='EAA',
                            intervention='all_external_organic_inputs_closed'), rows, vectors)
    tests.append(dict(check='no_serine_without_organic_inputs', passed=free['max_serine_availability'] < 1e-7,
                      value=free['max_serine_availability']))
    for rid, stoich in [
        ('TEST_ATP_USE', {'HC00012_c': -1, 'HC00011_c': -1, 'HC00018_c': 1, 'HC00019_c': 1}),
        ('TEST_NADPH_USE', {'HC00015_c': -1, 'HC00016_c': 1}),
    ]:
        with model:
            model.reactions.BG_ATP.bounds = (0, 0)
            model.reactions.BG_NADPH.bounds = (0, 0)
            add(model, rid, stoich, (0, BIG), rid, 'Negative-control bookkeeping demand')
            model.objective = rid
            value = model.slim_optimize(error_value=None)
            tests.append(dict(check='no_source_free_'+rid, passed=abs(value) < 1e-7, value=float(value)))

    results = pd.DataFrame(rows)
    main = results[results.group.eq('main')]
    positive = main[main.condition.eq('EAA_SER')]
    tests.append(dict(check='direct_serine_input_available_all_variants',
                      passed=positive.max_serine_availability.ge(1.-1e-7).all(), value=float(positive.max_serine_availability.min())))
    tests.append(dict(check='all_internal_reactions_C_N_balanced', passed=True, value=0.))
    checks = pd.DataFrame(tests)
    assert checks.passed.all(), checks[~checks.passed].to_string()
    results.to_csv(OUT / 'capacity_results.csv', index=False)
    pd.DataFrame(vectors).to_csv(OUT / 'selected_flux_vectors.csv', index=False)
    pd.DataFrame(fva_rows).to_csv(OUT / 'route_fva.csv', index=False)
    checks.to_csv(OUT / 'verification_checks.csv', index=False)
    main.pivot(index='condition', columns='variant', values='max_serine_availability').to_csv(OUT / 'main_capacity_matrix.csv')
    baseline_path = ROOT / 'notebooks/serine_spatial_bounded_transfer/execution_manifest.json'
    protected = json.loads(baseline_path.read_text())['protected_files']
    changed = [name for name, expected in protected.items() if not (ROOT / name).is_file() or sha(ROOT/name) != expected]
    assert not changed, changed
    contract = dict(status='isolated_chemical_capability_test_complete', objective='maximize_DM_SER_c_then_pFBA',
        biological_scope='Possible serine withdrawal, NOT predicted biomass, division, survival or extracellular release.',
        units='arbitrary normalized capacity; no measured mmol/gDW/h or per-cell conversion',
        nominal_concentration_to_flux_mapping=False, source_human_cell_specific_capacities_measured=False,
        PHGDH_zero_meaning='Limiting deficient case; zero activity is not measured from absence/low expression.',
        GCS_zero_meaning='Hypothesis to test; no deletion or absence established in PATU-8902.',
        folate_scope='Internal steady-state recycling; pool size, PLP and enzyme abundance not modeled.',
        elemental_audit='Carbon and nitrogen only; source lacks full formulas/charges. Not full atom/charge or thermodynamic validation.',
        medium_scope='Only Ser/Gly/formate boundaries in main hypotheses; background metabolism is bounded ATP/redox support.',
        SSP_positive_control='Upstream 3PG and glutamate unit supplies, not modeled glucose/glutamine conversion.',
        sources=SOURCES, checks_passed=int(checks.passed.sum()), checks_total=len(checks), cases=len(results),
        FVA_rows=len(fva_rows), protected_files_checked=len(protected), protected_files_different=changed,
        source_hashes={str(NATIVE.relative_to(ROOT)): sha(NATIVE), str(protocol_source.relative_to(ROOT)): sha(protocol_source),
                       str((HERE / 'inspect_networks.py').relative_to(ROOT)): sha(HERE / 'inspect_networks.py'),
                       str((HERE / 'existing_network_inventory.json').relative_to(ROOT)): sha(HERE / 'existing_network_inventory.json'),
                       str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        output_hashes={str(p.relative_to(HERE)): sha(p) for p in itertools.chain(OUT.glob('*.csv'), (HERE/'models').glob('*.xml'))})
    (HERE / 'execution_manifest.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(main.pivot(index='condition', columns='variant', values='max_serine_availability').to_string())
    print(json.dumps({k: contract[k] for k in ['cases', 'checks_passed', 'checks_total', 'FVA_rows', 'protected_files_checked']}, indent=2))


if __name__ == '__main__':
    main()
