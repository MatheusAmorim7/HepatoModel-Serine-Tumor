"""Assay of downstream methyl donors, not complete choline/betaine uptake.

DMG/SAR are finite matrix-accessible diagnostic pools. ETF denotes a formal
two-electron ledger, not protein abundance. No free electron sink is allowed.
"""
import json
import logging
from pathlib import Path
import sys

import cobra
from cobra.flux_analysis import pfba, flux_variability_analysis
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/serine_finite_formate_access_2026_10_24'))
from run_access import (CANDIDATE, NATIVE, ATP, SER, HOURS, FACTOR, sha,
                        load_fields, finite_envelope, apply_published)

MODES = ['native_Q', 'terminal_noATP']
BLOCKS = {'none': [], 'DMGDH_reaction': ['CUR_DMGDH'],
          'SARDH_reaction': ['CUR_SARDH'], 'SHMT1': ['r0230'],
          'SHMT2': ['r0231'], 'both_SHMT': ['r0230', 'r0231'],
          'ETF_reoxidation': ['CUR_ETF_REOX']}
DONORS = {'DMG': 'ASSAY_DMG_m', 'SAR': 'ASSAY_SAR_m'}
READOUT = ['CUR_DMGDH', 'CUR_SARDH', 'CUR_ETF_REOX', 'r0230', 'r0231',
           'r0292', 'r0229', 'r0507', 'r0508', 'r0017', ATP]


def add_reaction(model, rid, stoich, bounds=(0.0, 1000.0)):
    rxn = cobra.Reaction(rid)
    rxn.bounds = bounds
    rxn.add_metabolites({model.metabolites.get_by_id(mid): coef for mid, coef in stoich.items()})
    model.add_reactions([rxn])


def build(candidate, mode):
    model = candidate.copy()
    for mid, name, formula in [
            ('CUR_DMG_m', 'Diagnostic matrix dimethylglycine', 'C4H9NO2'),
            ('CUR_SAR_m', 'Diagnostic matrix sarcosine', 'C3H7NO2'),
            ('CUR_ETF_ox_m', 'Formal oxidized two-electron carrier', None),
            ('CUR_ETF_red_m', 'Formal reduced two-electron carrier', None)]:
        met = cobra.Metabolite(mid, name=name, compartment='m', formula=formula)
        model.add_metabolites([met])
    add_reaction(model, 'ASSAY_DMG_m', {'CUR_DMG_m': -1}, (0.0, 1000.0))
    add_reaction(model, 'ASSAY_SAR_m', {'CUR_SAR_m': -1}, (0.0, 1000.0))
    add_reaction(model, 'CUR_DMGDH', {'CUR_DMG_m': -1, 'HC00102_m': -1,
        'CUR_ETF_ox_m': -1, 'CUR_SAR_m': 1, 'HC00140_m': 1, 'CUR_ETF_red_m': 1})
    add_reaction(model, 'CUR_SARDH', {'CUR_SAR_m': -1, 'HC00102_m': -1,
        'CUR_ETF_ox_m': -1, 'HC00045_m': 1, 'HC00140_m': 1, 'CUR_ETF_red_m': 1})
    if mode == 'native_Q':
        # Net ETFDH transfer; respiratory ATP, if any, comes from native reactions.
        reox = {'CUR_ETF_red_m': -1, 'HC00329_m': -1,
                'CUR_ETF_ox_m': 1, 'HC00324_m': 1}
    elif mode == 'terminal_noATP':
        # Effective respiratory disposal, not a direct O2 reaction of DMGDH/SARDH.
        reox = {'CUR_ETF_red_m': -1, 'HC00017_m': -0.5,
                'CUR_ETF_ox_m': 1, 'HC00011_m': 1}
    else:
        raise ValueError(mode)
    add_reaction(model, 'CUR_ETF_REOX', reox)
    model.solver.configuration.timeout = 30
    model.solver.configuration.tolerances.feasibility = 1e-9
    return model


def fields_for(glycine):
    fields = load_fields()
    fields.loc[fields.substrate == 'glycine', 'initial_pmol'] = glycine
    additions = pd.DataFrame([
        dict(substrate='diagnostic_DMG', cobra_exchange='ASSAY_DMG_m',
             carbon_atoms=4, nitrogen_atoms=1, initial_pmol=0.0),
        dict(substrate='diagnostic_SAR', cobra_exchange='ASSAY_SAR_m',
             carbon_atoms=3, nitrogen_atoms=1, initial_pmol=0.0)])
    return pd.concat([fields, additions], ignore_index=True)


def build_disposal(candidate, native, mode):
    model = build(candidate, mode)
    model.add_reactions([native.reactions.get_by_id(rid).copy()
                         for rid in ['EX_HC00061_s', 'r2117']])
    # Literature-guided direction sensitivity, not a change to the original SBML.
    model.reactions.get_by_id('r0229').lower_bound = -1000.0
    model.reactions.get_by_id('r0888').lower_bound = -1000.0
    return model


def envelope(model, source, glycine, donor_stock=0.144, allow_formate_exit=False):
    fields = fields_for(glycine)
    stocks = finite_envelope(model, fields, 0.0)
    rid = DONORS[source]
    stocks[rid] = donor_stock
    model.reactions.get_by_id(rid).lower_bound = -donor_stock / (HOURS * FACTOR)
    if allow_formate_exit:
        model.reactions.get_by_id('EX_HC00061_s').bounds = (0.0, 1000.0)
    for target in ['r0338', 'r0295']:
        model.reactions.get_by_id(target).bounds = (0.0, 0.0)
    return fields, stocks


def energy_gates(model, mode):
    result = []
    inorganic = {'EX_HC00011_s', 'EX_HC00019_s', 'EX_HC00253_s',
                 'EX_HC00017_s', 'EX_HC00021_s', 'EX_HC00024_s'}
    for gate in ['closed', 'no_organic']:
        with model:
            for rxn in model.boundary:
                if gate == 'closed' or rxn.id not in inorganic:
                    rxn.bounds = (0.0, 0.0)
            model.reactions.get_by_id(ATP).bounds = (0.0, 1000.0)
            model.objective = ATP
            model.objective_direction = 'max'
            value = model.slim_optimize()
            passed = model.solver.status == 'optimal' and abs(value) < 1e-7
            result.append(dict(mode=mode, gate=gate, status=model.solver.status,
                               ATP_capacity=value, passed=passed))
    return result


def formal_element_checks():
    # Neutral transformed folates and a formal 2H carrier: not literal protein formulas.
    formulas = {'DMG': {'C': 4, 'H': 9, 'N': 1, 'O': 2},
                'SAR': {'C': 3, 'H': 7, 'N': 1, 'O': 2},
                'GLY': {'C': 2, 'H': 5, 'N': 1, 'O': 2},
                'THF': {'C': 19, 'H': 23, 'N': 7, 'O': 6},
                'CH2THF': {'C': 20, 'H': 23, 'N': 7, 'O': 6},
                'EOX': {}, 'ERED': {'H': 2}, 'O2': {'O': 2},
                'H2O': {'H': 2, 'O': 1}}
    equations = {
        'DMGDH': {'DMG': -1, 'THF': -1, 'EOX': -1, 'SAR': 1, 'CH2THF': 1, 'ERED': 1},
        'SARDH': {'SAR': -1, 'THF': -1, 'EOX': -1, 'GLY': 1, 'CH2THF': 1, 'ERED': 1},
        'terminal_noATP': {'ERED': -1, 'O2': -0.5, 'EOX': 1, 'H2O': 1}}
    rows = []
    for rid, equation in equations.items():
        for element in ['C', 'H', 'N', 'O']:
            residual = sum(coef * formulas[mid].get(element, 0) for mid, coef in equation.items())
            assert abs(residual) < 1e-12
            rows.append(dict(reaction=rid, element=element, residual=residual,
                             scope='neutral transformed diagnostic representation'))
    pd.DataFrame(rows).to_csv(HERE / 'formal_element_checks.csv', index=False)


def check_witness(model, fields, stocks, v):
    sv = float(np.max(np.abs(cobra.util.array.create_stoichiometric_matrix(model) @ v.to_numpy())))
    bounds = max(max(r.lower_bound - v[r.id], v[r.id] - r.upper_bound, 0.0) for r in model.reactions)
    inventory, carbon, nitrogen = [], 0.0, 0.0
    for field in fields.itertuples():
        amount = float(v.get(field.cobra_exchange, 0.0) * HOURS * FACTOR)
        final = stocks[field.cobra_exchange] + amount
        assert final > -1e-7
        carbon += amount * field.carbon_atoms
        nitrogen += amount * field.nitrogen_atoms
        inventory.append(dict(substrate=field.substrate, initial_pmol=stocks[field.cobra_exchange],
                              change_pmol=amount, final_pmol=final))
    assert max(sv, bounds, abs(carbon), abs(nitrogen)) < 1e-7
    assert abs(v['CUR_DMGDH'] + v['CUR_SARDH'] - v['CUR_ETF_REOX']) < 1e-7
    return inventory, dict(max_Sv=sv, max_bound_error=float(bounds),
                           carbon_residual_pmol=carbon, nitrogen_residual_pmol=nitrogen)


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    source_manifest = json.loads((HERE / 'source_manifest.json').read_text())
    assert len(source_manifest) == 5 and all(item['status'] == 'archived' for item in source_manifest)
    for item in source_manifest:
        accession = item['url'].split('/')[-1]
        assert sha(HERE / 'sources' / accession) == item['sha256']
    protected_paths = [CANDIDATE, NATIVE, Path(__file__),
        ROOT / 'config/PhysiCell_settings.xml',
        ROOT / 'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT / 'notebooks/serine_finite_formate_access_2026_10_24/run_access.py',
        HERE / 'curation_decisions.json', HERE / 'source_manifest.json']
    protected_paths += sorted((HERE / 'sources').glob('*.json'))
    protected = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    protocol = dict(protected=protected, modes=MODES, blocks=BLOCKS,
        donor_stocks_pmol={'DMG': 0.144, 'SAR': 0.144}, glycine_stocks_pmol=[0.0, 2.0],
        hours=HOURS, flux_to_pmol_h=FACTOR, ATP_maintenance_normalized=5.0,
        declared_before_execution=True,
        scope='Finite matrix-accessible donor assay, no beta/choline uptake, tumour, growth or rate calibration',
        limitations=['Transport costs/capacities not identified; matrix access is an assay assumption',
                     'DMGDH sarcosine side activity not included; reaction blocks are not selective gene/drug effects',
                     'ETF pair is a collapsed two-electron ledger, not enzyme abundance or literal ETF chemistry',
                     'Inherited medium includes placeholders; no concentration-dependent thermodynamics'])
    protocol['controls'] = ['no_donor', 'both_dehydrogenases_closed', 'no_oxygen',
                            'no_oxygen_maintenance_optional']
    protocol['disposal_sensitivity'] = dict(donor='DMG', glycine_pmol=[0.0, 2.0],
        blocks=['none', 'SHMT1'], modes=MODES,
        added_native=['EX_HC00061_s', 'r2117'],
        direction_sensitivities={'r0229': [-1000.0, 1000.0], 'r0888': [-1000.0, 1000.0]},
        formate_exchange_bounds=[0.0, 1000.0],
        reference='https://doi.org/10.1126/sciadv.1601273',
        interpretation='Test disposal of excess C1; not a fitted hepatocyte formate export rate')
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    formal_element_checks()
    candidate = cobra.io.read_sbml_model(str(CANDIDATE))
    rows, controls, gates, fluxes, inventories, ranges, equations = [], [], [], [], [], [], []
    disposal_rows, disposal_fluxes, disposal_stocks = [], [], []
    for mode in MODES:
        model = build(candidate, mode)
        for rid in ['CUR_DMGDH', 'CUR_SARDH', 'CUR_ETF_REOX', *DONORS.values()]:
            rxn = model.reactions.get_by_id(rid)
            equations.append(dict(mode=mode, reaction=rid, equation=rxn.reaction,
                                  lb=rxn.lower_bound, ub=rxn.upper_bound))
        own_gates = energy_gates(model, mode)
        gates.extend(own_gates)
        assert all(g['passed'] for g in own_gates), (mode, own_gates)
        for source in DONORS:
            for glycine in [0.0, 2.0]:
                for block, ids in BLOCKS.items():
                    with model:
                        fields, stocks = envelope(model, source, glycine)
                        for rid in ids:
                            model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                        model.objective = SER
                        model.objective_direction = 'max'
                        optimum = model.slim_optimize()
                        assert model.solver.status == 'optimal', (mode, source, glycine, block)
                        sol = pfba(model)
                        v = sol.fluxes
                        inventory, errors = check_witness(model, fields, stocks, v)
                        assert abs(v[ATP] - 5.0) < 1e-7 and abs(v[SER] - optimum) < 1e-7
                        serine = max(0.0, float(v[SER] * HOURS * FACTOR))
                        if block in ['ETF_reoxidation', 'both_SHMT']:
                            assert serine < 1e-7
                        tag = dict(mode=mode, donor=source, glycine_stock_pmol=glycine, block=block)
                        row = tag | dict(status=sol.status, serine_export_pmol=serine,
                            donor_uptake_pmol=-float(v[DONORS[source]] * HOURS * FACTOR),
                            oxygen_uptake_pmol=-float(v['EX_HC00017_s'] * HOURS * FACTOR)) | errors
                        row.update({rid + '_pmol': float(v[rid] * HOURS * FACTOR) for rid in READOUT})
                        rows.append(row)
                        inventories += [tag | item for item in inventory]
                        fluxes += [tag | dict(reaction=rid, flux_normalized=float(value),
                            amount_pmol=float(value * HOURS * FACTOR),
                            equation=model.reactions.get_by_id(rid).reaction)
                            for rid, value in v.items() if abs(value) > 1e-9]
                        if block == 'none':
                            fva = flux_variability_analysis(model, READOUT[:-1], fraction_of_optimum=1.0, processes=1)
                            ranges += [tag | dict(reaction=rid,
                                minimum_pmol=float(values['minimum'] * HOURS * FACTOR),
                                maximum_pmol=float(values['maximum'] * HOURS * FACTOR))
                                for rid, values in fva.iterrows()]
                for control in protocol['controls']:
                    with model:
                        envelope(model, source, glycine, 0.0 if control == 'no_donor' else 0.144)
                        if control == 'both_dehydrogenases_closed':
                            for rid in ['CUR_DMGDH', 'CUR_SARDH']:
                                model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                        if control.startswith('no_oxygen'):
                            model.reactions.get_by_id('EX_HC00017_s').bounds = (0.0, 0.0)
                        if control == 'no_oxygen_maintenance_optional':
                            model.reactions.get_by_id(ATP).bounds = (0.0, 5.0)
                        model.objective = SER
                        model.objective_direction = 'max'
                        value = model.slim_optimize()
                        status = model.solver.status
                        if not control.startswith('no_oxygen'):
                            assert status == 'optimal' and abs(value) < 1e-7
                        if control == 'no_oxygen_maintenance_optional':
                            assert status == 'optimal'
                        controls.append(dict(mode=mode, donor=source, glycine_stock_pmol=glycine,
                            control=control, status=status,
                            serine_export_pmol=float(value * HOURS * FACTOR) if status == 'optimal' else None))
        print(mode, 'completed', flush=True)
    native = cobra.io.read_sbml_model(str(NATIVE))
    apply_published(native)
    for mode in MODES:
        model = build_disposal(candidate, native, mode)
        own_gates = energy_gates(model, mode + '_formate_exit')
        gates.extend(own_gates)
        assert all(g['passed'] for g in own_gates), (mode, own_gates)
        for glycine in [0.0, 2.0]:
            for block in ['none', 'SHMT1']:
                with model:
                    fields, stocks = envelope(model, 'DMG', glycine, allow_formate_exit=True)
                    for rid in BLOCKS[block]:
                        model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                    model.objective = SER
                    model.objective_direction = 'max'
                    optimum = model.slim_optimize()
                    assert model.solver.status == 'optimal'
                    sol = pfba(model)
                    v = sol.fluxes
                    inventory, errors = check_witness(model, fields, stocks, v)
                    assert v['EX_HC00061_s'] > -1e-7 and abs(v[ATP] - 5.0) < 1e-7
                    assert abs(v[SER] - optimum) < 1e-7
                    tag = dict(mode=mode, donor='DMG', glycine_stock_pmol=glycine, block=block)
                    disposal_rows.append(tag | dict(status=sol.status,
                        serine_export_pmol=max(0.0, float(v[SER] * HOURS * FACTOR)),
                        formate_export_pmol=float(v['EX_HC00061_s'] * HOURS * FACTOR),
                        r0229_pmol=float(v['r0229'] * HOURS * FACTOR),
                        r0888_pmol=float(v['r0888'] * HOURS * FACTOR)) | errors)
                    disposal_stocks += [tag | item for item in inventory]
                    disposal_fluxes += [tag | dict(reaction=rid, flux_normalized=float(value),
                        amount_pmol=float(value * HOURS * FACTOR),
                        equation=model.reactions.get_by_id(rid).reaction)
                        for rid, value in v.items() if abs(value) > 1e-9]
        print(mode, 'formate disposal completed', flush=True)
    for filename, records in [('capacity_tests.csv', rows), ('resource_controls.csv', controls),
            ('energy_gates.csv', gates), ('active_fluxes.csv', fluxes),
            ('stock_balances.csv', inventories), ('route_FVA.csv', ranges),
            ('diagnostic_equations.csv', equations), ('disposal_sensitivity.csv', disposal_rows),
            ('disposal_fluxes.csv', disposal_fluxes), ('disposal_stocks.csv', disposal_stocks)]:
        pd.DataFrame(records).to_csv(HERE / filename, index=False)
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    verification = dict(capacity_tests=len(rows), resource_controls=len(controls),
        energy_gates=len(gates), disposal_tests=len(disposal_rows), all_capacity_tests_optimal=True,
        protected_unchanged=True, max_Sv=max(r['max_Sv'] for r in rows),
        max_external_CN_residual=max(max(abs(r['carbon_residual_pmol']),
            abs(r['nitrogen_residual_pmol'])) for r in rows),
        complete_choline_betaine_path_implemented=False,
        experimental_rates_fitted=False, tumour_support_demonstrated=False)
    (HERE / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
