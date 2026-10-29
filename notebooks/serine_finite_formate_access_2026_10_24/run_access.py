"""Finite-stock hepatic C1 access diagnostic, not a culture calibration.

Restore only native formate exchange/antiport reactions. Keep the active
PhysiCell configuration and candidate SBML unchanged. No free cofactor ports.
"""
import hashlib
import json
import logging
from pathlib import Path
import re
import sys

import cobra
from cobra.flux_analysis import pfba, flux_variability_analysis
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/serine_reference_challenge_2026_10_23'))
from audit_reference import NATIVE, CANDIDATE, ATP, apply_published

SER = 'EX_HC00068_s'
FORMATE = 'EX_HC00061_s'
HOURS = 12.0
FACTOR = 0.12  # Inherited 12-cell equivalent, not a new dry-mass calibration.
VOLUME_NL = 20.0
STOCKS = [0.0, 0.144, 0.288, 1.0]
VARIANTS = {
    'original_no_access': [],
    'native_lactate_antiport': [FORMATE, 'r2117'],
    'native_pyruvate_antiport': [FORMATE, 'r2095'],
    'lactate_plus_NADP_branch': [FORMATE, 'r2117', 'r0294'],
}
BLOCKS = {
    'PHGDH_GCS': ['r0338', 'r0295'],
    'plus_SHMT1': ['r0338', 'r0295', 'r0230'],
    'plus_SHMT2': ['r0338', 'r0295', 'r0231'],
    'plus_both_SHMT': ['r0338', 'r0295', 'r0230', 'r0231'],
    'plus_mito_formate_activation': ['r0338', 'r0295', 'r0229'],
    'plus_mito_C1_reduction': ['r0338', 'r0295', 'r0292'],
    'plus_mito_formate_transport': ['r0338', 'r0295', 'r0888'],
    'plus_cyto_formate_activation': ['r0338', 'r0295', 'r0228'],
    'plus_cyto_C1_reduction': ['r0338', 'r0295', 'r0293'],
    'both_formate_activation': ['r0338', 'r0295', 'r0228', 'r0229'],
    'both_C1_reduction': ['r0338', 'r0295', 'r0292', 'r0293'],
}
READOUT = ['r0230', 'r0231', 'r0228', 'r0229', 'r0292', 'r0293',
           'r0294', 'r0295', 'r0888', 'r2117', 'r2095', ATP]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fields():
    fields = pd.read_csv(ROOT / 'notebooks/serine_redox_coculture_2026_10_22/fields.csv')
    fields.loc[fields.substrate == 'formate', 'cobra_exchange'] = FORMATE
    fields.loc[fields.substrate == 'oxygen', 'initial_pmol'] = 36.0
    fields.loc[fields.substrate == 'glycine', 'initial_pmol'] = 2.0
    fields.loc[fields.substrate == 'serine', 'initial_pmol'] = 0.0
    return fields[fields.cobra_exchange.notna()].copy()


def finite_envelope(model, fields, formate_stock):
    """Constant flux over 12 h, with gross uptake <= initial stock per field."""
    stocks = {}
    for field in fields.itertuples():
        amount = formate_stock if field.substrate == 'formate' else field.initial_pmol
        stocks[field.cobra_exchange] = float(amount)
        if field.cobra_exchange not in model.reactions:
            continue
        rxn = model.reactions.get_by_id(field.cobra_exchange)
        assert len(rxn.metabolites) == 1
        assert next(iter(rxn.metabolites.values())) == -1.0
        rxn.lower_bound = max(rxn.lower_bound, -amount / (HOURS * FACTOR))
    if FORMATE in model.reactions:
        model.reactions.get_by_id(FORMATE).upper_bound = 0.0
    model.reactions.get_by_id(ATP).bounds = (5.0, 5.0)
    model.reactions.get_by_id(SER).bounds = (0.0, 0.2)
    return stocks


def source_inventory(native, candidate):
    """Map immediate native C1 suppliers; do not infer atom tracing from net flux."""
    rows = []
    for mid in ['HC00061_c', 'HC00061_m', 'HC00140_m', 'HC00140_c', 'HC00483_c']:
        met = native.metabolites.get_by_id(mid)
        for rxn in sorted(met.reactions, key=lambda r: r.id):
            rows.append(dict(target=mid, target_name=met.name, reaction=rxn.id,
                equation=rxn.reaction, lb=rxn.lower_bound, ub=rxn.upper_bound,
                present_in_candidate=rxn.id in candidate.reactions,
                participants='; '.join(f'{m.id}: {m.name}' for m in rxn.metabolites)))
    pd.DataFrame(rows).to_csv(HERE / 'native_C1_source_inventory.csv', index=False)
    methyl_donors = []
    for met in native.metabolites:
        if not re.search(r'(^choline|betaine|sarcosine|dimethylglycine|formaldehyde)', met.name, re.I):
            continue
        for rxn in sorted(met.reactions, key=lambda r: r.id):
            methyl_donors.append(dict(metabolite=met.id, name=met.name,
                reaction=rxn.id, equation=rxn.reaction,
                present_in_candidate=rxn.id in candidate.reactions))
    pd.DataFrame(methyl_donors).to_csv(HERE / 'choline_betaine_inventory.csv', index=False)


def energy_gates(model, variant):
    rows = []
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
            rows.append(dict(variant=variant, gate=gate, status=model.solver.status,
                             ATP_capacity=value, passed=passed))
    return rows


def main():
    HERE.mkdir(exist_ok=True)
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    protected_paths = [NATIVE, CANDIDATE, Path(__file__),
        ROOT / 'config/PhysiCell_settings.xml',
        ROOT / 'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT / 'notebooks/serine_redox_coculture_2026_10_22/fields.csv']
    protected = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    protocol = dict(protected=protected, variants=VARIANTS, blocks=BLOCKS,
        formate_stocks_pmol=STOCKS, duration_h=HOURS, volume_nL=VOLUME_NL,
        local_flux_to_pmol_h_factor=FACTOR, glycine_stock_pmol=2.0,
        serine_stock_pmol=0.0, oxygen_stock_pmol=36.0,
        ATP_maintenance_normalized=5.0, serine_export_ceiling_normalized=0.2,
        declared_before_execution=True,
        resource_controls=['no_glycine', 'antiport_closed', 'no_formate'],
        scope='Isolated hepatic capacity; constant allocation, no tumour, growth, signalling or fitted rates',
        stock_selection='Algebraic half/full export-ceiling requirements and excess; not physiological concentrations',
        limitation='Inherited placeholder amino-acid medium; no thermodynamic concentration constraints; no isotope fit')
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    native = cobra.io.read_sbml_model(str(NATIVE))
    apply_published(native)
    candidate = cobra.io.read_sbml_model(str(CANDIDATE))
    fields = load_fields()
    fields.to_csv(HERE / 'finite_fields.csv', index=False)
    source_inventory(native, candidate)
    rows, fluxes, inventories, ranges, gates, equations = [], [], [], [], [], []
    resource_controls = []
    for variant, additions in VARIANTS.items():
        model = candidate.copy()
        for rid in additions:
            assert rid not in model.reactions
            rxn = native.reactions.get_by_id(rid).copy()
            model.add_reactions([rxn])
            equations.append(dict(variant=variant, reaction=rid, equation=rxn.reaction,
                                  lb=rxn.lower_bound, ub=rxn.upper_bound))
        model.solver.configuration.timeout = 30
        model.solver.configuration.tolerances.feasibility = 1e-9
        own_gates = energy_gates(model, variant)
        gates.extend(own_gates)
        assert all(r['passed'] for r in own_gates), f'Failed energy gate: {variant}'
        selected_blocks = BLOCKS if variant == 'native_lactate_antiport' else {
            k: BLOCKS[k] for k in ['PHGDH_GCS', 'plus_SHMT1', 'plus_SHMT2', 'plus_both_SHMT']}
        for stock in STOCKS:
            for block, ids in selected_blocks.items():
                with model:
                    stocks = finite_envelope(model, fields, stock)
                    for rid in ids:
                        model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                    model.objective = SER
                    model.objective_direction = 'max'
                    optimum = model.slim_optimize()
                    assert model.solver.status == 'optimal', (variant, stock, block)
                    solution = pfba(model)
                    v = solution.fluxes
                    serine = max(0.0, float(v[SER] * FACTOR * HOURS))
                    sv = float(np.max(np.abs(cobra.util.array.create_stoichiometric_matrix(model) @ v.to_numpy())))
                    bound_error = max(max(r.lower_bound - v[r.id], v[r.id] - r.upper_bound, 0.0)
                                      for r in model.reactions)
                    c_net = n_net = 0.0
                    min_stock = float('inf')
                    for field in fields.itertuples():
                        rid = field.cobra_exchange
                        amount = float(v[rid] * FACTOR * HOURS) if rid in model.reactions else 0.0
                        final = stocks[rid] + amount
                        min_stock = min(min_stock, final)
                        c_net += amount * field.carbon_atoms
                        n_net += amount * field.nitrogen_atoms
                        inventories.append(dict(variant=variant, formate_stock_pmol=stock,
                            block=block, substrate=field.substrate, initial_pmol=stocks[rid],
                            change_pmol=amount, final_pmol=final))
                    assert sv < 1e-7 and bound_error < 1e-7
                    assert min_stock > -1e-7 and abs(c_net) < 1e-7 and abs(n_net) < 1e-7
                    assert abs(v[SER] - optimum) < 1e-7
                    if block == 'plus_both_SHMT' or stock == 0 or variant == 'original_no_access':
                        assert serine < 1e-7, (variant, stock, block, serine)
                    if block in {'both_formate_activation', 'both_C1_reduction'}:
                        assert serine < 1e-7
                    if variant != 'original_no_access' and block in {
                            'PHGDH_GCS', 'plus_SHMT1', 'plus_SHMT2'}:
                        assert abs(serine - min(stock, 0.288)) < 1e-7
                    row = dict(variant=variant, formate_stock_pmol=stock,
                        formate_initial_uM=stock * 1000 / VOLUME_NL, block=block,
                        status=solution.status, serine_export_pmol=serine,
                        formate_uptake_pmol=max(0.0, -float(v.get(FORMATE, 0.0)) * FACTOR * HOURS),
                        glycine_uptake_pmol=max(0.0, -float(v['EX_HC00045_s']) * FACTOR * HOURS),
                        max_Sv=sv, max_bound_error=float(bound_error), min_final_stock=min_stock,
                        carbon_residual_pmol=c_net, nitrogen_residual_pmol=n_net,
                        ATP_maintenance_pmol=float(v[ATP] * FACTOR * HOURS))
                    for rid in READOUT:
                        row[rid + '_pmol'] = float(v.get(rid, 0.0) * FACTOR * HOURS)
                    rows.append(row)
                    for rid, value in v.items():
                        if abs(value) > 1e-9:
                            fluxes.append(dict(variant=variant, formate_stock_pmol=stock,
                                block=block, reaction=rid, flux_normalized=float(value),
                                amount_pmol=float(value * FACTOR * HOURS),
                                equation=model.reactions.get_by_id(rid).reaction))
                    # At the optimum, distinguish a selected pFBA route from necessity.
                    if stock == 0.288 and variant != 'original_no_access' and block in {
                            'PHGDH_GCS', 'plus_SHMT1', 'plus_SHMT2'}:
                        targets = [rid for rid in READOUT if rid in model.reactions]
                        fva = flux_variability_analysis(model, targets, fraction_of_optimum=1.0, processes=1)
                        for rid, values in fva.iterrows():
                            ranges.append(dict(variant=variant, block=block, reaction=rid,
                                formate_stock_pmol=stock,
                                minimum_pmol=float(values['minimum'] * FACTOR * HOURS),
                                maximum_pmol=float(values['maximum'] * FACTOR * HOURS)))
        if additions:
            for control in ['no_glycine', 'antiport_closed', 'no_formate']:
                with model:
                    finite_envelope(model, fields, 0.288)
                    for rid in BLOCKS['PHGDH_GCS']:
                        model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
                    target = {'no_glycine': 'EX_HC00045_s',
                              'antiport_closed': 'r2095' if 'r2095' in additions else 'r2117',
                              'no_formate': FORMATE}[control]
                    model.reactions.get_by_id(target).bounds = (0.0, 0.0)
                    model.objective = SER
                    model.objective_direction = 'max'
                    optimum = model.slim_optimize()
                    assert model.solver.status == 'optimal' and abs(optimum) < 1e-7
                    resource_controls.append(dict(variant=variant, control=control,
                        status=model.solver.status, serine_export_pmol=optimum * FACTOR * HOURS))
        print(variant, 'completed', flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'capacity_tests.csv', index=False)
    pd.DataFrame(gates).to_csv(HERE / 'energy_gates.csv', index=False)
    pd.DataFrame(fluxes).to_csv(HERE / 'active_fluxes.csv', index=False)
    pd.DataFrame(inventories).to_csv(HERE / 'stock_balances.csv', index=False)
    pd.DataFrame(ranges).to_csv(HERE / 'route_FVA.csv', index=False)
    pd.DataFrame(equations).to_csv(HERE / 'restored_native_equations.csv', index=False)
    pd.DataFrame(resource_controls).to_csv(HERE / 'resource_controls.csv', index=False)
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    result = dict(tests=len(rows), energy_gates=len(gates),
        resource_controls=len(resource_controls), all_optimal=True,
        protected_unchanged=True, max_Sv=max(r['max_Sv'] for r in rows),
        max_carbon_residual=max(abs(r['carbon_residual_pmol']) for r in rows),
        max_nitrogen_residual=max(abs(r['nitrogen_residual_pmol']) for r in rows),
        minimum_final_stock=min(r['min_final_stock'] for r in rows),
        experimental_rates_fitted=False, tumour_support_demonstrated=False)
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
