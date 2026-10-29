"""Freeze public evidence roles and replay hepatic blocks under constant schedules."""
import hashlib
import argparse
import json
import logging
from pathlib import Path
import sys

import cobra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT/'notebooks/serine_redox_coculture_2026_10_22'
EVIDENCE = ROOT/'notebooks/serine_temporal_constraints_2026_10_22'
sys.path.insert(0, str(PRIOR))
from audit_constant_protocol import initial_stock
from run_coculture import RedoxJoint, previous, build, ROLES, SER_TASK, C1_TASK


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(spec, fields, template, native, q):
    joint = RedoxJoint(spec, fields, template, native, q)
    initial, idx = initial_stock(spec, fields)
    total = initial.copy()
    total[idx['oxygen']] = min(initial[idx['oxygen']]+36., 3.*spec['horizon'])
    solved = joint.solve(total, spec['horizon'])
    stock = initial.copy()
    external = np.zeros(len(stock))
    corrections = np.zeros(len(stock))
    donor = 36.
    net = solved['hnet']+solved['tnet']
    gross = np.zeros(len(stock))
    flux_rows = []
    for g in joint.ids:
        for rid, value in zip(joint.ids[g], solved['vectors'][g]):
            r = joint.r[g][rid]
            assert r.lower_bound-1e-7 <= value <= r.upper_bound+1e-7
            flux_rows.append(dict(group=g, reaction=rid, flux=value,
                                  lb=r.lower_bound, ub=r.upper_bound))
    for j, row in enumerate(fields.itertuples()):
        for g in joint.ids:
            rid = ROLES.get(row.substrate) if g == 'tumor' else row.cobra_exchange
            if rid in joint.r[g]:
                gross[j] += max(0., -joint.r[g][rid].flux*joint.factor[g])
    trace = []
    for step in range(round(spec['horizon']/spec['dt'])):
        allowed = stock/spec['dt']
        allowed[idx['serine']] += solved['export']
        pool_error = max(0., float((gross-allowed).max()))
        raw = stock+spec['dt']*net
        correction = np.maximum(0., -raw)
        assert correction.max() < 1e-10 and pool_error < 1e-7
        stock = raw+correction
        corrections += correction
        delivered = min(max(0., 4.-stock[idx['oxygen']]), 3.*spec['dt'], donor)
        donor -= delivered
        external[idx['oxygen']] += delivered
        stock[idx['oxygen']] += delivered
        t = (step+1)*spec['dt']
        balance = stock-initial-external-corrections-t*net
        cn = (stock-initial-external-corrections)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
        cn += t*solved['theta']*np.array([3*SER_TASK+C1_TASK, SER_TASK])
        assert abs(balance).max() < 1e-7 and abs(cn).max() < 1e-7
        trace.append(dict(time_h=t, serine_pmol=stock[idx['serine']],
            oxygen_pmol=stock[idx['oxygen']], oxygen_delivered_pmol=delivered,
            pool_error=pool_error, inventory_error=float(abs(balance).max()),
            CN_error=float(abs(cn).max())))
    out = HERE/'constant_interventions'/spec['case']
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(out/'trajectory.csv', index=False)
    pd.DataFrame(flux_rows).to_csv(out/'fluxes.csv', index=False)
    return dict(case=spec['case'], tumor_variant=spec['tumor_variant'],
        capacity_fraction=spec['capacity_fraction'], arm=spec['arm'],
        task_h=solved['theta']*spec['horizon'], export_pmol=solved['export']*spec['horizon'],
        hepatic_maintenance=solved['hep'], tumor_maintenance=solved['maintenance'],
        oxygen_delivered_pmol=float(external[idx['oxygen']]),
        max_pool_error=max(r['pool_error'] for r in trace),
        max_inventory_error=max(r['inventory_error'] for r in trace),
        max_CN_error=max(r['CN_error'] for r in trace))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-results', action='store_true', help='Regenerate summaries from saved constant interventions')
    args = parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver = 'glpk'
    assert json.loads((PRIOR/'verification.json').read_text())['all_passed']
    source_files = [p for folder in [PRIOR, EVIDENCE] for p in folder.iterdir() if p.is_file()]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    inherited = json.loads((PRIOR/'protocol.json').read_text())
    protected.update(inherited['protected_files'])
    assert all(sha(ROOT/p) == h for p, h in protected.items())
    specs = [s for s in inherited['cases'] if s['family'] == 'core' and s['feed_rate'] == 0
             and s['arm'] in ['PHGDH', 'PHGDH_SHMT']]
    assert len(specs) == 14
    decision = dict(data_scope='public_only', parameters_fitted=False,
        biological_reference='PATU-8902: extracellular serine dependence and conditional Gly+formate rescue',
        functional_reference='Banh2020 S2; primary-site PDAC, not a liver assay',
        corroborating_challenge='Shin2026 S5E nominal additions; relative confluence',
        quantitative_protocol_reproduction_ready=False,
        quantitative_protocol_blockers=['matched medium volume', 'viable cell trajectory',
            'nutrient time series', 'mapping tasks to measured cellular response'],
        computational_reference=dict(volume_nL=20, horizon_h=12,
            population_equivalents=dict(basal=252, local=12, tumor=7),
            Ser_uM=25, Gly_uM=100, formate_uM=500,
            scope='Inherited diagnostic; no reproduced experimental assay'),
        independent_validation=dict(reserved_dataset=None,
            reason='All selected sources have already informed route selection or model assessment.',
            rule='Freeze protocol and predictions before inspecting outcomes of a new eligible dataset.'),
        temporal_comparison='Stepwise selection and feasible constant schedules; neither calibrated nor global-optimum certificate',
        new_cases=specs, declared_before_execution=True,
        protected_inputs=protected)
    (HERE/'reference_decision.json').write_text(json.dumps(decision, indent=2))
    ledger = json.loads((EVIDENCE/'evidence_ledger.json').read_text())
    roles = dict(Banh2020=('functional reference', 'Route availability and qualitative nutrient response'),
        Shin2026=('corroborating challenge already inspected', 'Nominal additions and relative response; no direct flux fit'),
        Parker2020=('conditional magnitude', 'Net Ser exchange in source medium; not transport Vmax'),
        Beckers2011=('hepatic temporal direction', 'Protocol-specific extracellular concentrations'),
        Boon2020=('hepatic basal challenge', 'Donor-specific relative depletion'),
        Grankvist2024=('hepatic amino-acid constraint', 'Paired relative Gln/Glu in its own protocol'),
        Freyer2019=('hepatic functional challenge', 'Glucose/lactate/urea/ammonia in source protocol'))
    matrix = []
    for source in ledger['sources']:
        role, target = roles[source['id']]
        matrix.append(dict(source=source['id'], url=source['url'], role=role, target=target,
            usable=source['can_use'], limitation=source['cannot_use'],
            quantitative_gate=source['calibration_gate'], already_inspected=True,
            independent_holdout=False))
    pd.DataFrame(matrix).to_csv(HERE/'evidence_to_test.csv', index=False)
    fields = pd.read_csv(PRIOR/'fields.csv').fillna({'cobra_exchange':''})
    native = cobra.io.read_sbml_model(str(previous.NATIVE))
    template = build(native)
    q = json.loads(previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    results = []
    for spec in specs:
        if args.reuse_results:
            saved = pd.read_csv(HERE/'constant_interventions.csv')
            results.append(saved[saved['case'].eq(spec['case'])].iloc[0].to_dict())
        else:
            results.append(replay(spec, fields, template, native, q))
        print(spec['case'], round(results[-1]['task_h'], 6), flush=True)
    new = pd.DataFrame(results)
    new.to_csv(HERE/'constant_interventions.csv', index=False)
    old = pd.read_csv(PRIOR/'constant_protocol_comparison.csv').rename(columns={'constant_feasible_task_h':'task_h'})
    constant = pd.concat([old[['case','tumor_variant','capacity_fraction','arm','task_h','export_pmol']],
                          new[['case','tumor_variant','capacity_fraction','arm','task_h','export_pmol']]])
    step = pd.read_csv(PRIOR/'summary.csv')
    step = step[step.family.eq('core') & step.feed_rate.eq(0)]
    effects = []
    for schedule, frame in [('stepwise', step), ('constant_feasible', constant)]:
        for (variant, capacity), group in frame.groupby(['tumor_variant','capacity_fraction']):
            by_arm = group.set_index('arm')
            assert set(by_arm.index) == {'closed','funded','PHGDH','PHGDH_SHMT'}
            baseline = by_arm.loc['funded']
            for arm in ['closed','PHGDH','PHGDH_SHMT']:
                current = by_arm.loc[arm]
                effects.append(dict(schedule=schedule, tumor_variant=variant, capacity_fraction=capacity,
                    intervention=arm, source_on_task_h=baseline.task_h, intervention_task_h=current.task_h,
                    task_loss_h=baseline.task_h-current.task_h,
                    source_on_export_pmol=baseline.export_pmol, intervention_export_pmol=current.export_pmol))
    effects = pd.DataFrame(effects)
    effects.to_csv(HERE/'intervention_comparison.csv', index=False)
    compact = []
    for (variant, capacity, arm), group in effects.groupby(['tumor_variant','capacity_fraction','intervention']):
        compact.append(dict(tumor_variant=variant, capacity_fraction=capacity, intervention=arm,
            min_loss_in_tested_schedules_h=group.task_loss_h.min(),
            max_loss_in_tested_schedules_h=group.task_loss_h.max(),
            both_losses_above_0_05h=bool((group.task_loss_h > .05).all()),
            scope='Two tested schedules only; not a bound over all feasible trajectories'))
    pd.DataFrame(compact).to_csv(HERE/'schedule_sensitivity.csv', index=False)
    labels = dict(direct_only='Sem alternativa', cytosolic_PPP='Gly/formiato + PPP',
                  cytosolic_formate='Gly/formiato + ALDH', mitochondrial_GCS='GCS exploratoria')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    for ax, arm, title in zip(axes, ['closed','PHGDH'], ['A. Fechamento da exportacao hepatica', 'B. Bloqueio de PHGDH hepatica']):
        part = effects[effects.intervention.eq(arm)].copy()
        pairs = list(part[['tumor_variant','capacity_fraction']].drop_duplicates().itertuples(index=False, name=None))
        for offset, schedule, color, label in [(-.17,'stepwise','#16838a','Selecao por intervalo'),
                (.17,'constant_feasible','#b8733a','Protocolo constante viavel')]:
            indexed = part[part.schedule.eq(schedule)].set_index(['tumor_variant','capacity_fraction'])
            values = [indexed.loc[p, 'task_loss_h'] for p in pairs]
            ax.bar(np.arange(len(pairs))+offset, values, width=.32, color=color, label=label)
        ax.set(title=title, ylabel='Reducao nas tarefas (h equivalentes)', ylim=(-.12,4.1))
        ax.set_xticks(range(len(pairs)), [labels[v]+'\n'+str(c)+'x' for v,c in pairs], rotation=38, ha='right', fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.axhline(0,color='#777777',lw=.7)
        if arm == 'PHGDH':
            ax.text(.5, .52, 'Sem reducao detectavel nas tarefas\nA exportacao hepatica foi preservada',
                    transform=ax.transAxes, ha='center', va='center', color='#444444', fontsize=10)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(HERE/'interventions_and_schedules.png', dpi=190)
    fig.savefig(HERE/'interventions_and_schedules.pdf')
    plt.close(fig)
    assert all(sha(ROOT/p) == h for p,h in protected.items())
    assert (new[['hepatic_maintenance','tumor_maintenance']] >= 1-1e-7).all().all()
    assert new.filter(regex='^max_').max().max() < 1e-7
    assert effects[effects.intervention.eq('PHGDH')].task_loss_h.abs().max() < 1e-5
    assert effects[effects.intervention.eq('PHGDH')].intervention_export_pmol.min() > .2879
    assert effects[effects.intervention.eq('PHGDH_SHMT')].intervention_export_pmol.abs().max() < 1e-7
    verification = dict(new_constant_cases=len(new), replayed_intervals=len(new)*120,
        max_inventory_error=float(new.max_inventory_error.max()), max_CN_error=float(new.max_CN_error.max()),
        max_pool_error=float(new.max_pool_error.max()), all_maintenance_satisfied=True,
        protected_inputs_unchanged=True, independent_dataset_reserved=False, all_passed=True)
    (HERE/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
