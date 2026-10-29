"""Paired finite-pool supply/HPRT sensitivity; no growth or parameter fit."""
import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'notebooks/serine_demand_composition_2026_10_26'))
import run_sensitivity as demand

HOURS, TOL = demand.HOURS, demand.TOL
SER_STOCKS = [.1, .25, .5, .75, 1.]
HX_STOCKS = [0., .05, .1, .2, .4]
HPRT_CAPS = [0., .001, .004, .008, .14]
TARGETS = [.25, .55, .85]
COMPOSITION = dict(composition='S1_P1_C1', energy_mode='fixed_ATP',
    ser_factor=1., imp_factor=1., c1_factor=1.,
    ser_rate=demand.SER, imp_rate=demand.IMP, c1_rate=demand.C1,
    work_rate=demand.WORK)


def plan():
    rows = []
    for ser, hx, cap in itertools.product(SER_STOCKS, HX_STOCKS, HPRT_CAPS):
        for channel in ['closed', 'serine_only']:
            for target in [None]+TARGETS:
                rows.append(COMPOSITION | dict(panel='supply_grid', external_Ser_pmol=ser,
                    Ser_uM=ser*50., HX_pmol=hx, HX_uM=hx*50., HPRT_cap_pmol_h=cap,
                    channel=channel, target=target))
    # Boundary checks derive from declared tasks/stocks before any LP is run.
    critical = (HOURS*(demand.SER+demand.C1+2*demand.IMP)*.55-.5)/2
    for axis in ['HX_stock', 'HPRT_capacity']:
        for multiplier in [.99, 1., 1.01]:
            hx = critical*multiplier if axis == 'HX_stock' else .2
            cap = critical/HOURS*multiplier if axis == 'HPRT_capacity' else .14
            for channel in ['closed', 'serine_only']:
                rows.append(COMPOSITION | dict(panel='analytical_boundary', boundary_axis=axis,
                    boundary_multiplier=multiplier, external_Ser_pmol=.5, Ser_uM=25.,
                    HX_pmol=hx, HX_uM=hx*50., HPRT_cap_pmol_h=cap,
                    channel=channel, target=.55))
    assert len(rows) == 1012
    return [row | dict(case=f'supply_{i:04d}') for i, row in enumerate(rows)]


def prepare(row, cache, fields):
    # Clone a permissive HX template, then change only declared stocks/capacity.
    joint, initial, idx, _ = demand.clone(cache[('HX_HPRT',row['channel'])],fields)
    assert abs(joint.factor['tumor']-1.) < TOL
    for name, stock in [('serine',row['external_Ser_pmol']),('hypoxanthine',row['HX_pmol'])]:
        initial[idx[name]] = stock
        joint.budget[name].ub = stock/HOURS
    joint.r['tumor']['r0275'].bounds = (-row['HPRT_cap_pmol_h'],0.)
    demand.configure(joint,row)
    return joint, initial, idx


def validate(joint,row,fields,initial,idx,theta,flux):
    metrics, inventory = demand.validate(joint,row,fields,initial,idx,theta,flux)
    salvage = -flux['tumor']['r0275']
    hx_entry = -flux['tumor'][demand.prior.community.ROLES['hypoxanthine']]
    assert min(salvage,hx_entry) >= -TOL
    assert salvage <= row['HPRT_cap_pmol_h']+TOL
    assert abs(salvage-hx_entry) < TOL
    assert HOURS*hx_entry <= row['HX_pmol']+TOL
    metrics.update(HPRT_salvage_pmol=HOURS*salvage, HX_entry_pmol=HOURS*hx_entry,
        HPRT_capacity_utilisation=(salvage/row['HPRT_cap_pmol_h'] if row['HPRT_cap_pmol_h'] else 0.))
    return metrics, inventory


def protected():
    hashes = demand.protected()
    for folder in [demand.HERE, ROOT/'notebooks/serine_purine_material_2026_10_25']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.md','.csv','.json']:
                hashes[p.relative_to(ROOT).as_posix()] = demand.sha(p)
    active = ROOT/'config/hepatonet_pan_zonal_official_fastcore_reduced.xml'
    hashes[active.relative_to(ROOT).as_posix()] = demand.sha(active)
    return hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    out=HERE/('smoke' if args.smoke else 'outputs')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'results.csv').exists(), 'Previous results are preserved; use a new stage'
    cases=plan()
    if args.smoke:
        cases=[r for r in cases if (r['panel']=='supply_grid' and r['external_Ser_pmol']==.5 and
            r['HX_pmol'] in [0.,.2] and r['HPRT_cap_pmol_h'] in [0.,.004,.14] and
            r['target'] in [None,.55]) or r['panel']=='analytical_boundary']
        assert len(cases)==36
    frozen=protected()
    protocol=dict(cases=cases, declared_before_execution=True,script_sha256=demand.sha(__file__),
        protected=frozen,volume_nL=20.,duration_h=HOURS,
        populations=dict(basal=252,local=12,tumour=7),
        composition_rates_pmol_h=dict(Ser=demand.SER,IMP=demand.IMP,C1=demand.C1,
            task_ATP=demand.WORK,maintenance_ATP=demand.MAINT),
        external_Ser_pmol=SER_STOCKS, HX_pmol=HX_STOCKS,HPRT_cap_pmol_h=HPRT_CAPS,
        targets=TARGETS, capacities='Task maximum capped at theta=1, plus paired fixed targets',
        grid_scope='Exploratory ranges, not inferred human/patient distributions or confidence intervals',
        HPRT='Aggregate tumour reaction ceiling, not measured protein expression, Km, Vmax or inhibitor dose',
        boundary='0.0734pmol HX and 0.0061166667pmol/h HPRT follow from the declared 55% reference task and 0.5pmol Ser; checked at -1%, exact and +1%',
        fixed_background='Reference composition, external ribose access, tumour Ser assembly/GCS closed, local formate export closed, hepatic PHGDH available, diagnostic DMG0.144pmol',
        optimisation='Maximum theta or fixed theta, then minimal hepatic Ser, gross glucose, oxygen and weighted L1. Full fixed maintenance.',
        oxygen='Inherited finite 4pmol initial+32pmol reservoir, feed <=3pmol/h',
        no_growth=True,no_fit=True,source_not_measured_induction=True,active_PhysiCell_changed=False,
        negative_results='All infeasible targets retained; no demand, energy or nutrient repair')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs=demand.load_inputs()
    fields=inputs[-1]
    cache=demand.templates(inputs)
    fields.to_csv(out/'fields.csv',index=False)
    _,initial,idx,_=demand.clone(cache[('HX_HPRT','closed')],fields)
    pd.DataFrame([dict(substrate=f.substrate,pmol=initial[idx[f.substrate]],uM=initial[idx[f.substrate]]*50.)
        for f in fields.itertuples()]).to_csv(out/'background_initial_stocks.csv',index=False)
    results=[]
    with gzip.open(out/'fluxes.csv.gz','wt',newline='') as vf, (out/'stocks.csv').open('w',newline='') as sf:
        fw=csv.DictWriter(vf,fieldnames=['case','group','reaction','flux']);fw.writeheader()
        iw=None
        for i,row in enumerate(cases):
            joint,initial,idx=prepare(row,cache,fields)
            theta,flux=demand.solve(joint,row['target'] is not None)
            if flux is None:
                results.append(row|dict(status='infeasible_fixed_target' if row['target'] is not None else 'infeasible_maintenance'))
            else:
                metrics,inventory=validate(joint,row,fields,initial,idx,theta,flux)
                results.append(row|dict(status='optimal')|metrics)
                for g in flux:
                    fw.writerows(dict(case=row['case'],group=g,reaction=rid,flux=v) for rid,v in flux[g].items())
                if iw is None:
                    iw=csv.DictWriter(sf,fieldnames=['case']+list(inventory[0]));iw.writeheader()
                iw.writerows(dict(case=row['case'])|r for r in inventory)
            if (i+1)%20==0 or i==len(cases)-1:
                pd.DataFrame(results).to_csv(out/'results.csv',index=False)
                print(f'{i+1}/{len(cases)} {results[-1]["status"]}',flush=True)
    assert all(demand.sha(ROOT/name)==h for name,h in frozen.items())
    checks=dict(cases=len(results),status_counts=pd.DataFrame(results).status.value_counts().to_dict(),
        stock_and_flux_checks=True,protected_unchanged=True,no_biological_calibration=True)
    (out/'run_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
