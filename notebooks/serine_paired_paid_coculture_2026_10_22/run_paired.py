"""Pair hepatic source interventions with a resource-paid tumor task module.

Fixed populations, well-mixed finite stocks. No growth or signaling inference.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys
import time
import cobra
import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'notebooks/serine_funded_coculture_2026_10_21'
ENERGY=ROOT/'notebooks/serine_resource_paid_energy_2026_10_22'
sys.path.insert(0,str(OLD))
import run_joint as previous
sys.path.insert(0,str(ENERGY))
from paid_energy import PaidSelector, build, BACKGROUND, ROLES, MAINTENANCE, WORK_ATP, SER_TASK, C1_TASK

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def plan():
    cases=[]
    for variant,gly,arm,policy in itertools.product(['native_C1','curated_transport'],
            ['reference','no_initial_glycine'],['closed','funded','PHGDH','PHGDH_SHMT'],
            ['glucose_sparing','oxygen_sparing']):
        cases.append(dict(case=f'{variant}_{gly}_{arm}_{policy}',variant=variant,
            medium=gly,arm=arm,policy=policy,dt=.1,horizon=12.,yield_ATP=24.,control='none'))
    for policy in ['glucose_sparing','oxygen_sparing']:
        base=dict(variant='curated_transport',medium='reference',arm='closed',policy=policy,
                  dt=.1,horizon=12.,yield_ATP=24.)
        cases.append(base|dict(case='rescue_'+policy,control='serine_rescue'))
        cases.append(base|dict(case='no_resources_'+policy,control='no_resources',horizon=1.))
        cases.append(base|dict(case='energy_closed_'+policy,control='energy_routes_closed',horizon=1.))
    for arm in ['closed','funded','PHGDH','PHGDH_SHMT']:
        cases.append(dict(case='refinement_'+arm,variant='curated_transport',medium='reference',
            arm=arm,policy='glucose_sparing',dt=.05,horizon=12.,yield_ATP=24.,control='refinement'))
    return cases

class Joint:
    def __init__(self,spec,fields,template,q):
        self.fields,self.spec=fields,spec
        paid=PaidSelector(template,dict(ATP_per_glucose_oxidative=spec['yield_ATP'],
                          policy=spec['policy'],energy_routes_closed=spec['control']=='energy_routes_closed'),q)
        self.model=m=paid.model
        # Keep the paid tumor's existing task and maintenance constraints intact.
        self.ta,self.theta=paid.alpha,paid.theta
        self.groups={'tumor':m.copy()}
        self.r={'tumor':{r.id:r for r in m.reactions}}
        self.factor={'tumor':1.}
        for role,rid in ROLES.items():
            r=m.reactions.get_by_id(rid)
            r.bounds=(-paid.caps[role],0 if role in ['glucose','oxygen'] else 140.)
        tag='map_02' if spec['variant']=='native_C1' else 'map_14'
        for group in ['basal','local']:
            hep=cobra.io.read_sbml_model(str(previous.hepatic.MAP/(tag+'_'+group+'.xml')))
            hep.solver='glpk'
            hep.reactions.get_by_id(previous.hepatic.ATP).bounds=(0,5)
            hep.reactions.get_by_id(previous.hepatic.SER).bounds=(0,.2 if group=='local' and spec['arm']!='closed' else 0)
            if group=='local':
                for rid in previous.KOS.get(spec['arm'],[]):
                    hep.reactions.get_by_id(rid).bounds=(0,0)
            self.groups[group]=hep.copy()
            ids=hep.reactions.list_attr('id')
            for met in list(hep.metabolites):
                met.id=group+'__'+met.id
            for r in list(hep.reactions):
                r.id=group+'__'+r.id
            hep.repair()
            m.add_reactions([r.copy() for r in hep.reactions])
            self.r[group]={rid:m.reactions.get_by_id(group+'__'+rid) for rid in ids}
            self.factor[group]=previous.COUNTS[group]*previous.FACTOR
        p=m.problem
        self.ha=p.Variable('HEP_MAINTENANCE',lb=0,ub=1)
        m.add_cons_vars(self.ha)
        for g in ['basal','local']:
            m.add_cons_vars(p.Constraint(self.r[g][previous.hepatic.ATP].flux_expression-5*self.ha,
                                        lb=0,ub=0,name='hep_maintenance_'+g))
        self.export=self.factor['local']*self.r['local'][previous.hepatic.SER].flux_expression
        self.gross,self.net,self.budget={},{},{}
        for f in fields.itertuples():
            gross,net=0,0
            for g in self.groups:
                rid=ROLES.get(f.substrate) if g=='tumor' else f.cobra_exchange
                if rid in self.r[g]:
                    r=self.r[g][rid]
                    gross+=self.factor[g]*r.reverse_variable
                    net+=self.factor[g]*r.flux_expression
            self.gross[f.substrate],self.net[f.substrate]=gross,net
            expr=gross-self.export if f.substrate=='serine' else gross
            self.budget[f.substrate]=p.Constraint(expr,ub=0,name='shared_pool_'+f.substrate)
            m.add_cons_vars(self.budget[f.substrate])
        self.faces={}
        for key,expr in dict(hep=self.ha,tumor_maintenance=self.ta,export=self.export,
                            tasks=self.theta,oxygen=self.gross['oxygen'],glucose=self.gross['glucose']).items():
            self.faces[key]=p.Constraint(expr,name='paired_preserve_'+key)
            m.add_cons_vars(self.faces[key])
        self.ids={g:list(self.r[g]) for g in self.groups}
        self.S={g:create_stoichiometric_matrix(original) for g,original in self.groups.items()}
        self.l1={var:self.factor[g] for g in self.r for r in self.r[g].values()
                 for var in [r.forward_variable,r.reverse_variable]}
        self.calls=0

    def solve(self,stock,dt):
        m,p=self.model,self.model.problem
        for face in self.faces.values():
            face.lb=face.ub=None
        for f,amount in zip(self.fields.itertuples(),stock):
            self.budget[f.substrate].ub=max(0,float(amount))/dt
        def opt(expr,direction):
            m.objective=p.Objective(expr,direction=direction)
            self.calls+=1
            return float(m.slim_optimize(error_value=None))
        for key,expr in [('hep',self.ha),('tumor_maintenance',self.ta),('export',self.export),('tasks',self.theta)]:
            value=opt(expr,'max')
            self.faces[key].lb=self.faces[key].ub=value
        order=['glucose','oxygen'] if self.spec['policy']=='glucose_sparing' else ['oxygen','glucose']
        for key in order:
            value=max(0.,opt(self.gross[key],'min'))
            self.faces[key].lb=max(0,value-1e-10)
            self.faces[key].ub=value+1e-10
        m.objective=p.Objective(0,direction='min')
        m.objective.set_linear_coefficients(self.l1)
        m.slim_optimize(error_value=None)
        self.calls+=1
        vectors={g:np.array([self.r[g][rid].flux for rid in self.ids[g]]) for g in self.r}
        residual=max(float(abs(self.S[g]@v).max()) for g,v in vectors.items())
        violation=max(max(self.r[g][rid].lower_bound-v,self.r[g][rid].upper_bound*-1+v,0)
                      for g in self.r for rid,v in zip(self.ids[g],vectors[g]))
        hnet,tnet,gross=[np.zeros(len(self.fields)) for _ in range(3)]
        for j,f in enumerate(self.fields.itertuples()):
            for g in self.r:
                rid=ROLES.get(f.substrate) if g=='tumor' else f.cobra_exchange
                if rid in self.r[g]:
                    v=self.r[g][rid].flux*self.factor[g]
                    (tnet if g=='tumor' else hnet)[j]+=v
                    gross[j]+=max(0,-v)
        export=self.factor['local']*self.r['local'][previous.hepatic.SER].flux
        limits=stock/dt
        limits[self.fields.index[self.fields.substrate.eq('serine')][0]]+=export
        pool_error=float(np.maximum(0,gross-limits).max())
        atp=2*self.r['tumor']['PAID_GLYCOLYSIS'].flux+self.spec['yield_ATP']*self.r['tumor']['PAID_OXIDATION'].flux
        atp_cost=sum(-r.metabolites.get(self.model.metabolites.HC00012_c,0)*r.flux
                     for rid,r in self.r['tumor'].items() if rid not in ['PAID_GLYCOLYSIS','PAID_OXIDATION'])
        assert max(residual,violation,pool_error,abs(atp-atp_cost))<1e-7
        assert all(abs(self.r['tumor'][rid].flux)<1e-10 for rid in BACKGROUND)
        return dict(hep=float(self.ha.primal),maintenance=float(self.ta.primal),theta=float(self.theta.primal),
                    export=export,hnet=hnet,tnet=tnet,vectors=vectors,residual=residual,
                    bound_error=violation,pool_error=pool_error,atp=atp,atp_error=abs(atp-atp_cost))

def simulate(spec,fields,template,q,selector_class=Joint):
    out=HERE/'outputs'/spec['case']
    out.mkdir(parents=True,exist_ok=True)
    joint=selector_class(spec,fields,template,q)
    stock=fields.initial_pmol.to_numpy().copy()
    idx={n:i for i,n in enumerate(fields.substrate)}
    if spec['medium']=='no_initial_glycine':
        stock[idx['glycine']]=0
    if spec['control']=='serine_rescue':
        stock[idx['serine']]=2.  # Declared 100 uM instead of 25 uM in 20 nL.
    if spec['control']=='no_resources':
        stock[:]=0
    initial=stock.copy()
    hcum,tcum,external,correction=[np.zeros(len(fields)) for _ in range(4)]
    donor=0 if spec['control']=='no_resources' else 36.
    totals=dict(hep_h=0.,maintenance_h=0.,task_h=0.,export_pmol=0.,ATP_paid_pmol=0.)
    trace,inventory=[],[]
    eligible=True
    eligible_h=eligible_tasks=0.
    t0=time.perf_counter()
    with gzip.open(out/'fluxes.csv.gz','wt') as stream:
        names=[g+'__'+rid for g in joint.ids for rid in joint.ids[g]]
        stream.write('state_id,'+','.join(names)+'\n')
        for step in range(round(spec['horizon']/spec['dt'])):
            value=joint.solve(stock,spec['dt'])
            stream.write(str(step)+','+','.join(str(float(v)) for g in joint.ids for v in value['vectors'][g])+'\n')
            delta=spec['dt']*(value['hnet']+value['tnet'])
            raw=stock+delta
            assert raw.min()>-1e-7
            fix=np.maximum(0,-raw)
            stock=raw+fix
            correction+=fix
            hcum+=spec['dt']*value['hnet']; tcum+=spec['dt']*value['tnet']
            feed=min(max(0,4-stock[idx['oxygen']]),3*spec['dt'],donor)
            stock[idx['oxygen']]+=feed; external[idx['oxygen']]+=feed; donor-=feed
            assert stock[idx['oxygen']]<=4+1e-7
            totals['hep_h']+=value['hep']*spec['dt']
            totals['maintenance_h']+=value['maintenance']*spec['dt']
            totals['task_h']+=value['theta']*spec['dt']
            totals['export_pmol']+=value['export']*spec['dt']
            totals['ATP_paid_pmol']+=value['atp']*spec['dt']
            eligible=eligible and min(value['hep'],value['maintenance'])>=1-1e-7
            if eligible:
                eligible_h+=spec['dt']; eligible_tasks+=value['theta']*spec['dt']
            balance=stock-initial-external-hcum-tcum-correction
            atoms=(stock-initial-external-correction)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
            atoms+=np.array([3*SER_TASK*totals['task_h']+C1_TASK*totals['task_h'],SER_TASK*totals['task_h']])
            assert max(abs(balance).max(),abs(atoms).max())<1e-7
            trace.append(dict(step=step,time_h=(step+1)*spec['dt'],hep=value['hep'],
                maintenance=value['maintenance'],theta=value['theta'],export_pmol_h=value['export'],
                ATP_paid_pmol_h=value['atp'],max_Sv=value['residual'],bound_error=value['bound_error'],
                pool_error=value['pool_error'],ATP_error=value['atp_error'],
                carbon_residue=atoms[0],nitrogen_residue=atoms[1],eligible=eligible))
            for j,f in enumerate(fields.itertuples()):
                inventory.append(dict(step=step,time_h=(step+1)*spec['dt'],substrate=f.substrate,
                    initial=initial[j],remaining=stock[j],external=external[j],hepatic=hcum[j],
                    tumor=tcum[j],correction=correction[j],residue=balance[j]))
    pd.DataFrame(trace).to_csv(out/'trajectory.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)
    pd.DataFrame([dict(group=g,reaction=rid,lb=r.lower_bound,ub=r.upper_bound,equation=r.reaction)
                  for g in joint.r for rid,r in joint.r[g].items()]).to_csv(out/'reaction_contract.csv',index=False)
    result=spec|totals|dict(eligible_h=eligible_h,eligible_task_h=eligible_tasks,
        oxygen_external_pmol=external[idx['oxygen']],seconds=time.perf_counter()-t0,
        calls=joint.calls,serine_initial_pmol=initial[idx['serine']])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(spec['case'],'tasks',round(result['task_h'],6),'export',round(result['export_pmol'],6),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver='glpk'
    specs=plan()
    if args.smoke:
        specs=[s|dict(case='smoke_'+s['arm'],horizon=.2) for s in specs if s['variant']=='curated_transport'
            and s['medium']=='reference' and s['policy']=='glucose_sparing' and s['control']=='none']
    assert len({s['case'] for s in specs})==len(specs)
    protected=[ROOT/'config/PhysiCell_settings.xml',ROOT/'custom_modules/custom.cpp',previous.NATIVE,
               ENERGY/'paid_energy.py',OLD/'run_joint.py',previous.PARKER]
    protected += [previous.hepatic.MAP/(tag+'_'+g+'.xml') for tag in ['map_02','map_14'] for g in ['basal','local']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    suffix='_smoke' if args.smoke else ''
    fields=previous.hepatic.fields()
    fields.to_csv(HERE/'fields.csv',index=False)
    protocol=dict(cases=specs,counts=previous.COUNTS,volume_nL=20,horizon_h=12,
        units='pmol and hours; normalized hepatic flux factor inherited',
        order='hepatic maintenance, tumor maintenance, hepatic export, tumor tasks, resource policy, L1',
        oxygen='finite 36 pmol donor, replacement toward 4 pmol after step, max 3 pmol/h',
        energy='tumor ATP from glucose only; hypothetical oxidative yield 24; all four formal energy/redox sources closed',
        redox_limit='No paid tumor Gly/formate rescue implemented. Gly alternatives tested in hepatocytes only.',
        scope='Fixed populations, no spatial agents, no growth, no induction kinetics, no fitted rates',
        serine_cap='12 possible hepatic producers; fixed exploratory export ceiling .024 pmol/h',
        no_external_serine_uptake_in_hepatocytes=True,inputs_sha256=hashes)
    (HERE/('protocol'+suffix+'.json')).write_text(json.dumps(protocol,indent=2))
    q=json.loads(previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    template=build(cobra.io.read_sbml_model(str(previous.NATIVE)))
    rows=[]
    for spec in specs:
        saved=HERE/'outputs'/spec['case']/'summary.json'
        if args.resume and saved.exists():
            result=json.loads(saved.read_text())
            assert all(result.get(k)==v for k,v in spec.items())
        else:
            result=simulate(spec,fields,template,q)
        rows.append(result)
        pd.DataFrame(rows).to_csv(HERE/('summary'+suffix+'.csv'),index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    (HERE/('execution'+suffix+'.json')).write_text(json.dumps(dict(completed=len(rows),inputs_unchanged=True),indent=2))

if __name__=='__main__':
    main()
