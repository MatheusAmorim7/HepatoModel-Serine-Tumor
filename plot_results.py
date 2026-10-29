"""Plot freshly recomputed poster cases; reference CSVs are not plot inputs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS=['#A34867','#168477','#77828E']

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.set_ylabel('Minimum hepatic serine (pmol over 12 h)')

def save(fig,out,name):
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def make_figures(data,out):
    fig,ax=plt.subplots(figsize=(9,5.4))
    compositions=['S0.5_P1_C1','S1_P1_C1','S1_P1_C2']
    labels=['Lower material Ser\n0.5x Ser','Reference\n1x Ser/IMP/C1','Higher additional C1\n2x C1']
    conditions=['no_HX','HX_HPRT','HX_HPRT_blocked']
    for j,(condition,label) in enumerate(zip(conditions,['No HX','HX + HPRT active','HX + HPRT blocked'])):
        values=[]
        for c in compositions:
            row=data[(data.stage=='demand')&(data.composition==c)&
                     (data.energy_mode=='fixed_ATP')&(data.salvage_condition==condition)&
                     (data.channel=='serine_only')&data.target.eq(.55)].iloc[0]
            assert row.status=='optimal'
            values.append(max(0.,row.hepatic_Ser_export_pmol))
        bars=ax.bar(np.arange(3)+(j-1)*.24,values,.24,label=label,color=COLORS[j])
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3)
    ax.set_xticks(np.arange(3),labels)
    ax.set_ylim(0,.29)
    ax.legend(frameon=False,ncol=3,fontsize=9)
    ax.set_title('Figure 2: source requirement depends on demand')
    clean(ax)
    save(fig,out,'figure2_demand')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    part=data[(data.stage=='supply')&(data.channel=='serine_only')&data.target.eq(.55)&data.external_Ser_pmol.eq(.5)]
    for cap,label,color in [(0.,'HPRT blocked',COLORS[0]),(.14,'HPRT active',COLORS[1])]:
        selected=part[part.HPRT_cap_pmol_h.eq(cap)&part.panel.eq('supply_grid')].copy()
        if cap==.14:
            selected=__import__('pandas').concat([selected,part[part.panel.eq('analytical_boundary')&
                part.boundary_axis.eq('HX_stock')&part.boundary_multiplier.eq(1.)]])
        selected=selected.sort_values('HX_pmol')
        axes[0].plot(selected.HX_pmol*50,selected.hepatic_Ser_export_pmol.clip(lower=0),
                     marker='o',label=label,color=color)
    axes[0].set_xlabel('Initial hypoxanthine (uM)')
    axes[0].set_title('A  External hypoxanthine')
    axes[0].legend(frameon=False)
    selected=part[part.HX_pmol.eq(.2)&part.panel.eq('supply_grid')].sort_values('HPRT_cap_pmol_h')
    values=selected.hepatic_Ser_export_pmol.clip(lower=0)
    bars=axes[1].bar(np.arange(len(selected)),values,color=COLORS[1])
    axes[1].bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3)
    axes[1].set_xticks(np.arange(len(selected)),[f'{v:g}' for v in selected.HPRT_cap_pmol_h])
    axes[1].set_xlabel('Tumour HPRT flux limit (pmol/h)')
    axes[1].set_title('B  HPRT capacity, HX = 10 uM')
    for ax in axes:
        clean(ax)
        ax.set_ylim(0,.18)
    save(fig,out,'figure3_supply_HPRT')
