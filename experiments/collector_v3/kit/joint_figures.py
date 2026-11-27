"""Scientific figures from audited data only; no invented uncertainty bands."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=dict(joint='#222222',joint_discounted='#0072B2',joint_window='#D55E00',
            matched_recent='#009E73',article_pd='#CC79A7',legacy_lp_mean='#999999')


def read(path):
    return list(csv.DictReader(Path(path).open()))


def render(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    e1=read(root/'E1-audit/rounds.csv')
    fig,ax=plt.subplots(figsize=(6,3.4))
    for mode in ('joint','no_dual','legacy_lp_mean'):
        series=sorted({r['series'] for r in e1})
        x=[int(s[1:])/1e6 for s in series]
        y=[100*np.mean([int(r['accepted']) for r in e1 if r['series']==s and r['policy']==mode]) for s in series]
        ax.plot(x,y,'.-',label=mode)
    ax.set(xlabel='Episode budget (decimal MB)',ylabel='Accepted (%)',title='E1 real distributed data; 2 seed blocks per budget')
    ax.legend();save(fig,'01_E1_budget')
    e5=read(root/'E5-audit/rounds.csv');seeds=sorted({r['seed'] for r in e5})
    fig,ax=plt.subplots(figsize=(7,3.6))
    for mode in COLORS:
        curves=[]
        for seed in seeds:
            seq=sorted([r for r in e5 if r['policy']==mode and r['seed']==seed],key=lambda r:int(r['round']))
            if seq:curves.append(np.cumsum([int(r['accepted']) for r in seq]))
        if curves:
            for y in curves:ax.plot(np.arange(1,121),y,color=COLORS[mode],alpha=.25,lw=.7)
            ax.plot(np.arange(1,121),np.mean(curves,axis=0),label=mode,color=COLORS[mode])
    for x in (40.5,80.5):ax.axvline(x,color='grey',ls='--',lw=.7)
    ax.set(xlabel='Round',ylabel='Cumulative accepted updates',title='E5 real data; thin lines = individual blocks (n=2)')
    ax.legend(ncol=2,fontsize=7);save(fig,'02_E5_recovery')
    shadow=read(root/'E5-shadow/shadow_phases.csv')
    methods=['joint','discount_0.9','discount_0.95','discount_0.98','window_16','window_32','window_64']
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
    for phase,ax in zip((1,2),axes):
        y=[np.mean([float(r['path_brier']) for r in shadow if int(r['phase'])==phase and r['learner']==m]) for m in methods]
        ax.bar(np.arange(len(methods)),y,color=['#222222']+['#0072B2']*3+['#D55E00']*3)
        ax.set_xticks(np.arange(len(methods)),methods,rotation=65,ha='right',fontsize=7)
        ax.set(title=f'Phase {phase+1}',ylabel='Path Brier loss (lower is better)')
    fig.suptitle('Same Joint history/actions; predictive shadows, not policy returns',fontsize=10)
    save(fig,'03_shadow_tradeoff')
    fig,axes=plt.subplots(2,2,figsize=(8,5.2),sharex=True)
    for seed in seeds:
        seq=sorted([r for r in e5 if r['policy']=='joint' and r['seed']==seed],key=lambda r:int(r['round']))
        x=[int(r['round']) for r in seq]
        for ax,key in zip(axes.flat,('pre40_label_fraction','mirror_remaining','mirror_price','feasible_actions')):
            ax.plot(x,[float(r[key]) if r[key] else np.nan for r in seq],label=seed)
            for point in (40.5,80.5):ax.axvline(point,color='grey',ls='--',lw=.6)
            ax.set(ylabel=key)
    for ax in axes[-1]:ax.set_xlabel('Round')
    axes[0,0].legend(fontsize=7);fig.suptitle('E5 original Joint: history, quota and price diagnostics',fontsize=10)
    save(fig,'04_joint_mechanisms')
    local=read(root/'local/validation/summary.csv')
    families=['joint_discounted','joint_window','matched_recent','article_pd']
    cases=[(c,q) for c in ('stationary','abrupt','gradual','shared_host') for q in (.1,.5)]
    vals=np.array([[next(float(r['versus_joint_pp']) for r in local if r['family']==f and r['case']==c and float(r['quota_fraction'])==q) for c,q in cases] for f in families])
    fig,ax=plt.subplots(figsize=(8,3.4))
    im=ax.imshow(vals,cmap='RdBu',vmin=-4,vmax=4,aspect='auto')
    for i in range(len(families)):
        for j in range(len(cases)):ax.text(j,i,f'{vals[i,j]:+.2f}',ha='center',va='center',fontsize=8)
    ax.set_yticks(range(len(families)),families)
    ax.set_xticks(range(len(cases)),[f'{c}\nq={q}' for c,q in cases],fontsize=7)
    ax.set_title('Local synthetic validation: difference vs tuned Joint (pp), 8 paired seeds')
    fig.colorbar(im,ax=ax,label='percentage points');save(fig,'05_local_null_and_drift')
    atlas=read(root/'local/validation/atlas_anchor.csv')
    fig,ax=plt.subplots(figsize=(6,3.5))
    ms=list(COLORS)[:5]
    ax.bar(ms,[100*np.mean([float(r['fraction']) for r in atlas if r['family']==m]) for m in ms],color=[COLORS[m] for m in ms])
    ax.set(ylabel='Accepted coverage proxy (%)',title='Historical RIPE ICMP anchor; not cloud confirmation')
    ax.tick_params(axis='x',rotation=25);save(fig,'06_atlas_limitation')
    (output/'SCOPE.txt').write_text('E1/E5 real observations; shadows use original Joint actions only. Local synthetic/RIPE figures are proxies, not TCP/cloud measurements. No confidence bands from two pilot seeds.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();render(a.root,a.out)
