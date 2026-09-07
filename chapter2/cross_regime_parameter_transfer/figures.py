"""Descriptive figures; historical parsing requires complete fresh results."""
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chapter2.esn_optimisation import load_strict_json
from chapter2.esn_data import file_sha256
from .config import *
from .optimisation import scale_current
from .protection import output_path, write_new


def save(fig, stem, root=None):
    root = RESULTS/'figures' if root is None else Path(root)
    paths = [output_path(root/(stem+s)) for s in ('.png','.pdf')]
    if any(p.exists() for p in paths):
        plt.close(fig)
        raise FileExistsError(stem)
    root.mkdir(parents=True, exist_ok=True)
    try:
        for p in paths: fig.savefig(p,bbox_inches='tight',dpi=120)
    finally: plt.close(fig)
    return [str(p) for p in paths]


def design_figures(root=None):
    paths=[]; fig,ax=plt.subplots(figsize=(8,4))
    for label,cs in (('Old regular optimisation z-score',REGULAR),('Old chaotic optimisation z-score',CHAOTIC)):
        ax.plot(CURRENTS,(np.array(CURRENTS)-np.mean(cs))/np.std(cs),marker='o',label=label)
    ax.plot(CURRENTS,scale_current(CURRENTS),marker='s',label='Fixed physical scaling')
    ax.set(xlabel='Physical I',ylabel='Scaled I'); ax.legend()
    paths+=save(fig,'01_current_scaling',root)
    rows=[f for fs in FOLDS.values() for f in fs]; values=np.zeros((5,5))
    for i,(_,training,held) in enumerate(rows):
        for j,c in enumerate(CURRENTS): values[i,j]=1 if c in training else 2 if c==held else 0
    fig,ax=plt.subplots(figsize=(8,4));ax.imshow(values,vmin=0,vmax=2)
    ax.set(yticks=range(5),yticklabels=[r[0] for r in rows],xticks=range(5),xticklabels=CURRENTS)
    for i in range(5):
        for j in range(5):ax.text(j,i,('forbidden','fit','validate')[int(values[i,j])],ha='center',va='center',color='white')
    paths+=save(fig,'02_parameter_folds',root)
    return paths


def optimisation_figures(histories,root=None):
    fig,ax=plt.subplots(figsize=(8,4))
    for s,h in histories.items():
        y=[t['objective'] for t in h['trials']]
        ax.plot(range(1,len(y)+1),np.minimum.accumulate(y),label=s)
    ax.set(xlabel='Trial',ylabel='Best bounded proxy (not authoritative rank)',yscale='symlog');ax.legend()
    paths=save(fig,'03_search_convergence',root)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(s,h) in zip(axes,histories.items()):
        for i,c in enumerate(h['confirmations']):
            ax.plot([str(x['seed']) for x in c['seed_results']],[x['aggregate']['worst_nrmse'] for x in c['seed_results']],marker='o',label=str(i+1))
        ax.set(title=s,ylabel='Worst held-out-current NRMSE',yscale='log');ax.legend(title='Candidate')
    return paths+save(fig,'04_five_seed_parameter_transfer',root)


def prediction_figures(records,arrays,root=None):
    paths=[]
    for number,code,current in ((5,'RR',1.67),(6,'RC',3.20),(7,'CC',3.34),(8,'CR',1.67)):
        r=next(r for r in records if r['matrix_code']==code and r['seed']==42 and r['current']==current and r['window']=='short_1')
        a=arrays(r);fig,axes=plt.subplots(3,1,figsize=(9,6),sharex=True)
        for i,ax in enumerate(axes):
            ax.plot(a['time'],a['targets'][:,i],color='black',label='Truth')
            ax.plot(a['time'],a['predictions'][:,i],lw=.7,label='Prediction');ax.set_ylabel('xyz'[i])
        axes[0].set_title(f'Fresh {code}, I={current}, seed 42');axes[0].legend();axes[-1].set_xlabel('Time')
        paths+=save(fig,f'{number:02d}_fresh_{code.lower()}',root)
    return paths


def values(records,key):
    result=[]
    for code in ('RR','RC','CC','CR'):
        v=[r['metrics'][key] for r in records if r['family']=='fixed_short' and r['matrix_code']==code and r['metrics'][key] is not None]
        result.append(float(np.mean(v)) if key=='diverged' else statistics.median(v) if v else np.nan)
    return result


def metric_figures(records,root=None):
    paths=[]
    for n,key in ((9,'nrmse_state'),(10,'valid_prediction_time'),(11,'diverged')):
        fig,ax=plt.subplots();ax.bar(('RR','RC','CC','CR'),values(records,key))
        ax.set_ylabel('Divergence rate' if key=='diverged' else 'Median '+key)
        paths+=save(fig,f'{n:02d}_{key}',root)
    return paths


def phase_figure(records,arrays,root=None):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,code in zip(axes,('RC','CR')):
        r=next(r for r in records if r['matrix_code']==code and r['seed']==42 and r['window']=='short_1');a=arrays(r)
        for k in ('targets','predictions'):ax.plot(a[k][:,0],a[k][:,1],lw=.5,label=k)
        ax.set(title=code,xlabel='x',ylabel='y');ax.legend()
    return save(fig,'12_phase_portraits',root)


def comparison_figure(new,old,root=None):
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,('nrmse_state','valid_prediction_time','diverged')):
        for offset,(label,rs) in enumerate((('Historical',old),('Fresh follow-up',new))):
            ax.bar(np.arange(4)+offset*.35,values(rs,key),width=.35,label=label)
        ax.set(xticks=np.arange(4)+.175,xticklabels=('RR','RC','CC','CR'),title=key);ax.legend()
    fig.suptitle('Different trajectories: descriptive comparison, not paired test')
    return save(fig,'13_historical_vs_fresh',root)


def historical_records():
    from .experiment import audit_final
    audit_final()
    return load_strict_json(PROJECT/'chapter2/cross_regime_improvement/results/evaluation_records.json')['records']


def generate_final():
    from .experiment import audit_final
    records=audit_final()['records'];manifest=RESULTS/'figures/manifest.json'
    if manifest.exists():
        saved=load_strict_json(manifest)
        if saved['evaluation_sha256']!=file_sha256(RESULTS/'evaluation.json') or any(file_sha256(p)!=sha for p,sha in saved['figures'].items()):
            raise RuntimeError('figure provenance mismatch')
        return list(saved['figures'])
    def arrays(r):
        with np.load(r['raw_arrays_path'],allow_pickle=False) as a:return dict(a)
    histories={s:load_strict_json(RESULTS/'optimisation'/f'{s}_history.json') for s in SCENARIOS}
    paths=design_figures()+optimisation_figures(histories)+prediction_figures(records,arrays)
    paths+=metric_figures(records)+phase_figure(records,arrays)+comparison_figure(records,historical_records())
    write_new(manifest,{'evaluation_sha256':file_sha256(RESULTS/'evaluation.json'),'figures':{p:file_sha256(p) for p in paths}})
    return paths
