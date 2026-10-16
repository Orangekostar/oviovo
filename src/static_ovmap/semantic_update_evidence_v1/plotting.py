"""Evidence-only A panels; vector exports and inspectable previews."""
from pathlib import Path
import numpy as np
from .common import read,verified

def render(root,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    root=Path(root);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    summary=verified(root/'A/summary.json');cand=read(root/'A/candidates.json.gz')['rows']
    colors={'S_LL':'#0072B2','G_LL':'#E69F00','V_LL':'#CC79A7','MV_ATTN':'#009E73'}
    def save(fig,name):
        for suffix in ('svg','pdf','png'):fig.savefig(dest/(name+'.'+suffix),dpi=160,bbox_inches='tight')
        svg=dest/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    names=['LOFO_SIMPLE','B_FC','S_LL','G_LL','V_LL','MV_ATTN']
    fig,ax=plt.subplots(figsize=(8.2,3.5));x=np.arange(len(names))
    for seed,shift,marker in [(17,-.12,'o'),(29,.12,'s')]:
        values=[]
        for name in names:
            row=next(r for r in summary['main_rows']+summary['LOFO_rows'] if r['method']==name and r['seed']==(None if name=='B_FC' else seed))
            values.append(row['metrics']['H_macro']*100)
        use=np.arange(len(names))!=1
        ax.scatter((x+shift)[use],np.asarray(values)[use],label=f'Seed {seed}',marker=marker,s=45)
        for xi,v in zip((x+shift)[use],np.asarray(values)[use]):
            dy=7 if seed==17 or v<3 else -14
            ax.annotate(f'{v:.2f}',(xi,v),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=7)
    fc=next(r for r in summary['main_rows'] if r['method']=='B_FC')['metrics']['H_macro']*100
    ax.scatter([1],[fc],marker='D',s=45,color='#666666',label='Shared static FC')
    ax.annotate(f'{fc:.2f}',(1,fc),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.set(xticks=x,xticklabels=['LOFO simple','Frozen FC','S','G','V','MV'],ylabel='Class-macro H (%)',ylim=(-1,100),title='Exposed CONF12 · clean 8V · fixed matched1000 (FC static)')
    ax.legend(loc='upper right');ax.grid(axis='y',alpha=.15);save(fig,'A_main')
    fig,axes=plt.subplots(1,2,figsize=(9,4.3),sharex=True,sharey=True)
    for ax,seed in zip(axes,(17,29)):
        for kind,marker,color in [('MIX','o','#0072B2'),('CAP','x','#D55E00')]:
            rows=[r for r in cand if r['seed']==seed and r['recipe']['kind']==kind]
            ax.scatter([r['metrics']['B_macro']*100 for r in rows],[r['metrics']['N_macro']*100 for r in rows],
                       label=kind+' + gamma (55)',marker=marker,s=20,alpha=.65,color=color)
        for (method,color),marker in zip(colors.items(),('s','D','P','^')):
            r=next(r for r in summary['main_rows'] if r['method']==method and r['seed']==seed)
            ax.scatter(r['metrics']['B_macro']*100,r['metrics']['N_macro']*100,color=color,marker=marker,s=45,edgecolor='#222222',linewidth=.5,label=method.replace('_LL','').replace('_ATTN',''))
        ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Base class-macro accuracy (%)',title=f'Seed {seed} · all 110 recipes');ax.grid(alpha=.15)
    axes[0].set_ylabel('Novel class-macro accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,fontsize=7)
    fig.subplots_adjust(bottom=.23)
    save(fig,'A_tradeoffs')
    angles=read(root/'A/angles.json.gz')['rows'];fig,axes=plt.subplots(2,3,figsize=(10,5.5),sharey=True)
    for si,seed in enumerate((17,29)):
        for gi,group in enumerate(('all','base','novel')):
            ax=axes[si,gi];data=[];labels=[]
            for method in colors:
                a=[r['degrees'] for r in angles if r['seed']==seed and r['method']==method and r['degrees'] is not None and (group=='all' or r['base']==(group=='base'))]
                data.append(a);labels.append(method.replace('_LL','').replace('_ATTN','')+'\nn='+str(len(a)))
            box=ax.boxplot(data,showfliers=False,patch_artist=True)
            ax.set_xticks(np.arange(1,len(labels)+1),labels)
            for patch,color in zip(box['boxes'],colors.values()):patch.set_facecolor(color);patch.set_alpha(.55)
            ax.set(title=f'{group} · seed {seed}',ylim=(0,180));ax.grid(axis='y',alpha=.15)
            if gi==0:ax.set_ylabel('Angle from FC (degrees)')
    fig.tight_layout();save(fig,'A_angles')
    fig,axes=plt.subplots(2,3,figsize=(10,5.5))
    for si,seed in enumerate((17,29)):
        for gi,group in enumerate(('all','base','novel')):
            ax=axes[si,gi];rows=[next(r for r in summary['fix_harm'] if r['method']==m and r['seed']==seed and r['group']==group) for m in colors]
            ax.bar(np.arange(4)-.17,[r['corrected'] for r in rows],width=.34,color='#0072B2',label='Corrected')
            ax.bar(np.arange(4)+.17,[r['harmed'] for r in rows],width=.34,color='#D55E00',hatch='//',label='Harmed')
            ax.set(xticks=np.arange(4),xticklabels=['S','G','V','MV'],title=f'{group} · seed {seed} · n={rows[0]["originals"]}',ylabel='Original objects')
            ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'A_fix_harm')
