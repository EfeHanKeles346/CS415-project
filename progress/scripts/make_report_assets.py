"""Build report tables and vector figures from saved experimental evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; F=ROOT/'figures'
F.mkdir(exist_ok=True)
def read(name): return json.loads((R/name).read_text())
def write(name,text): (R/name).write_text(text+'\n')
a=read('audit.json'); s=read('baseline_summary.json'); c=read('experiment_config.json')
m=s['Majority']; u=s['TFIDF-unweighted']; b=s['TFIDF-balanced']
val=lambda model,key:model[key]['mean']
fmt=lambda n:f'{n:,}'
numbers=dict(UnweightedAcc=f"{100*val(u,'accuracy'):.2f}", UnweightedMAE=f"{val(u,'mae'):.3f}",
    UnweightedFone=f"{val(u,'macro_f1'):.3f}",BalancedFone=f"{val(b,'macro_f1'):.3f}",
    MajorityAcc=f"{100*val(m,'accuracy'):.2f}",MajorityFone=f"{val(m,'macro_f1'):.3f}",
    RemovedPct=f"{100*(a['raw_rows']-a['clean_rows'])/a['raw_rows']:.2f}",
    FiveStarPct=f"{100*a['clean_rating_counts']['5']/a['clean_rows']:.2f}",
    TwoStarPct=f"{100*a['clean_rating_counts']['2']/a['clean_rows']:.2f}",
    LongPct=f"{100*a['fraction_over_256_words']:.2f}",CPUName=c['cpu'],PythonVersion=c['python'],
    SklearnVersion=c['sklearn'],VectorSeconds=f"{c['vectorization_seconds']:.1f}")
write('report_numbers.tex','\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items()))

def table(caption,label,cols,header,rows,wide=False,foot=''):
    env='table*' if wide else 'table'
    return '\n'.join([f'\\begin{{{env}}}[t]',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
        r'\centering\footnotesize',f'\\begin{{tabular}}{{@{{}}{cols}@{{}}}}',r'\toprule',
        header+r'\\\midrule',*[' & '.join(row)+r'\\' for row in rows],r'\bottomrule',
        r'\end{tabular}',foot,f'\\end{{{env}}}'])

write('audit_table.tex',table('Measured data-curation counts on the full archive.','tab:audit','lr',
    'Item & Count',[[name,fmt(n)] for name,n in [
    ('Raw reviews',a['raw_rows']),('Raw products',a['raw_products']),
    ('Invalid or missing-field rows',a['invalid_rows']),
    ('Conflicting normalized-text groups',a['conflicting_text_groups']),
    ('Rows in conflicting groups removed',a['conflicting_rows_removed']),
    ('Same-label duplicate rows removed',a['same_label_duplicate_rows_removed']),
    ('Retained reviews',a['clean_rows']),('Retained products',a['clean_products'])]]))

splits=['train','validation','test']
rows=[]
for star in range(1,6):
    rows.append([str(star)]+[fmt(a['splits'][x]['rating_counts'][str(star)]) for x in splits])
rows.extend([['Total']+[fmt(a['splits'][x]['rows']) for x in splits],
             ['Products']+[fmt(a['splits'][x]['products']) for x in splits]])
write('split_table.tex',table('Retained review counts by star rating and partition. Test data are reserved for final prediction evaluation.',
    'tab:splits','lrrr','Stars & Train & Validation & Test',rows))

rows=[]
for name,model in [('Majority',m),('TF--IDF, unweighted',u),('TF--IDF, balanced',b)]:
    row=[name]
    for key,scale,dec in [('accuracy',100,2),('macro_f1',1,4),('mae',1,4),('large_error_rate',100,2),('fit_seconds',1,2)]:
        mean=model[key]['mean']*scale; sd=model[key]['sd']*scale
        row.append((f'{mean:.{dec}f}' if name=='Majority' else f'${mean:.{dec}f} \\pm {sd:.{dec}f}$') if not(key=='fit_seconds' and name=='Majority') else '--')
    rows.append(row)
write('baseline_table.tex',table('Validation performance: mean $\\pm$ sample standard deviation over seeds 17, 29, and 43. Accuracy and LER are percentages; macro-F1 uses the 0--1 scale; MAE is in stars. The deterministic majority baseline is evaluated once.',
    'tab:baselines','lccccc','Model & Accuracy $\\uparrow$ & Macro-F1 $\\uparrow$ & MAE $\\downarrow$ & LER $\\downarrow$ & Fit time (s)',rows,wide=True))

mae_reduction=100*(val(u,'mae')-val(b,'mae'))/val(u,'mae')
ler_reduction=100*(val(u,'large_error_rate')-val(b,'large_error_rate'))/val(u,'large_error_rate')
write('results_discussion.tex',
    f"The unweighted lexical classifier improves accuracy to {100*val(u,'accuracy'):.2f}\\%, macro-F1 to {val(u,'macro_f1'):.4f}, and MAE to {val(u,'mae'):.4f}. "
    f"The balanced variant reaches {100*val(b,'accuracy'):.2f}\\% accuracy and {val(b,'macro_f1'):.4f} macro-F1, with MAE {val(b,'mae'):.4f} and LER {100*val(b,'large_error_rate'):.2f}\\%. "
    f"Compared with the unweighted model, balancing reduces MAE by {mae_reduction:.2f}\\% and LER by {ler_reduction:.2f}\\% relative, while increasing macro-F1 by {val(b,'macro_f1')-val(u,'macro_f1'):.4f}. "
    "All four aggregate metrics improve in this pilot; the remaining trade-offs occur at the classwise precision/recall level. "
    "These gains concern class weighting in a linear model, so they do not satisfy or test the proposal's CE-versus-ordinal-loss success criterion. "
    "All six fits terminated before the 30-epoch cap without convergence warnings: 25 epochs for each unweighted run and 26 for each balanced run.")
cu=read('per_class_unweighted_17.json'); cb=read('per_class_balanced_17.json')
write('error_discussion.tex',
    f"For seed 17, two-star recall increases from {100*cu['2']['recall']:.2f}\\% to {100*cb['2']['recall']:.2f}\\%, "
    f"while its precision decreases from {100*cu['2']['precision']:.2f}\\% to {100*cb['2']['precision']:.2f}\\%. "
    f"Five-star recall decreases from {100*cu['5']['recall']:.2f}\\% to {100*cb['5']['recall']:.2f}\\%, "
    f"but five-star precision increases from {100*cu['5']['precision']:.2f}\\% to {100*cb['5']['precision']:.2f}\\%. "
    "Class balancing therefore shifts the decision boundary toward less frequent ratings rather than improving every precision and recall simultaneously. ")

# Constructed probability vectors: this is an illustration, never a trained-model result.
def losses(p,y,alpha=1,lam=.5):
    p=np.asarray(p,dtype=float)
    ce=-np.log(p[y-1]); d=np.abs(np.arange(1,6)-y).astype(float)**alpha
    oll=-np.sum(d*np.log1p(-p))
    return [float(ce),float(oll),float(lam*ce+(1-lam)*oll)]
near=[.10,.80,.04,.03,.03]; far=[.10,.03,.03,.04,.80]
ln,lf=losses(near,1),losses(far,1)
assert np.isclose(ln[0],lf[0]) and lf[1]>ln[1]
assert np.isclose(losses(near,1,lam=1)[2],ln[0])
assert np.isclose(losses(near,1,lam=0)[2],ln[1])
write('loss_illustration.json',json.dumps(dict(y=1,alpha=1,lambda_ce=.5,near=near,far=far,
    near_losses=ln,far_losses=lf,note='Constructed vectors, not model predictions; endpoint and ordering assertions passed.'),indent=2))
write('loss_table.tex',table('Computed loss illustration, not a trained experiment. True rating is 1, $\\alpha=1$, and $\\lambda=0.5$.',
    'tab:loss','lrrr','Probability mass pattern & CE & OLL & MLL',
    [['Near-error vector']+[f'{x:.3f}' for x in ln],['Far-error vector']+[f'{x:.3f}' for x in lf]],
    foot=r'\par\smallskip\parbox{\columnwidth}{\scriptsize Near: $(.10,.80,.04,.03,.03)$. Far: $(.10,.03,.03,.04,.80)$. Entries follow star order 1--5.}'))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.5,2.35),layout='constrained')
x=np.arange(1,6)
for offset,split,color,label in [(-.18,'train','#24546a','Train'),(.18,'validation','#b36b27','Validation')]:
    values=np.array([a['splits'][split]['rating_counts'][str(i)] for i in x])/a['splits'][split]['rows']*100
    ax.bar(x+offset,values,width=.34,color=color,label=label)
ax.set(xlabel='Star rating',ylabel='Share of reviews (%)',xticks=x,ylim=(0,75))
ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(F/'data_profile.pdf');fig.savefig(F/'data_profile.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained')
for ax,variant,label in zip(axes,['unweighted','balanced'],['Unweighted','Class-balanced']):
    cm=np.asarray(read(f'confusion_{variant}_17.json'))
    pct=100*cm/cm.sum(axis=1,keepdims=True)
    ax.imshow(pct,cmap='Blues',vmin=0,vmax=100)
    ax.set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=x,yticklabels=x,
        xlabel='Predicted rating',ylabel='True rating',title=label)
    ax.spines['top'].set_visible(True);ax.spines['right'].set_visible(True)
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f'{pct[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if pct[i,j]>55 else '#173146')
fig.savefig(F/'confusion.pdf');fig.savefig(F/'confusion.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,2.28))
ax.set(xlim=(0,10),ylim=(0,3.1));ax.axis('off')
def box(x,y,w,h,text,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',
        lw=1,linestyle='--' if dashed else '-',edgecolor='#24546a',facecolor='#f5f7f8' if dashed else '#eaf1f4'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#506574',lw=1))
box(.12,1.92,1.7,.9,'568,454 raw\nFine Foods reviews')
box(2.15,1.92,2.03,.9,'Normalize + deduplicate\n393,001 retained')
box(4.53,1.92,2.0,.9,'Product-hash split\n70 / 15 / 15 targets')
box(7.03,1.92,2.72,.9,'TF-IDF baselines\nValidation: 56,791 reviews\n2 variants x 3 seeds')
arrow((1.85,2.37),(2.1,2.37));arrow((4.22,2.37),(4.48,2.37));arrow((6.57,2.37),(6.98,2.37))
box(4.53,.22,2.0,1.0,'TextCNN / DistilBERT\nCE / OLL / MLL\nMatched validation\nsearch',True)
box(7.03,.22,2.72,1.0,'Freeze selected settings\n3 seeds; then final test\n59,366 reserved reviews',True)
arrow((5.53,1.87),(5.53,1.27));arrow((6.57,.72),(6.98,.72))
ax.text(.13,.73,'Solid: completed\nDashed: planned',fontsize=8,color='#405362',va='center')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
fig.savefig(F/'pipeline.pdf');fig.savefig(F/'pipeline.png',dpi=220);plt.close(fig)
print('Generated 3 figures, 3 measured tables, 1 loss-illustration table, and result text.')
