import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
d=json.loads((root/'checkpoint_statistics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172B4D','axes.labelcolor':'#526177','xtick.color':'#526177','ytick.color':'#526177','svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(12,5))
for ax,(metric,title,sub,color) in zip(axes,[('input','Input → Output A','Same reference A at every checkpoint','#3568B0'),('injection','Plasticity','Correction → Output B','#D78132'),('empowerment','Empowerment','Output A → Output B','#25998D')]):
 r=sorted([x for x in d['means'] if x['metric']==metric],key=lambda x:x['rlvr_step'])
 x=[v['rlvr_step'] for v in r];y=[v['mean'] for v in r]
 ax.errorbar(x,y,yerr=[[v['mean']-v['ci95_low'] for v in r],[v['ci95_high']-v['mean'] for v in r]],color=color,lw=2.4,marker='o',ms=7,capsize=4,elinewidth=1.3)
 for xx,v in zip(x,r):ax.annotate(f"{v['mean']:.4f}",(xx,v['ci95_high']),xytext=(0,7),textcoords='offset points',ha='center',color=color,fontweight='bold',fontsize=10)
 ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=31);ax.text(0,1.035,sub,transform=ax.transAxes,fontsize=9,color='#526177')
 ax.set_xticks(x);ax.set_xlim(-32,283);ax.set_xlabel('Additional GRPO steps',labelpad=8);ax.set_ylabel('Entropy difference (bits/token)',fontsize=9)
 low=min(0,min(v['ci95_low'] for v in r));high=max(v['ci95_high'] for v in r);gap=high-low
 ax.set_ylim(low-.13*gap if low<0 else 0, high+.24*gap)
 ax.axhline(0,color='#9EACBD',lw=.9,ls='--');ax.grid(axis='y',color='#E5EBF2');ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 for side in ['bottom','left']:ax.spines[side].set_color('#D8E0E9')
 ax.tick_params(length=0,pad=5)
fig.suptitle('Fixed earlier output + a calculation opening',x=.065,y=.98,ha='left',fontsize=19,fontweight='bold')
fig.text(.065,.89,'Qwen3-0.6B • Original 50 questions × 10 reference spans × 3 checkpoints = 1,500 evaluations',fontsize=10,color='#526177')
fig.text(.065,.085,'B starts with “The calculation is”; scores cover the next 20 sampled tokens. All evaluations completed.',fontsize=9,color='#526177')
fig.text(.065,.047,'Bars: pointwise 95% question-bootstrap intervals. Equal question weights; separate y-axis scales.',fontsize=8.5,color='#526177')
fig.text(.065,.012,'Plasticity and empowerment are working names for entropy-ablation proxies. A is fixed; B varies across checkpoints.',fontsize=8.5,color='#526177')
fig.subplots_adjust(left=.065,right=.98,top=.71,bottom=.24,wspace=.4)
fig.savefig(root/'fixed_a_opening_trends.png',dpi=220,facecolor='white');fig.savefig(root/'fixed_a_opening_trends.svg',facecolor='white')
