from pathlib import Path
import json,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).parent
results=json.loads((r/'results.json').read_text())
names=list(dict.fromkeys(x['variant'] for x in results))
fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
for offset,pose,color in [(-.18,'hold_preclose','#3d75ad'),(.18,'hold_transport','#dd924a')]:
 vals=[next(x['joint3_mean_abs_error_rad']*1000 for x in results if x['variant']==n and x['pose']==pose) for n in names]
 axes[0].bar([i+offset for i in range(len(names))],vals,width=.35,color=color,label=pose.replace('hold_',''))
axes[0].axhline(12,color='red',linestyle='--',label='12 mrad controller threshold')
axes[0].set_xticks(range(len(names)),[n.replace('_','\n') for n in names],fontsize=8)
axes[0].set_ylabel('Joint 3 mean absolute error (mrad)');axes[0].legend(fontsize=8)
for name in names:
 p=r/name/'telemetry.jsonl';a=[json.loads(l) for l in p.read_text().splitlines()]
 a=[x for x in a if x['phase']=='hold_preclose']
 axes[1].plot([x['t']-6 for x in a],[1000*(x['q'][2]-x['target'][2]) for x in a],label=name,linewidth=1)
axes[1].set_xlabel('Time after fixed-pose hold starts (s)');axes[1].set_ylabel('Joint 3 signed error (mrad)')
axes[1].legend(ncol=3,fontsize=8);axes[1].grid(alpha=.2)
fig.savefig(r/'control_comparison.png',dpi=160)
fig.savefig(r/'control_comparison.pdf')
