import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'four_steps_results.json').read_text());arms=report['arms'];names=list(arms)
x=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(10,4))
for offset,key,label,color in [(-.25,'successes','Task success','#2f855a'),(0,'closed_trials','Closing command','#3182ce'),(.25,'physical_grasp_trials','Physical grasp','#805ad5')]:
 heights=[arms[name][key] for name in names];bars=axes[0].bar(x+offset,heights,.23,label=label,color=color)
 axes[0].bar_label(bars,padding=3)
axes[0].set_xticks(x,names);axes[0].set_ylim(0,9.7);axes[0].set_ylabel('Count / 9 trials');axes[0].set_title('Corrected collision physics');axes[0].legend(loc='upper right',fontsize=8)
axes[1].bar(names,[arms[name]['median_command_hz'] for name in names],color='#718096');axes[1].axhline(10,color='#c53030',linestyle='--',label='Dataset: 10 Hz');axes[1].set_ylim(0,11);axes[1].set_ylabel('Commands / simulation second');axes[1].set_title('Actual command timing');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'corrected_comparison.png',dpi=180);plt.close(fig)
