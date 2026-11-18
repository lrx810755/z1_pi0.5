from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path(__file__).resolve().parent;a=json.loads((P/'timeline.json').read_text());r=a['records'];t=[x['time_s'] for x in r];font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[1,1.2]})
axes[0].step(t+[24.6],[x['label'] for x in r]+[r[-1]['label']],where='post',color='#167b91',linewidth=3)
axes[0].set_yticks([0,1]);axes[0].set_yticklabels(['0 开爪','1 闭爪'],fontproperties=font);axes[0].set_ylim(-.2,1.4);axes[0].set_title('真实训练标签：原始 episode_000058 → 合并训练集 episode 0',fontproperties=font,fontsize=15)
for mid,text in [(5,'接近：保持开爪'),(15.8,'闭爪 → 抬升 → 搬运：保持闭爪'),(23.1,'释放／撤离')]:axes[0].text(mid,1.2,text,ha='center',fontproperties=font,fontsize=10)
axes[1].plot(t,[x['raw_command_rad'] for x in r],label='示范原始夹爪指令',linewidth=2);axes[1].plot(t,[x['actual_joint_rad'] for x in r],label='实际夹爪关节角',linewidth=1.5);axes[1].legend(prop=font);axes[1].set_ylabel('角度 / rad',fontproperties=font);axes[1].set_xlabel('示范时间 / 秒',fontproperties=font)
for ax in axes:
 ax.axvline(10,color='#ca6d21',linestyle='--');ax.axvline(21.6,color='#ca6d21',linestyle='--');ax.grid(alpha=.2);ax.set_xlim(0,24.6)
axes[0].annotate('10.0秒：0 → 1',xy=(10,.5),xytext=(6,.65),arrowprops={'arrowstyle':'->'},fontproperties=font)
axes[0].annotate('21.6秒：1 → 0',xy=(21.6,.5),xytext=(17,.25),arrowprops={'arrowstyle':'->'},fontproperties=font)
fig.tight_layout();fig.savefig(P/'binary_timeline.png',dpi=160);plt.close(fig)
