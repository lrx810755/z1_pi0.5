import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,Patch
R=Path(__file__).parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(1,3,figsize=(16,5));colors={'通过':'#88c8a0','规划失败':'#c9cdd3','其他':'#eea48f'}
for ax,batch,title in zip(axes[:2],['original_grid20','optimized_grid_h40_20'],['34毫米原几何参数：流程9/20；流程+稳定复核8/20','40毫米候选：流程11/20；严格10/20；追加0/2，未采用']):
 ds=json.loads((R/(batch+'_results.json')).read_text())
 for d in ds:
  c,row=d['case']['target_grid'];label='通过' if d['passed'] else ('规划失败' if d.get('failed_stage')=='planning' else '失败')
  ax.add_patch(Rectangle((c,row),1,1,facecolor=colors.get(label,colors['其他']),edgecolor='white',linewidth=2));ax.text(c+.5,row+.5,f'({c},{row})\n{label}',ha='center',va='center',fontproperties=font,fontsize=10)
 ax.set(xlim=(0,5),ylim=(0,4),aspect='equal',xticks=[],yticks=[]);ax.set_title(title,fontproperties=font,fontsize=11)
ax=axes[2];ds=json.loads((R/'range_flexible9_results.json').read_text());xs=[.114,.12083045,.1275];zs=[.009,.015,.0215]
for d in ds:
 o=d['case']['relative_target_overrides'];c=xs.index(o['x_m']);row=zs.index(o['z_m']);label='通过' if d['passed'] else ('规划失败' if d.get('failed_stage')=='planning' else '搬运丢失')
 ax.add_patch(Rectangle((c,row),1,1,facecolor=colors.get(label,colors['其他']),edgecolor='white',linewidth=2));ax.text(c+.5,row+.5,label,ha='center',va='center',fontproperties=font,fontsize=10)
ax.set(xlim=(0,3),ylim=(0,3),aspect='equal',xticks=[.5,1.5,2.5],yticks=[.5,1.5,2.5]);ax.set_xticklabels(['114','120.83','127.5']);ax.set_yticklabels(['9','15','21.5']);ax.set_xlabel('固定指局部X：插入深度坐标（毫米）',fontproperties=font);ax.set_ylabel('固定指局部Z：标称横向参考（毫米）',fontproperties=font);ax.set_title('原范围采样：34毫米、原朝向和位置偏移候选',fontproperties=font,fontsize=11)
fig.suptitle('倒圆台真实物理抓取实验：固定−90°，独立规划，无ACT推理',fontproperties=font,fontsize=14)
fig.text(.5,.015,'每格中心各测一次；局部边界为离散采样。流程包含向蓝盒搬运；标称参考可叠加候选偏移，规划失败未进行物理闭爪。',ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(R/'实验结果对比.png',dpi=170);plt.close(fig)
