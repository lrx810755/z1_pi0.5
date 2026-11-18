from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).parent
before=np.load(R/'semantic18.npy');after=np.load(R/'semantic18_cup_hidden.npy');labels=json.loads((R/'semantic18.json').read_text())['idToLabels'];reverse={name.strip():int(k) for k,v in labels.items() for name in v['class'].split(',')};mask=before==reverse['cup'];under,counts=np.unique(after[mask],return_counts=True);under_counts={json.loads((R/'semantic18_cup_hidden.json').read_text())['idToLabels'][str(k)]['class']:int(v) for k,v in zip(under,counts)}
assert under_counts.get('robot',0)==0
col=np.full((*before.shape,3),195,dtype=np.uint8)
for name,color in [('robot',(215,75,75)),('table',(55,135,220)),('cup',(250,175,45))]:col[before==reverse[name]]=color
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
page=Image.new('RGB',(1280,800),'#171717');draw=ImageDraw.Draw(page)
for i,(name,title) in enumerate([('full_scene_trial18.png','A. Replayed Trial 18, ~24s'),('full_scene_trial18_cup_hidden.png','B. Cup hidden; grey strip remains'),('full_scene_trial18_robot_hidden.png','C. Robot + cup hidden; grey strip remains'),(None,'D. Red=robot, blue=table, orange=cup')]):
 x=(i%2)*640;y=(i//2)*400;im=Image.open(R/name) if name else Image.fromarray(col);page.paste(im,(x,y+32));draw.text((x+8,y+5),title,font=font,fill='white')
page.save(R/'遮挡核查对照.png');stats=dict(trial=18,approx_sim_time_s=24,cup_pixels=int(mask.sum()),under_cup_without_cup=under_counts,robot_transparency_enabled=False,grey_region_persists_when_robot_hidden=True,conclusion='Grey strip is the work surface, not the robot or its shadow; this replay does not show through-robot visibility. Other specific frames require individual verification.')
(R/'occlusion_verified.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False))
(R/'遮挡核查结论.md').write_text('''# 腕部相机“透过机械臂看杯子”核查\n\n核查对象：杯子独立场景30次评估中的19次未闭爪失败视频。查看19次各6个时刻的腕部画面，并选第18次约24秒做完整场景静态位姿复现。\n\n确认：该例中杯子后方看似机械臂延伸的浅灰区域是桌面。为机器人、桌面、杯子分别添加语义标签后，杯子像素在隐藏杯子后全部对应桌面或背景，没有对应机器人实体的像素。隐藏整个机器人后浅灰区域仍保留，因此该区域也不是机器人投影。\n\n机器人实际使用不透明OmniPBR灰色材质；opacity默认1且enable_opacity默认false。场景脚本未开启机器人透视。旧VLA相机外壳被隐藏，但不是机械臂本体。腕部相机设置20毫米近裁剪，机器人视觉网格doubleSided=false；这些是需逐帧关注的设置，并不能仅凭设置就认定为本次异常的原因。\n\n局限：此结论针对已复现的代表画面，不证明整批视频的每一帧都没有遮挡错误。先前确认的物理碰撞体轻微穿入是另一个问题，不应用来解释这里的渲染遮挡。\n\n对照图：遮挡核查对照.png。复现脚本：segment_scene.py。语义统计：occlusion_verified.json。所有修改仅存在诊断进程中的临时场景，未保存到训练或评估资产。\n''')
