from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;a=json.loads((P/'coverage.json').read_text());font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
im=Image.new('RGB',(1300,850),'white');d=ImageDraw.Draw(im)
for panel,(title,key) in enumerate([('原始160条：各格条数','original'),('纠偏28条训练：各格条数','correction_train')]):
 x0=80+panel*650;y0=110;cw=100;ch=115;d.text((x0,20),title,font=font,fill='black')
 for c in a['cells']:
  x,y=c['grid'];n=c[key];shade=int(245-min(1,n/8)*130);box=(x0+x*cw,y0+(4-y)*ch,x0+(x+1)*cw,y0+(5-y)*ch);d.rectangle(box,fill=(shade,shade,255),outline='gray');d.text((box[0]+38,box[1]+40),str(n),font=font,fill='black')
 for i in range(6):d.text((x0+i*cw-15,700),f'{.36+i*.066:.3f}',font=font,fill='black');d.text((x0-70,y0+i*ch-10),f'{.16-i*.086:.3f}',font=font,fill='black')
 d.text((x0,750),'X / m（数值越大越远）；Y / m（向上增大）',font=font,fill='black')
im.save(P/'coverage_grid.png')
s='''# 纠偏示范空间覆盖检查

按192条训练清单，读取各episode_start.json中的瓶子初始位置和记录target_grid；32条纠偏中28条训练、4条验证。瓶子位置与夹爪偏移类别分别统计。以下坐标为世界坐标，不按摄像机画面的左右命名。

原始160条占20个格子，每格8条，分布规则均匀；纠偏32条占16格，但实际28条训练只占15格。实际范围：原始X=0.360397–0.689943m、Y=-0.268464–0.063707m；纠偏X=0.405127–0.687625m、Y=-0.266542–0.047026m。

## 集中区域

|X范围m|Y范围m|纠偏训练条数|占28条比例|说明|
|---|---|---:|---:|---|
|0.624–0.690|-0.098–-0.012|6|21.4%|最远一列的中间负Y区域|
|0.360–0.426|-0.270–-0.184|4|14.3%|近端、负Y边缘|
|0.492–0.558|-0.184–-0.098|4|14.3%|中部、负Y区域|

三格合计14/28=50%。整列X=0.624–0.690有10/28=35.7%，原始示范在该列为32/160=20%。X=0.426–0.492这一列只有2/28=7.1%。因此纠偏并非主要集中桌面中央，而是少数格子和最远一列偏多。

## 原始范围内没有纠偏训练样本的格子

|X范围m|Y范围m|备注|
|---|---|---|
|0.492–0.558|-0.270–-0.184|原始8条，纠偏0条|
|0.360–0.426|-0.184–-0.098|原始8条，纠偏0条|
|0.426–0.492|-0.184–-0.098|原始8条，纠偏0条|
|0.426–0.492|-0.098–-0.012|原始8条；纠偏2条均为验证集，没有参加训练|
|0.360–0.426|-0.012–0.074|原始8条，纠偏0条|

当前配置工作空间X=[0.36,0.69]、Y=[-0.27,0.16]，5×5划分。Y=[0.074,0.160]整行在本次192条清单中没有示范。这属于相对当前配置的覆盖空白，不等于能确认过去采集时就计划覆盖这一行，也不能据此证明当前评估实际采到了该行。

## 偏移类别也不均衡

28条纠偏训练：低姿态10、倾斜7、低且倾斜2、前后偏移far3/near2、right2、高姿态2、left0。低／倾斜／低且倾斜合计19/28=67.9%，与采集重点一致，但左右纠偏覆盖少且单侧缺失。left/right/far/near为夹爪相对正确抓取姿态的偏移类别，不是瓶子位于桌面的方位。

## 抽帧后是否还集中

按两个训练器的真实起始帧权重重新归一化，这三个集中格子分别占纠偏池预期概率总和：
'''
for name,k in [('第一次','old_correction_mass'),('第二次','new_correction_mass')]:
 vals=[c for c in a['cells'] if c['grid'] in [[4,2],[0,0],[2,1]]];mass=sum(c[k] for c in vals);s+=f'- {name}：占全部训练采样{mass*100:.2f}%，即纠偏池的{mass/.3*100:.2f}%。\n'
s+='''
因此随机抽帧不会自动均衡空间位置；片段长度和阶段权重还会改变各格子的实际训练关注。以上是预期采样概率，不是实测抽样计数。

## 结论与用途

纠偏空间分布确实不均衡，且方向类别不均衡。但原始160条在已覆盖20格内较均匀，因此不能说整个训练集只学了少数位置；也不能把空间不均衡直接认定为模型失败原因。最远一列纠偏数量不少，远端推理弱不能简单解释为该列没有数据，仍需检查该列纠偏类别、实际姿态、闭爪转换和位置边缘覆盖。

下一次若补录，可优先填补上述5个格子的训练纠偏缺口，并在已有集中格子内按实际失败姿态补缺；若只提高现有纠偏池权重，会同时放大现有集中。网格分层采样可以平衡已有样本的关注，但不能产生空格子里不存在的纠偏示范。

详见coverage.json（192条位置、类别、网格及两版预期采样概率）和coverage_grid.png。坐标界限按当前配置给出；原始记录target_grid与计数为直接读取。该检查描述瓶子初始空间覆盖，没有对纠偏全过程每个夹爪姿态做密度统计。
'''
(P/'纠偏覆盖检查.md').write_text(s)
assert sum(c['correction_train'] for c in a['cells'])==28;assert sum(c['correction_validation'] for c in a['cells'])==4;assert sum(c['original'] for c in a['cells'])==160
print('coverage counts verified; grid and report written')
