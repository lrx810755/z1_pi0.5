import json,collections,hashlib
from pathlib import Path
from experiment_core import valid_trial,DISTRACTOR_XY
from analysis import failure_kind,evidence
L=Path('/home/lp531/z1_queue_logs/cup_shape_height_test_20261108')
def report():
 results=[r for p in sorted((L/'cases').glob('*/result.json')) if valid_trial(r:=json.loads(p.read_text()))]
 for r in results:r.update(evidence(L/'cases'/r['case']['id']),failure_kind=failure_kind(r))
 groups=[]
 for shape in ['frustum','cylinder']:
  for height in [34,32]:
   for position in ['near','middle','far']:
    rows=[r for r in results if (r['case']['shape'],r['case']['height_mm'],r['case']['position'])==(shape,height,position)]
    groups.append(dict(shape=shape,height_mm=height,position=position,attempts=len(rows),successes=sum(r['passed'] for r in rows),closure_attempts=sum(r['closure_attempted'] for r in rows),failure_kinds=dict(collections.Counter(r['failure_kind'] for r in rows if not r['passed'])),contact_heights_mm=[r['first_bilateral_contact_height_mm'] for r in rows if r['first_bilateral_contact_height_mm']],failures=dict(collections.Counter(r.get('failed_stage','verification') for r in rows if not r['passed'])),table_contact=sum(r['table_contact'] for r in rows)))
 source=json.loads((L/'original_source_hashes.json').read_text());changed=[p for p,h in source.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 payload=dict(distractor_xy_m=DISTRACTOR_XY,completed=len(results),expected=120,complete=len(results)==120,successes=sum(r['passed'] for r in results),groups=groups,production_sources_changed=changed)
 (L/'case_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 (L/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 lines=['# 杯体形状与抓取高度实验','',f"已完成 {len(results)}/120 次；物理抓取及抬升通过 {payload['successes']} 次。",'', '|形状|高度毫米|位置|完成|实际进入闭爪|完整抓取服务通过|手指碰桌次数|失败分类|','|---|---:|---|---:|---:|---:|---:|---|']
 for g in groups:lines.append('|'+ '|'.join(map(str,[g['shape'],g['height_mm'],g['position'],g['attempts'],g['closure_attempts'],g['successes'],g['table_contact'],g['failure_kinds']]))+'|')
 lines+=['','成功要求：原生产引导抓取通过、双指物理夹持有效、最终离桌、杯底上升至少30毫米；沿用原服务内部自带的带载抬升到蓝盒上方验证；未调用放置服务，不包含下降放置／松爪。34/32毫米是标定目标高度，实际手指接触高度保存在scene_events及trace中；未规划到位者不能用于比较夹持性能。','近／中／远：原工作区第1／3／5列、同一第2行；固定XY分别(0.393,-0.141)、(0.525,-0.141)、(0.657,-0.141)米。','每次完整重新复位及规划；规划失败也保留，单列阶段，未用成功重试凑数。','倒圆台：底半径18／顶半径25.2／高61毫米；原柱体：半径25.2／高61毫米。保持原扫描外观、80克质量和所有控制／验收参数。其他物体统一固定在y=0.12米一侧，避免与测试点重叠；所有120次使用同一布置。','每个组实际Isaac及MoveIt几何证据分别见isaac_geometry.json、moveit_geometry_verified.json。','每次尝试在cases目录保留result.json、operations.json、trace.jsonl与三个视角MP4；复位前失败可能没有动作录像。',f'生产源文件变化：{changed}']
 (L/'实验结果.md').write_text('\n'.join(lines)+'\n')
 return payload
if __name__=='__main__':print(json.dumps(report(),ensure_ascii=False))
