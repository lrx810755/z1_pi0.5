import json,collections
from pathlib import Path
R=Path(__file__).parent

def category(d):
 if d['passed']:return '通过'
 e=d.get('error','')
 if d.get('failed_stage')=='reset':return '复位或试验协议失败'
 if d.get('failed_stage')=='planning':return '规划失败'
 if 'Finger/table contact' in e:return '接近时夹爪触桌'
 if '闭爪前' in e:return '闭爪前定位失败'
 if '预紧位移过大' in e:return '闭爪位移超限'
 if '最终关节误差' in e or 'within tolerance' in e:return '机械臂轨迹跟踪超限'
 if '未到达任务安全位' in e:return '搬运目标高度或位置不足'
 if '纯物理抬升' in e:return '抬升或运输中丢失夹持'
 if '连续夹持验证未通过' in e:return '闭爪后夹持不足'
 return '其他失败'

def batch(prefix):
 out=[]
 for p in sorted((R/'cases').glob(prefix+'*/result.json')):
  d=json.loads(p.read_text());c=d['case'];s=d.get('final_status',{})
  out.append(dict(id=c['id'],cup_position_m=c['cup_position_m'],overrides=c.get('relative_target_overrides',{}),profile_overrides=c.get('profile_overrides',{}),passed=d['passed'],category=category(d),maximum_rise_mm=round(d['maximum_rise_m']*1000,3),final_root_z_mm=round(s.get('object_position_m',[0,0,0])[2]*1000,3),error=d.get('error'),table_contact=d.get('table_contact')))
 return dict(tested=len(out),passed=sum(d['passed'] for d in out),categories=dict(collections.Counter(d['category'] for d in out)),cases=out)

if __name__=='__main__':
 result={p:batch(p) for p in ['original_grid','original_range','screen_','optimized_grid','optimized_range']};(R/'阶段汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:{a:v[a] for a in ['tested','passed','categories']} for k,v in result.items()},ensure_ascii=False,indent=2))
