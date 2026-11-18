"""Additional predicate audit and video contact sheets from recorded frames."""
import json
from pathlib import Path
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

A=Path(__file__).resolve().parent
d=json.loads((A/'failure_audit.json').read_text())


def roll(q):
    r=Rotation.from_quat(q).as_matrix();x=r[:,0];up=np.array([0.,0.,1.])
    z=up-float(up@x)*x;z/=np.linalg.norm(z);y=np.cross(z,x)
    return math.degrees(math.atan2(r[:,1]@z,r[:,1]@y))


details=[]
for item in d['episodes']:
    root=Path(item['source_result']).parent
    rows=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
    other_good=[];all_good=[];open_pair=[]
    initial=rows[0]['contact_status']['object_position_m']
    for row in rows:
        s=row['contact_status']
        pair=s.get('left_finger_contact_object')==s.get('right_finger_contact_object')=='cup'
        if pair and row['execution_target'][-1]<-.8:
            open_pair.append(row)
        valid=(pair and row['execution_target'][-1]>-.2
            and s.get('gripper_obstructed') is True and s.get('contact_angle_valid') is True
            and s.get('inside_grasp_envelope') is True and s.get('relative_pose_stable') is True
            and s.get('geometry_contact_valid') is True and s.get('physical_test_lift_active') is True
            and s.get('object_touching_table') is False
            and s['object_position_m'][2]-initial[2]>=.03
            and not s.get('left_finger_touching_table') and not s.get('right_finger_touching_table'))
        if valid:
            other_good.append(row)
            if s.get('cylinder_tilt_valid') is True:
                all_good.append(row)
    close=item['events']['first_close']['trace_index']
    near=[(i,r) for i,r in enumerate(rows[:close+1]) if r['contact_status']['object_position_m'][2]>-.01
          and r['contact_status']['gripper_object_distance_m']<.22]
    first_tilt=next((i for i,r in enumerate(rows[:close+1]) if (r['contact_status'].get('cylinder_tilt_deg') or 0)>10),None)
    extra=dict(episode=item['episode'],other_grasp_gates_passed_samples=len(other_good),all_grasp_gates_passed_samples=len(all_good),
        min_tilt_when_other_gates_passed_deg=min((r['contact_status']['cylinder_tilt_deg'] for r in other_good),default=None),
        max_tilt_when_other_gates_passed_deg=max((r['contact_status']['cylinder_tilt_deg'] for r in other_good),default=None),
        first_close_tool_roll_deg=roll(rows[close]['contact_status']['gripper_world_orientation_xyzw']),
        first_tilt10_before_close_s=(rows[first_tilt]['simulation_elapsed_s'] if first_tilt is not None else None),
        first_close_s=rows[close]['simulation_elapsed_s'],
        any_bilateral_while_open=bool(open_pair),max_z_bilateral_while_open=max((r['contact_status']['object_position_m'][2] for r in open_pair),default=None),
        close_attempt_pose_roll_range_deg=[min((roll(r['contact_status']['gripper_world_orientation_xyzw']) for i,r in near),default=None),max((roll(r['contact_status']['gripper_world_orientation_xyzw']) for i,r in near),default=None)])
    details.append(extra)
    if item['episode'] not in [2,3,4,5,9,18,24]:
        continue
    # Video adds one pre-inference frame per prediction plus each completed action.
    written=np.cumsum([int(r['execution_substep']==5 and not r['policy_action_paused'])+
        int(r['policy_action_index']==0 and r['execution_substep']==0) for r in rows])
    events=[('preclose',max(0,close-30)),('first close',close),
            ('contact / nearest',item['events'].get('first_bilateral_contact',item['events']['best_closed_alignment'])['trace_index']),
            ('max cup height',item['events']['maximum_height']['trace_index']),('final',len(rows)-1)]
    montage=np.full((3*270,5*320,3),245,np.uint8)
    mappings=[]
    for j,name in enumerate(['wrist','global_third_person','global_right']):
        cap=cv2.VideoCapture(str(root/'videos'/f'{name}.mp4'))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert 0<=count-int(written[-1])<=1,(item['episode'],name,count,int(written[-1]))
        for k,(label,i) in enumerate(events):
            frame=min(count-1,max(0,int(written[i])-1));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,img=cap.read();assert ok
            img=cv2.resize(img,(320,240))
            montage[j*270+30:j*270+270,k*320:(k+1)*320]=img
            cv2.putText(montage,f'EP{item["episode"]:02d} {name[:6]} {label}',(k*320+4,j*270+13),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
            cv2.putText(montage,f'sim={rows[i]["simulation_elapsed_s"]:.2f}s frame={frame}',(k*320+4,j*270+26),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
            mappings.append(dict(camera=name,event=label,trace_index=i,video_frame=frame))
        cap.release()
    cv2.imwrite(str(A/f'episode_{item["episode"]:02d}_events.jpg'),montage)
    (A/f'episode_{item["episode"]:02d}_video_map.json').write_text(json.dumps(mappings,indent=2))

(A/'predicate_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
failed=[r for r in d['episodes'] if not r['final_box_success']]
stats=dict(final_failed_preclose_tilt_over10=[r['episode'] for r in failed if r['preclose_max_tilt_deg']>10],
    final_failed_preclose_tilt_over60=[r['episode'] for r in failed if r['preclose_max_tilt_deg']>60],
    no_contact_no_major_preclose_tilt=[r['episode'] for r in failed if not r['ever_bilateral'] and r['preclose_max_tilt_deg']<=10],
    final_only_predicate_details=[r for r in details if d['episodes'][r['episode']-1]['final_box_success'] and not d['episodes'][r['episode']-1]['strict_success']])
(A/'mechanism_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stats,ensure_ascii=False,indent=2))
