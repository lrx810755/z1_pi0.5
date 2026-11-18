exec(open('/home/lp531/z1_queue_logs/cup_wrist_penetration_20261026/occlusion.py').read().split('\na=json.loads')[0])
a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());out=[]
for n,f in [(2,.25),(8,.4),(9,.4),(18,.4),(20,.4),(22,.4)]:
 ep=next(e for e in a if e['episode']==n);tr=[json.loads(l) for l in (Path(ep['path'])/'trace.jsonl').read_text().splitlines()];times=[]
 for t in tr:
  if t['execution_substep']==0 and t['policy_action_index']==0:times.append(t['simulation_elapsed_s'])
  if t['execution_substep']==5:times.append(t['simulation_elapsed_s'])
 ti=times[int(len(times)*f)];t=min(tr,key=lambda t:abs(t['simulation_elapsed_s']-ti));s=t['contact_status'];rg=Rotation.from_quat(s['gripper_world_orientation_xyzw']);rc=Rotation.from_quat(s['object_orientation_xyzw']);offset=rg.inv().apply(np.array(s['object_position_m'])-s['gripper_world_position_m']);cup=rg.inv().apply(rc.apply(g[next(k for k,v in meta.items() if isinstance(v,dict) and v.get('path','').endswith('/Object_2'))]))+offset;ctri=cup[g['cup_faces']];pts=cup[::100];occ=0;center=offset+rg.inv().apply(rc.apply([0,0,.03]));q=np.append(center,1).dot(inv)
 mover=g[next(k for k in meta if k.startswith('mover_'))];mv=Rotation.from_rotvec([0,t['joint_state'][-1],0]).apply(mover)+[.049,0,0];tri=np.concatenate([verts[g['stator_faces']],mv[g['mover_faces']]],axis=0);v0=tri[:,0];e1=tri[:,1]-v0;e2=tri[:,2]-v0;tvec=origin-v0;cross=np.cross(tvec,e1)
 for p in pts:
  h=hit(p)
  if h and h['distance']<closest_triangles(p-origin,ctri)-.0005:occ+=1
 out.append(dict(episode=n,frame_fraction=f,time=ti,occluded=occ,total=len(pts),cup_center_projected=[float(320+250*q[0]/-q[2]),float(180-250*q[1]/-q[2])],cup_center_camera_depth=float(-q[2]),camera_to_cup_center=float(np.linalg.norm(center-origin)),cup_center_local=center.tolist()))
print(json.dumps(out,indent=2));(R/'suspect_frames_both_jaws.json').write_text(json.dumps(out,indent=2))
