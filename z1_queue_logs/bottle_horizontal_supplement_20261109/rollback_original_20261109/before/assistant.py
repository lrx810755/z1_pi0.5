#!/usr/bin/python3
"""Independent cup approach, native Pilz blended sequence, one arm execution."""
import importlib.util,sys,copy,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import Settings,route_candidates,validate_timing,search_strategies
from geometry import Geometry
from combined_transport import CombinedBottleTransport
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('z1_isaac_bridge.horizontal_base_snapshot',R/'base_assistant_snapshot.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
# Private bottle supplement: operator-selected relative heading order, retaining duplicates.
base.GRASP_PROFILES['beverage_bottle']['candidate_azimuth_offsets_deg']=(0.,0.,30.,45.,45.,-30.,-45.,-45.)
base.GRASP_PROFILES['beverage_bottle']['candidate_count']=8+len(base.GRASP_PROFILES['beverage_bottle'].get('extra_candidate_azimuths_deg',()))
from moveit_msgs.srv import GetMotionSequence
from moveit_msgs.msg import MotionSequenceItem,RobotState,Constraints,JointConstraint
class HorizontalAssistant(CombinedBottleTransport,base.GraspAssistant):
    def __init__(self):
        super().__init__()
        if self.target_object!='beverage_bottle' or self.ee_link!='link06':raise RuntimeError('Independent recorder requires beverage_bottle and link06')
        self.declare_parameter('bottle_collection_column',-1)
        self.settings=Settings(speed_base=self.grasp_alignment_velocity_scale,maximum_axis_pitch_deg=8.)
        self.geometry=Geometry();self.sequence=self.create_client(GetMotionSequence,'/plan_sequence_path');self.new_plan=None
    def _start_state(self,positions,names=None):
        state=RobotState();state.is_diff=True
        state.joint_state.name=list(names or [f'joint{i}' for i in range(1,7)])
        state.joint_state.position=list(positions)
        if 'jointGripper' not in state.joint_state.name:
            state.joint_state.name.append('jointGripper');state.joint_state.position.append(self.gripper_open)
        state.joint_state.velocity=[0.]*len(state.joint_state.name)
        return state
    def _calibrated_candidates(self,obj):
        # Pure pose generation; no legacy approach path or sweep precondition.
        import math
        profile=base.GRASP_PROFILES['beverage_bottle'];geometry_profile=base.OBJECT_GRASP_PROFILES['beverage_bottle']
        transform=self._tf_buffer.lookup_transform(self.ee_link,'gripperStator',base.Time()).transform
        t=transform.translation;q=transform.rotation;tool_q=(q.x,q.y,q.z,q.w)
        calibrated=profile['calibrated_grasp_reference'];count=int(profile.get('candidate_count',self.candidate_count));branches=len(profile['candidate_azimuth_offsets_deg']);out=[]
        for index in range(count):
            c=self._sample_candidate(obj,'beverage_bottle',candidate_index=index,avoid_box=False)
            pitch,roll=base._bottle_candidate_pose_variant(index%branches,branches,float(geometry_profile.get('approach_pitch_max_deg',profile['pitch_deg'])),float(geometry_profile.get('approach_pitch_min_deg',profile['pitch_deg'])),float(profile['roll_deg']),bool(geometry_profile.get('mirror_wrist_roll_candidates',False)))
            azimuth=base._candidate_azimuth_deg(c['azimuth_deg'],index,profile['candidate_azimuth_offsets_deg'],profile.get('extra_candidate_azimuths_deg',()))
            az=math.radians(azimuth);ph=math.radians(pitch)
            c.update(azimuth_deg=azimuth,pitch_deg=pitch,roll_deg=roll,orientation=base.orientation_for_approach((math.cos(ph)*math.cos(az),math.cos(ph)*math.sin(az),math.sin(ph)),roll))
            local=list(calibrated['object_in_gripper_local_m']);local[0]-=float(geometry_profile.get('additional_insertion_depth_m',0.))
            reference=base.quaternion_multiply(c['orientation'],tool_q) if profile.get('alignment_reference_uses_tool_rotation',False) else c['orientation']
            desired=base._bottle_reference_for_orientation(tuple(local),reference,float(calibrated['captured_object_center_minus_stator_world_z_m']))
            desired=base._cylinder_candidate_local_reference(desired,index,profile.get('candidate_local_offsets_yz_m',()))
            point=c['target_point']
            if profile.get('alignment_geometry_z_offset_m') is not None:point=(obj[0],obj[1],obj[2]+float(profile['alignment_geometry_z_offset_m']))
            c['position']=base._ee_position_for_object_in_stator(point,c['orientation'],(float(t.x),float(t.y),float(t.z)),tool_q,desired)
            center_z=obj[2]+float(base.GRASP_CONFIG['object_center_z_offsets_m']['beverage_bottle']);stator_q=base.quaternion_multiply(c['orientation'],tool_q)
            c.update(desired_grasp_point_in_stator_m=desired,planning_geometry_point_m=point,desired_object_in_stator_m=base._object_in_stator_after_world_tool_offset(desired,stator_q,(0.,0.,point[2]-center_z)),standoff_m=math.dist(c['target_point'],c['position']),final_grasp_target=True)
            out.append(c)
        return out
    def _collection_column(self):
        column=int(self.get_parameter('bottle_collection_column').value)
        if column not in range(5):raise RuntimeError('采集前必须设置 bottle_collection_column 为0..4')
        return column
    def _plan_no_retreat(self,start,final,orientation):
        self._raise_if_stop_requested()
        result=self._plan_candidate(dict(position=final.tolist(),orientation=orientation.tolist()),
            linear=False,open_gripper=True,start_state=self._start_state(start),
            position_tolerance=.001,orientation_tolerance=.01,
            velocity_scale=self.settings.final_speed,acceleration_scale=self.settings.final_speed)
        _,max_delta=self._trajectory_score(result.planned_trajectory.joint_trajectory)
        limit=float(base.GRASP_PROFILES['beverage_bottle'].get('max_joint_delta_rad',self.max_joint_delta))
        if max_delta>limit:raise RuntimeError('No-retreat path exceeds original joint-change limit')
        return result.planned_trajectory
    def _generate_plan(self):
        self.new_plan=None
        column=self._collection_column()
        with self._lock:
            self._approved_trajectory=None;self._approved_target='';self._grasp_candidate=None
            start=[self._joint_positions[f'joint{i}'] for i in range(1,7)]
        self._set_bottle_finger_collision(False)
        self._command_scene_grasp('unfreeze','beverage_bottle')
        if not self._wait_scene_grasp('UNFROZEN','',2.):raise RuntimeError('Bottle must remain dynamic before planning')
        obj=np.asarray(self._current_object_position('beverage_bottle'));candidates=self._calibrated_candidates(obj)
        self._set_status('PLANNING','正在搜索外侧高位/低位连续通道',False)
        import xacro,xml.etree.ElementTree as ET
        from geometry import URDF
        robot=ET.fromstring(xacro.process_file(str(URDF)).toxml())
        for i in range(6):
            limit=robot.find(f"joint[@name='joint{i+1}']/limit")
            clipped=float(np.clip(start[i],float(limit.get('lower')),float(limit.get('upper'))))
            if abs(clipped-start[i])>.001:raise RuntimeError('Start outside joint limit by more than 0.001rad')
            start[i]=clipped
        sequence_available=(column!=0 and self.sequence.wait_for_service(timeout_sec=3))
        searches=[]
        for candidate_index,candidate in enumerate(candidates):
            final=np.asarray(candidate['position']);orientation=np.asarray(candidate['orientation']);axis=Rotation.from_quat(orientation).as_matrix()[:,0]
            if abs(axis[2])>np.sin(np.deg2rad(self.settings.maximum_axis_pitch_deg)):continue
            searches.append((candidate_index,candidate,route_candidates(final,obj,axis,self.settings)))
        rejected=[];trajectory=None
        # Exhaust all direct-low candidates before permitting high fallback.
        # Each request remains bounded; only high fallback has a phase deadline.
        route_count=len(searches[0][2]) if searches else 0
        direct_indices=[i for i,route in enumerate(searches[0][2]) if route['height_m']==.080 and route['azimuth_deg']==0.] if searches else []
        for strategy in search_strategies(column):
            if strategy!='no_retreat' and not sequence_available:
                rejected.append(dict(strategy=strategy,error='Pilz sequence service unavailable; using direct MoveIt fallback'))
                continue
            indices=([None] if strategy=='no_retreat' else direct_indices if strategy=='direct_low' else range(route_count))
            phase_deadline=time.monotonic()+90. if strategy=='via_high' else float('inf')
            for route_index in indices:
                for candidate_index,candidate,routes in searches:
                    self._raise_if_stop_requested()
                    if time.monotonic()>=phase_deadline:break
                    final=np.asarray(candidate['position']);orientation=np.asarray(candidate['orientation'])
                    route=(dict(strategy='no_retreat',high=None,low=final,retreat_m=0.,height_m=0.,azimuth_deg=0.)
                           if strategy=='no_retreat' else dict(routes[route_index],strategy=strategy))
                    try:
                        if strategy=='no_retreat':
                            result=self._plan_no_retreat(start,final,orientation)
                        else:
                            segment_count=2 if strategy=='direct_low' else 3
                            candidate_deadline=min(phase_deadline,time.monotonic()+segment_count*self.planning_time+15.)
                            result=self._plan_route(start,route,final,orientation,candidate_deadline)
                        evidence=self._audit_trajectory(result,start,route,final,orientation)
                        if time.monotonic()>=phase_deadline:raise RuntimeError('Planning phase deadline reached during geometry audit')
                        trajectory=result;selected=route;selected_candidate=copy.deepcopy(candidate);break
                    except (RuntimeError,ValueError) as exc:
                        rejected.append(dict(strategy=strategy,grasp_candidate=candidate_index,corridor_candidate=route_index,retreat_m=route['retreat_m'],height_m=route['height_m'] if strategy=='via_high' else None,azimuth_deg=route['azimuth_deg'] if strategy=='via_high' else None,error=str(exc)))
                if trajectory is not None or time.monotonic()>=phase_deadline:break
            if trajectory is not None:break
        logs=Path('/home/lp531/z1_queue_logs/bottle_horizontal_supplement_20261109');logs.mkdir(parents=True,exist_ok=True)
        if trajectory is None:
            (logs/f'rejected_{time.time_ns()}.json').write_text(json.dumps(rejected,indent=2))
            raise RuntimeError(f'No valid continuous low-first/high-fallback corridor after {len(rejected)} candidates: '+(rejected[-1]['error'] if rejected else 'planning budget exhausted'))
        evidence.update(created_at=time.time(),version=4,collection_column=column,strategy=selected['strategy'],target='beverage_bottle',high_pose_m=(selected['high'].tolist() if selected['strategy']=='via_high' else None),outer_pose_m=(None if selected['strategy']=='no_retreat' else selected['low'].tolist()),final_pose_m=final.tolist(),orientation_xyzw=orientation.tolist(),retreat_m=selected['retreat_m'],azimuth_deg=selected['azimuth_deg'],height_m=selected['height_m'],final_speed_scaling=self.settings.final_speed,blend_radius_m=self.settings.blend_radius_m,rejected_candidates=rejected)
        (logs/f'plan_{time.time_ns()}.json').write_text(json.dumps(evidence,indent=2))
        self.plan_column=column
        self.plan_object=obj.copy();self.plan_candidate=json.dumps(selected_candidate,sort_keys=True,default=str);self.new_plan=trajectory
        with self._lock:self._approved_trajectory=trajectory;self._approved_target='beverage_bottle';self._grasp_candidate=selected_candidate
        if selected['strategy']=='no_retreat':return '不回退：MoveIt直接规划到最终抓取位'
        return ('外侧低位→水平精确对位：两段已连续融合' if selected['strategy']=='direct_low' else '直接接近不可用，外侧高位→低位→最终位：三段已连续融合')
    def _plan_route(self,start,route,final,orientation,deadline):
        self._raise_if_stop_requested()
        remaining=deadline-time.monotonic()
        if remaining<=1.:raise RuntimeError('Candidate search budget exhausted')
        request=GetMotionSequence.Request()
        segments=([(route['low'],.08,'PTP'),(final,self.settings.final_speed,'LIN')]
                  if route.get('strategy')=='direct_low' else
                  [(route['high'],.08,'PTP'),(route['low'],.08,'LIN'),(final,self.settings.final_speed,'LIN')])
        for i,(position,speed,planner) in enumerate(segments):
            item=MotionSequenceItem();req=item.req;req.group_name=self.group_name;req.pipeline_id='pilz_industrial_motion_planner';req.planner_id=planner;req.allowed_planning_time=min(self.planning_time,(remaining-1.)/len(segments));req.num_planning_attempts=1
            req.max_velocity_scaling_factor=float(speed);req.max_acceleration_scaling_factor=min(.08,float(speed));req.start_state.is_diff=True
            if i==0:req.start_state=self._start_state(start)
            req.goal_constraints=[self._pose_constraints(dict(position=position.tolist(),orientation=orientation.tolist()),position_tolerance=.001,orientation_tolerance=.01)]
            item.blend_radius=self.settings.blend_radius_m if i<len(segments)-1 else 0.;request.request.items.append(item)
        future=self.sequence.call_async(request)
        result=self._wait_future(future,min(len(segments)*self.planning_time+15,remaining))
        if result is None:
            future.cancel();raise RuntimeError('Three-segment sequence planning timed out')
        if result.response.error_code.val!=1:raise RuntimeError(f'Sequence planning error {result.response.error_code.val}')
        if len(result.response.planned_trajectories)!=1:raise RuntimeError('Expected one blended trajectory')
        return result.response.planned_trajectories[0]
    def _audit_trajectory(self,trajectory,start,route,final,orientation):
        jt=trajectory.joint_trajectory
        outer=route["low"];rotation=Rotation.from_quat(orientation)
        order=[jt.joint_names.index(f'joint{i}') for i in range(1,7)]
        q=np.array([[p.positions[i] for i in order] for p in jt.points]);v=[[p.velocities[i] for i in order] if len(p.velocities)==len(jt.joint_names) else [] for p in jt.points];times=[p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in jt.points]
        validate_timing(times,v)
        if np.max(np.abs(q[0]-np.asarray(start)))>.01:raise RuntimeError('Sequence start differs from current arm state')
        # Densely check actual blended path, not just requested endpoint positions.
        dense=[]
        for a,b in zip(q[:-1],q[1:]):
            dense.extend(a+(b-a)*t for t in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/.005))+1)))
        dense.append(q[-1]);self.geometry.check_path([list(x)+[self.gripper_open] for x in dense],floor=self.settings.finger_floor_m)
        positions=[];errors=[]
        direction=((final-outer)/np.linalg.norm(final-outer) if route['retreat_m']>0 else None)
        for joints in dense:
            p,rot=self.geometry.link_fk(joints);progress=(float(np.dot(p-outer,direction)) if direction is not None else 0.);distance=float(np.linalg.norm(p-final))
            angle=float((rotation.inv()*Rotation.from_matrix(rot)).magnitude())
            # Blending stays outside; all motion closer than that boundary must already be level/aligned.
            if distance<route["retreat_m"]-2*self.settings.blend_radius_m:
                if abs(p[2]-final[2])>self.settings.final_height_tolerance_m or angle>self.settings.final_orientation_tolerance_rad:raise RuntimeError('Near-object path still descends or rotates')
                lateral=np.linalg.norm((p-outer)-progress*direction)
                if lateral>.002:raise RuntimeError('Final entry is not a horizontal straight corridor')
            positions.append(p.tolist());errors.append(angle)
        if np.linalg.norm(np.asarray(positions[-1])-final)>.003:raise RuntimeError('Final FK does not match grasp goal')
        return dict(points=len(q),duration_s=times[-1],single_execution=True,all_internal_joint_velocities_nonzero=True,min_joint_velocity_inf_norm=float(np.min(np.max(np.abs(np.array(v)[1:-1]),axis=1))))
    def _execute_plan(self):
        try:
            with self._lock:
                if self.new_plan is None or self._approved_trajectory is not self.new_plan or self._approved_target!='beverage_bottle' or self.target_object!='beverage_bottle':
                    raise RuntimeError('New plan invalidated by reset/stop/target change')
                if json.dumps(self._grasp_candidate,sort_keys=True,default=str)!=self.plan_candidate:
                    raise RuntimeError('Grasp candidate changed after planning')
                if self.plan_column!=self._collection_column():raise RuntimeError('Grid column changed after planning')
                jt=self.new_plan.joint_trajectory
                current=np.array([self._joint_positions[n] for n in jt.joint_names])
                if np.max(np.abs(current-np.array(jt.points[0].positions)))>.01:
                    raise RuntimeError('Arm moved after planning; regenerate complete sequence')
            if np.linalg.norm(np.asarray(self._current_object_position('beverage_bottle'))-self.plan_object)>.003:
                raise RuntimeError('Bottle moved after planning; regenerate complete sequence')
            return super()._execute_plan()
        finally:
            self.new_plan=None
base.GraspAssistant=HorizontalAssistant
if __name__=='__main__':base.main()
