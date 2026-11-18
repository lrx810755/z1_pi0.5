#!/usr/bin/python3
"""Independent cup approach, native Pilz blended sequence, one arm execution."""
import importlib.util,sys,copy,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import Settings,waypoints,validate_timing
from geometry import Geometry
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('z1_isaac_bridge.horizontal_base_snapshot',R/'base_assistant_snapshot.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
from moveit_msgs.srv import GetMotionSequence
from moveit_msgs.msg import MotionSequenceItem,RobotState,Constraints,JointConstraint
class HorizontalAssistant(base.GraspAssistant):
    def __init__(self):
        super().__init__()
        if self.target_object!='cup':raise RuntimeError('This independent recorder only supports cup')
        if self.ee_link!='link06':raise RuntimeError('FK audit currently requires link06 as planning link')
        self.settings=Settings(speed_base=self.grasp_alignment_velocity_scale)
        self.geometry=Geometry();self.sequence=self.create_client(GetMotionSequence,'/plan_sequence_path');self.new_plan=None
    def _generate_plan(self):
        self.new_plan=None
        # Frozen support code supplies calibrated cup grasp pose and acceptance metadata.
        # Its old approach trajectory is discarded and is never executed.
        super()._generate_plan()
        with self._lock:
            candidate=copy.deepcopy(self._grasp_candidate)
            start=[self._joint_positions[f'joint{i}'] for i in range(1,7)]
        if not candidate.get('final_grasp_target'):raise RuntimeError('Missing calibrated final grasp pose')
        # Reset can leave a sub-milliradian residual beyond a hard joint limit.
        # Match a bounded planning-start projection; never relax physical limits.
        import xacro,xml.etree.ElementTree as ET
        from geometry import URDF
        robot=ET.fromstring(xacro.process_file(str(URDF)).toxml())
        for i in range(6):
            limit=robot.find(f"joint[@name='joint{i+1}']/limit")
            clipped=float(np.clip(start[i],float(limit.get('lower')),float(limit.get('upper'))))
            if abs(clipped-start[i])>.001:raise RuntimeError('Start outside joint limit by more than 0.001rad')
            start[i]=clipped
        final=np.asarray(candidate['position']);orientation=np.asarray(candidate['orientation']);rotation=Rotation.from_quat(orientation)
        axis=rotation.as_matrix()[:,0]
        if abs(axis[2])>np.sin(np.deg2rad(self.settings.maximum_axis_pitch_deg)):raise RuntimeError('Final grasp pitch is not near horizontal')
        obj=np.asarray(self._current_object_position('cup'));outer=waypoints(final,obj,self.settings,approach_axis=axis)
        if not self.sequence.wait_for_service(timeout_sec=3):raise RuntimeError('Pilz blended sequence service unavailable; refuse stop/start fallback')
        # Seed exterior IK from the calibrated grasp branch, not the singular Home pose.
        from types import SimpleNamespace
        seed=RobotState();seed.is_diff=True
        seed.joint_state.name=list(self._approved_trajectory.joint_trajectory.joint_names)
        seed.joint_state.position=list(self._approved_trajectory.joint_trajectory.points[-1].positions)
        if 'jointGripper' not in seed.joint_state.name:
            seed.joint_state.name.append('jointGripper');seed.joint_state.position.append(self.gripper_open)
        outer_joints=None
        try:
            outer_plan=self._plan_candidate(dict(position=outer.tolist(),orientation=orientation.tolist()),linear=True,start_state=seed,open_gripper=True,position_tolerance=.001,orientation_tolerance=.01,velocity_scale=self.settings.final_speed)
            outer_joints=outer_plan.planned_trajectory.joint_trajectory
        except RuntimeError as exc:
            self.get_logger().info('Exterior seeded IK unavailable; try same blended sequence with Cartesian PTP goal: '+str(exc))
        request=GetMotionSequence.Request()
        for i,(position,speed,planner) in enumerate([(outer,.08,'PTP'),(final,self.settings.final_speed,'LIN')]):
            item=MotionSequenceItem();req=item.req;req.group_name=self.group_name;req.pipeline_id='pilz_industrial_motion_planner';req.planner_id=planner;req.allowed_planning_time=self.planning_time;req.num_planning_attempts=1
            req.max_velocity_scaling_factor=float(speed);req.max_acceleration_scaling_factor=min(.08,float(speed));req.start_state.is_diff=True
            if i==0:
                req.start_state.joint_state.name=[f'joint{j}' for j in range(1,7)]+['jointGripper'];req.start_state.joint_state.position=start+[self.gripper_open]
            req.goal_constraints=[self._pose_constraints(dict(position=position.tolist(),orientation=orientation.tolist()),position_tolerance=.001,orientation_tolerance=.01)]
            if i==0 and outer_joints is not None:
                goals=Constraints()
                for name,value in zip(outer_joints.joint_names,outer_joints.points[-1].positions):
                    joint=JointConstraint();joint.joint_name=name;joint.position=float(value);joint.tolerance_above=.0001;joint.tolerance_below=.0001;joint.weight=1.;goals.joint_constraints.append(joint)
                req.goal_constraints=[goals]
            item.blend_radius=self.settings.blend_radius_m if i==0 else 0.;request.request.items.append(item)
        result=self._wait_future(self.sequence.call_async(request),2*self.planning_time+15)
        if result is None or result.response.error_code.val!=1:raise RuntimeError('Complete PTP/LIN sequence planning failed')
        paths=result.response.planned_trajectories
        if len(paths)!=1:raise RuntimeError('Expected one blended trajectory, not separate stopped segments')
        trajectory=paths[0];jt=trajectory.joint_trajectory
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
        direction=(final-outer)/np.linalg.norm(final-outer)
        for joints in dense:
            p,rot=self.geometry.link_fk(joints);progress=float(np.dot(p-outer,direction));distance=float(np.linalg.norm(p-final))
            angle=float((rotation.inv()*Rotation.from_matrix(rot)).magnitude())
            # Blending stays outside; all motion closer than that boundary must already be level/aligned.
            if distance<self.settings.exterior_offset_m-2*self.settings.blend_radius_m:
                if abs(p[2]-final[2])>self.settings.final_height_tolerance_m or angle>self.settings.final_orientation_tolerance_rad:raise RuntimeError('Near-object path still descends or rotates')
                lateral=np.linalg.norm((p-outer)-progress*direction)
                if lateral>.002:raise RuntimeError('Final entry is not a horizontal straight corridor')
            positions.append(p.tolist());errors.append(angle)
        if np.linalg.norm(np.asarray(positions[-1])-final)>.003:raise RuntimeError('Final FK does not match grasp goal')
        evidence=dict(created_at=time.time(),target='cup',outer_pose_m=outer.tolist(),final_pose_m=final.tolist(),orientation_xyzw=orientation.tolist(),speed_base=self.settings.speed_base,speed_multiplier=.8,final_speed_scaling=self.settings.final_speed,blend_radius_m=self.settings.blend_radius_m,points=len(q),duration_s=times[-1],single_execution=True,all_internal_joint_velocities_nonzero=True,finger_floor_m=self.settings.finger_floor_m,min_joint_velocity_inf_norm=float(np.min(np.max(np.abs(np.array(v)[1:-1]),axis=1))))
        logs=Path('/home/lp531/z1_queue_logs/cup_horizontal_demo_20261105');logs.mkdir(parents=True,exist_ok=True);(logs/f'plan_{time.time_ns()}.json').write_text(json.dumps(evidence,indent=2))
        self.plan_object=obj.copy()
        self.plan_candidate=json.dumps(candidate,sort_keys=True,default=str)
        self.new_plan=trajectory
        with self._lock:self._approved_trajectory=trajectory;self._approved_target='cup'
        return '新接近轨迹已整体规划：外侧转姿下降→连续融合→0.8倍低速水平精确对位'
    def _execute_plan(self):
        try:
            with self._lock:
                if self.new_plan is None or self._approved_trajectory is not self.new_plan or self._approved_target!='cup' or self.target_object!='cup':
                    raise RuntimeError('New plan invalidated by reset/stop/target change')
                if json.dumps(self._grasp_candidate,sort_keys=True,default=str)!=self.plan_candidate:
                    raise RuntimeError('Grasp candidate changed after planning')
                jt=self.new_plan.joint_trajectory
                current=np.array([self._joint_positions[n] for n in jt.joint_names])
                if np.max(np.abs(current-np.array(jt.points[0].positions)))>.01:
                    raise RuntimeError('Arm moved after planning; regenerate complete sequence')
            if np.linalg.norm(np.asarray(self._current_object_position('cup'))-self.plan_object)>.003:
                raise RuntimeError('Cup moved after planning; regenerate complete sequence')
            return super()._execute_plan()
        finally:
            self.new_plan=None
base.GraspAssistant=HorizontalAssistant
if __name__=='__main__':base.main()
