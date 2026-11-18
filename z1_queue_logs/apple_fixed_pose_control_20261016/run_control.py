"""Independent fixed-pose experiment; never changes source or USD on disk."""
import sys,pathlib,faulthandler
faulthandler.enable()

source=pathlib.Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
sys.path.insert(0,str(source.parent))
s=source.read_text()
s=s.replace("    context.initialize_physics()", """    group = os.environ.get('APPLE_CONTROL_GROUP','pd')
    if group in ('position_iterations_double','velocity_iterations_increase'):
        api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(articulation_root))
        if group == 'position_iterations_double': api.CreateSolverPositionIterationCountAttr().Set(64)
        else: api.CreateSolverVelocityIterationCountAttr().Set(8)
    context.initialize_physics()""")
setup='''
    audit_root = "/home/lp531/z1_queue_logs/apple_fixed_pose_control_20261016/" + os.environ.get("APPLE_CONTROL_GROUP","pd")
    os.makedirs(audit_root,exist_ok=True)
    audit_rows = [json.loads(x) for x in open("/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016/episode_000044/frames.jsonl")]
    first_close = next(i for i,r in enumerate(audit_rows) if r["action"][-1] > -0.895)
    names = list(robot.dof_names)
    order = ["joint1","joint2","joint3","joint4","joint5","joint6","jointGripper"]
    poses = {
        "preclose":audit_rows[max(0,first_close-2)]["action"],
        "transport":next(r["action"] for r in audit_rows[first_close:] if r["phase"] == "TRANSPORT")}
    poses = {label: np.asarray([values[order.index(n)] if n != "jointGripper" else -0.9 for n in names],dtype=np.float32) for label,values in poses.items()}
    home=np.asarray([SAFE_HOME_BY_JOINT[n] for n in names],dtype=np.float32)
    ctrl=robot.get_articulation_controller()
    view=robot._articulation_view
    baseline_p,baseline_d=[np.asarray(x).copy() for x in ctrl.get_gains()]
    baseline_effort=np.asarray(ctrl.get_max_efforts()).copy()
    baseline_position_iterations=np.asarray(view.get_solver_position_iteration_counts()).copy()
    baseline_velocity_iterations=np.asarray(view.get_solver_velocity_iteration_counts()).copy()
    j=names.index("joint3")
    variants=[dict(name="baseline"),dict(name="P_half",p=.5),dict(name="P_double",p=2),
              dict(name="D_half",d=.5),dict(name="D_double",d=2),
              dict(name="effort_double",effort=2),dict(name="position_iterations_double",pos=2),
              dict(name="velocity_iterations_increase",vel=8),dict(name="baseline_repeat")]
    experiment = os.environ.get("APPLE_CONTROL_GROUP", "pd")
    if experiment == "pd": variants=[v for v in variants if v["name"] not in ("position_iterations_double","velocity_iterations_increase")]
    else: variants=[v for v in variants if v["name"] == experiment]
    jobs=[(v,label,target) for v in variants for label,target in poses.items()]
    metadata=dict(names=names,source_episode="episode_000044",source_first_close_frame=first_close,
                  baseline_p=baseline_p.tolist(),baseline_d=baseline_d.tolist(),baseline_max_effort=baseline_effort.tolist(),
                  baseline_position_iterations=baseline_position_iterations.tolist(),baseline_velocity_iterations=baseline_velocity_iterations.tolist(),
                  physics_dt=context.get_physics_dt(),poses={k:v.tolist() for k,v in poses.items()},variants=variants,
                  limits=np.asarray(view.get_dof_limits()).tolist(), protocol="1s home + 5s quintic ramp + 10s fixed hold; one factor at a time")
    with open(audit_root+"/metadata.json","w") as f:json.dump(metadata,f,indent=2)
    audit_file=open(audit_root+"/telemetry.jsonl","w",buffering=1)
    from isaacsim.core.utils.types import ArticulationAction
    audit_index=-1
    audit_frames=0
    audit_case_start=0.0
'''
s=s.replace('    training_objects = initialize_training_objects()',setup+'\n    training_objects = initialize_training_objects()',1)
s=s.replace("    training_objects = initialize_training_objects()\n    reset_interface = SceneResetInterface(\n        robot, training_objects, GRIPPER_STATOR\n    )", "    from types import SimpleNamespace\n    reset_interface = SimpleNamespace(spin_once=lambda:None, close=lambda:None)")
a=s.index('    try:\n        while simulation_app.is_running():',s.index('def main():'))
b=s.index('    except KeyboardInterrupt:',a)
s=s[:a]+'''    try:
        while simulation_app.is_running():
            job_index=int((context.current_time+1e-5)//16.0)
            if job_index>=len(jobs):break
            if job_index!=audit_index:
                audit_index=job_index
                variant,label,destination=jobs[audit_index]
                p=baseline_p.copy();d=baseline_d.copy();effort=baseline_effort.copy()
                p[j]*=variant.get("p",1);d[j]*=variant.get("d",1);effort[j]*=variant.get("effort",1)
                print("STAGE set_gains",flush=True)
                ctrl.set_gains(kps=p,kds=d);ctrl.set_max_efforts(effort)
                print("STAGE reset_pose",flush=True)
                pos=baseline_position_iterations*variant.get("pos",1)
                vel=np.maximum(baseline_velocity_iterations,variant.get("vel",0))
                robot.set_joint_positions(home)
                robot.set_joint_velocities(np.zeros_like(home))
                print("STAGE readback",flush=True)
                audit_case_start=context.current_time
                readback=dict(case=audit_index,variant=variant["name"],pose=label,
                              p=[np.asarray(x).tolist() for x in ctrl.get_gains()],effort=np.asarray(ctrl.get_max_efforts()).tolist(),
                              pos=np.asarray(view.get_solver_position_iteration_counts()).tolist(),vel=np.asarray(view.get_solver_velocity_iteration_counts()).tolist())
                with open(audit_root+"/readback.jsonl","a") as f:f.write(json.dumps(readback)+"\\n")
                print("CONTROL_CASE "+json.dumps(readback),flush=True)
            elapsed=context.current_time-audit_case_start
            if elapsed<1:phase="home";target=home
            elif elapsed<6:
                phase="ramp";u=min(1.,max(0.,(elapsed-1)/5));blend=10*u**3-15*u**4+6*u**5
                target=home+(destination-home)*blend
            else:phase="hold";target=destination
            ctrl.apply_action(ArticulationAction(joint_positions=target))
            context.step(render=(audit_frames%4==0))
            audit_frames+=1
            reset_interface.spin_once()
            audit_file.write(json.dumps(dict(case=audit_index,variant=variant["name"],pose=label,t=context.current_time,
                 elapsed=elapsed,phase=phase,target=target.tolist(),q=np.asarray(robot.get_joint_positions()).tolist(),
                 qd=np.asarray(robot.get_joint_velocities()).tolist(),measured_effort=np.asarray(robot.get_measured_joint_efforts()).tolist()))+"\\n")
        audit_file.close()
        with open(audit_root+"/completed.json","w") as f:json.dump(dict(completed_jobs=len(jobs),simulation_time=context.current_time),f)
'''+s[b:]
s=s.replace('    except KeyboardInterrupt:', '    except Exception:\n        import traceback\n        traceback.print_exc()\n        raise\n    except KeyboardInterrupt:')
exec(compile(s,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
