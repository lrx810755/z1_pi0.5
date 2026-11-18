import sys,pathlib
source=pathlib.Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
sys.path.insert(0,str(source.parent))
s=source.read_text()
setup='''
    audit_root = "/home/lp531/z1_queue_logs/apple_pd_audit_20261016"
    audit_rows = [json.loads(x) for x in open("/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016/episode_000044/frames.jsonl")]
    first_close = next(i for i,r in enumerate(audit_rows) if r["action"][-1] > -0.895)
    a = audit_rows[max(0,first_close-2)]["action"]
    b = next(r["action"] for r in audit_rows[first_close:] if r["phase"] == "TRANSPORT")
    names = list(robot.dof_names)
    joint_order = ["joint1","joint2","joint3","joint4","joint5","joint6","jointGripper"]
    home = [SAFE_HOME_BY_JOINT[n] for n in names]
    def ordered(x): return [x[joint_order.index(n)] for n in names]
    audit_a, audit_b = ordered(a), ordered(b)
    audit_a[names.index("jointGripper")] = -0.9
    audit_b[names.index("jointGripper")] = -0.9
    gain_readback = robot.get_articulation_controller().get_gains()
    prim_info = {}
    for prim in stage.Traverse():
        if prim.GetName() in names:
            prim_info[str(prim.GetPath())] = {attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if any(k in attr.GetName().lower() for k in ("limit","drive","friction"))}
    with open(audit_root+"/runtime_parameters.json","w") as f:
        json.dump(dict(names=names, gains=[np.asarray(x).tolist() for x in gain_readback],max_efforts=np.asarray(robot.get_articulation_controller().get_max_efforts()).tolist(), limits=np.asarray(robot._articulation_view.get_dof_limits()).tolist(), usd=prim_info, targets=dict(home=home,preclose=audit_a,transport=audit_b)),f,indent=2)
    audit_file = open(audit_root+"/telemetry.jsonl","w",buffering=1)
    from isaacsim.core.utils.types import ArticulationAction
'''
s=s.replace('    training_objects = initialize_training_objects()',setup+'\n    training_objects = initialize_training_objects()',1)
needle='            context.step(render=True)'
inject='''            t = context.current_time
            if t < 3: phase="home"; target=home
            elif t < 8: phase="ramp_preclose"; f=(t-3)/5; target=[x+(y-x)*f for x,y in zip(home,audit_a)]
            elif t < 18: phase="hold_preclose"; target=audit_a
            elif t < 23: phase="ramp_transport"; f=(t-18)/5; target=[x+(y-x)*f for x,y in zip(audit_a,audit_b)]
            else: phase="hold_transport"; target=audit_b
            robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(target)))
            context.step(render=True)
            audit_file.write(json.dumps(dict(t=t,phase=phase,target=target,q=np.asarray(robot.get_joint_positions()).tolist(),qd=np.asarray(robot.get_joint_velocities()).tolist(), gains=[np.asarray(x).tolist() for x in robot.get_articulation_controller().get_gains()],measured_efforts=np.asarray(robot.get_measured_joint_efforts()).tolist()))+"\\n")
            if t > 34: break'''
s=s.replace(needle,inject,1)
exec(compile(s,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
