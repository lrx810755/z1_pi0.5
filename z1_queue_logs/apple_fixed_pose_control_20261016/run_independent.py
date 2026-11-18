import sys,pathlib,os
original=pathlib.Path('/home/lp531/z1_queue_logs/apple_pd_audit_20261016/diagnose.py').read_text()
# This is the earlier working single-run script, with one fixed control change.
group=os.environ['APPLE_CONTROL_GROUP']
root='/home/lp531/z1_queue_logs/apple_fixed_pose_control_20261016/'+group
pathlib.Path(root).mkdir(exist_ok=True)
original=original.replace('/home/lp531/z1_queue_logs/apple_pd_audit_20261016',root)
original=original.replace('s=source.read_text()', '''s=source.read_text()
group=os.environ['APPLE_CONTROL_GROUP']
if group in ('position_iterations_double','velocity_iterations_increase'):
    setting="api.CreateSolverPositionIterationCountAttr().Set(64)" if group=='position_iterations_double' else "api.CreateSolverVelocityIterationCountAttr().Set(8)"
    s=s.replace('    context.initialize_physics()', "    api=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(articulation_root))\\n    "+setting+"\\n    context.initialize_physics()")
change={'P_half':(0,.5),'P_double':(0,2),'D_half':(1,.5),'D_double':(1,2),'effort_double':(2,2)}.get(group)
if change:
    index,factor=change
    s=s.replace('    robot = configure_drive_gains(articulation_root)', "    adjusted=list(DRIVE_GAINS['joint3'])\\n    adjusted["+str(index)+"]*="+str(factor)+"\\n    DRIVE_GAINS['joint3']=tuple(adjusted)\\n    robot = configure_drive_gains(articulation_root)")
''')
# Capture solver readback in addition to gains/limits.
original=original.replace('limits=np.asarray(robot._articulation_view.get_dof_limits()).tolist(),', 'limits=np.asarray(robot._articulation_view.get_dof_limits()).tolist(), solver_position_iterations=np.asarray(robot._articulation_view.get_solver_position_iteration_counts()).tolist(), solver_velocity_iterations=np.asarray(robot._articulation_view.get_solver_velocity_iteration_counts()).tolist(),')
original=original.replace('elif t < 8: phase="ramp_preclose"; f=(t-3)/5','elif t < 6: phase="ramp_preclose"; f=(t-3)/3')
original=original.replace('elif t < 18: phase="hold_preclose"','elif t < 14: phase="hold_preclose"')
original=original.replace('elif t < 23: phase="ramp_transport"; f=(t-18)/5','elif t < 17: phase="ramp_transport"; f=(t-14)/3')
original=original.replace('if t > 34: break','if t > 26: break')
# The control experiment requires no reset services or ROS command clients.
original=original.replace("s=s.replace('    training_objects = initialize_training_objects()'", "s=s.replace('    training_objects = initialize_training_objects()'")
needle="exec(compile(s,str(source),'exec')"
original=original.replace(needle,"s=s.replace('    training_objects = initialize_training_objects()\\n    reset_interface = SceneResetInterface(\\n        robot, training_objects, GRIPPER_STATOR\\n    )','    from types import SimpleNamespace\\n    reset_interface=SimpleNamespace(spin_once=lambda:None,update_freeze_state=lambda:None,update_contact_state=lambda:None,close=lambda:None)')\n"+needle)
exec(compile(original,__file__,'exec'))
