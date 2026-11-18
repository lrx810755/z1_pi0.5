"""Scoped300g cup and doubled cup/table friction; shared assets stay immutable."""
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent
CUP = '/World/Task/Objects/Cup'
MATERIAL = '/World/PhysicsMaterials/EvalCupFriction2'


def apply(stage):
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    cup = stage.GetPrimAtPath(CUP)
    assert cup.IsValid() and cup.HasAPI(UsdPhysics.RigidBodyAPI)
    UsdPhysics.MassAPI.Apply(cup).CreateMassAttr().Set(.30)
    material = UsdShade.Material.Define(stage, MATERIAL)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.40)
    api.CreateDynamicFrictionAttr(.30)
    api.CreateRestitutionAttr(0.)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set('min')
    physx.CreateRestitutionCombineModeAttr().Set('min')
    stack = [cup]
    bound = []
    while stack:
        prim = stack.pop()
        stack.extend(prim.GetChildren())
        if prim == cup or prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
            bound.append(str(prim.GetPath()))
    assert CUP+'/Collider' in bound
    print('EVAL CUP OVERRIDE: mass=.300kg; static=.400 dynamic=.300 min-combine; gripper and table materials unchanged', flush=True)


def material_values(stage, path):
    from pxr import UsdPhysics, UsdShade
    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose='physics')
    assert material, path
    prim = material.GetPrim()
    api = UsdPhysics.MaterialAPI(prim)
    combine = prim.GetAttribute('physxMaterial:frictionCombineMode')
    return dict(path=str(prim.GetPath()), static=float(api.GetStaticFrictionAttr().Get()),
                dynamic=float(api.GetDynamicFrictionAttr().Get()), restitution=float(api.GetRestitutionAttr().Get()),
                combine=combine.Get() if combine else None)


def verify(stage, objects, robot, expected_drives, gripper_path):
    from pxr import UsdPhysics
    authored_mass = float(UsdPhysics.MassAPI(stage.GetPrimAtPath(CUP)).GetMassAttr().Get())
    actual_mass = float(objects[CUP].get_mass())
    assert math.isclose(authored_mass,.30,abs_tol=1e-6) and math.isclose(actual_mass,.30,abs_tol=1e-6)
    cup = material_values(stage,CUP+'/Collider')
    table = material_values(stage,'/World/Task/WorkSurface')
    gripper = material_values(stage,gripper_path)
    assert math.isclose(cup['static'],.40,abs_tol=1e-6) and math.isclose(cup['dynamic'],.30,abs_tol=1e-6)
    assert cup['combine']=='min' and cup['restitution']==0.
    assert table['static']>=.40 and table['dynamic']>=.30 and table['combine'] in [None,'average','min']
    assert math.isclose(gripper['static'],6.,abs_tol=1e-6) and math.isclose(gripper['dynamic'],5.,abs_tol=1e-6)
    assert gripper['combine']=='max'
    controller = robot.get_articulation_controller()
    kp,kd = controller.get_gains()
    efforts = controller.get_max_efforts()
    drives = {}
    for i,name in enumerate(robot.dof_names):
        if name in expected_drives:
            actual = [float(kp[i]),float(kd[i]),float(efforts[i])]
            assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-5) for a,b in zip(actual,expected_drives[name]))
            drives[name] = actual
    evidence = dict(passed=True,authored_cup_mass_kg=authored_mass,actual_cup_mass_kg=actual_mass,
        cup_material=cup,table_material=table,gripper_material=gripper,
        effective_cup_table_friction=dict(static=.40,dynamic=.30),
        effective_cup_gripper_friction=dict(static=6.,dynamic=5.),
        drive_parameters_unchanged=True,drive_parameters=drives,
        mass_readback='initialized PhysX rigid-body view',target_collisions_enabled=True)
    path=R/'physics_verified.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(evidence,indent=2)+'\n');temp.replace(path)
    print('EVAL CUP PHYSICS VERIFIED: '+json.dumps(evidence),flush=True)

