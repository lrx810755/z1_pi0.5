"""Experimental scene wrapper; production source stays unchanged."""
from pathlib import Path
import sys

SOURCE=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')

METHOD='''
    def _comparison_command(self, command):
        from protocol import validate_profile
        if not hasattr(self, '_comparison_pub'):
            self._comparison_pub = self._node.create_publisher(self._String, '/smolvla/grasp_comparison_status', 10)
        reply = {'request_id': command.get('request_id')}
        try:
            stage = omni.usd.get_context().get_stage()
            material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/PhysicsMaterials/SmolVLAGripper'))
            controller = self._robot.get_articulation_controller()
            index = list(self._robot.dof_names).index('jointGripper')
            efforts = np.asarray(controller.get_max_efforts()).copy()
            kps, kds = controller.get_gains()
            if not hasattr(self, '_comparison_original'):
                self._comparison_original = {'static': float(material.GetStaticFrictionAttr().Get()),
                    'dynamic': float(material.GetDynamicFrictionAttr().Get()), 'effort': float(efforts[index]),
                    'stiffness': float(kps[index]), 'damping': float(kds[index])}
            mode = command.get('mode', 'set')
            if mode == 'probe':
                values = self._comparison_original.copy()
            else:
                values = self._comparison_original.copy() if mode == 'restore' else dict(command['profile'])
                validate_profile(values['static'], values['dynamic'], values['effort'])
                material.GetStaticFrictionAttr().Set(float(values['static']))
                material.GetDynamicFrictionAttr().Set(float(values['dynamic']))
                efforts[index] = values['effort']
                controller.set_max_efforts(efforts)
                if mode == 'restore':
                    kps[index], kds[index] = values['stiffness'], values['damping']
                    controller.set_gains(kps=kps, kds=kds)
            actual = {'static': float(material.GetStaticFrictionAttr().Get()),
                'dynamic': float(material.GetDynamicFrictionAttr().Get()),
                'effort': float(controller.get_max_efforts()[index]),
                'stiffness': float(controller.get_gains()[0][index]),
                'damping': float(controller.get_gains()[1][index])}
            for key in ('static','dynamic','effort'):
                if abs(actual[key]-values[key]) > 1e-4:
                    raise RuntimeError('parameter readback mismatch: '+key)
            reply.update(ok=True, actual=actual, original=self._comparison_original)
        except Exception as exc:
            reply.update(ok=False, error=str(exc))
        message = self._String(); message.data = json.dumps(reply)
        self._comparison_pub.publish(message)
'''

def patched_source():
    source=SOURCE.read_text()
    anchor='    def _grasp_command_callback(self, message):\n'
    branch='            if kind == "freeze":\n'
    if source.count(anchor)!=1 or source.count(branch)!=1:
        raise RuntimeError('scene source changed; experimental wrapper needs review')
    source=source.replace(anchor,METHOD+'\n'+anchor)
    source=source.replace(branch,'            if kind == "comparison_parameters":\n                self._comparison_command(command)\n                return\n            elif kind == "freeze":\n')
    return source

if __name__=='__main__':
    sys.path.insert(0,str(SOURCE.parent))
    # Keep imports and __file__ relative to the original scene implementation.
    namespace={'__name__':'__main__','__file__':str(SOURCE),'__package__':None}
    exec(compile(patched_source(),str(SOURCE),'exec'),namespace)
