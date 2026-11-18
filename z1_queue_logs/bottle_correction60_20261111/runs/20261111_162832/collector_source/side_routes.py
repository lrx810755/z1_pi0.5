"""Side-entry corridors preserve the requested error and physical collision checks."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def side_entry_routes(original):
    bad=np.asarray(original['bad_position'],float)
    q=original['bad_orientation']
    forward=Rotation.from_quat(q).apply([1.,0,0]);forward[2]=0
    forward/=np.linalg.norm(forward)
    for clearance in (.06,.10):
        route=copy.deepcopy(original)
        behind=bad-clearance*forward
        high=behind.copy();high[2]=max(bad[2]+.05,.14)
        route['strategy']='side_entry_'+str(clearance)
        route['preparation']=[
            dict(stage='prepare_safe_behind',position=high.tolist(),orientation=q,linear=False),
            dict(stage='prepare_lower_behind',position=behind.tolist(),orientation=q,linear=True),
            dict(stage='prepare_side_entry',position=bad.tolist(),orientation=q,linear=True)]
        # Return directly when no tilt was requested; avoid withdrawing the
        # fingers vertically through the cup rim. More complex recovery routes
        # remain available as fallbacks from the original route generator.
        if route['recovery']:
            final=copy.deepcopy(route['recovery'][-1])
            final['stage']='recover_direct';final['linear']=True
            route['recovery']=[final]
        yield route
