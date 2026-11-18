"""Pure geometry and acceptance rules; speed is a scaling ratio, not m/s."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Settings:
    speed_base: float=.08
    speed_multiplier: float=.8
    exterior_offset_m: float=.060
    retreat_candidates_m: tuple=(.050,.060,.080,.100)
    blend_radius_m: float=.003
    finger_floor_m: float=.003
    final_height_tolerance_m: float=.002
    final_orientation_tolerance_rad: float=.035
    maximum_axis_pitch_deg: float=7.
    @property
    def final_speed(self):return self.speed_base*self.speed_multiplier

def waypoints(final,object_position,cfg,approach_axis=None):
    final=np.asarray(final,float);toward=np.asarray(object_position,float)-final
    direction=toward.copy() if approach_axis is None else np.asarray(approach_axis,float).copy();direction[2]=0
    if np.dot(direction,toward)<0:direction=-direction
    if not np.isfinite(direction).all() or np.linalg.norm(direction)<.01:raise ValueError('Invalid horizontal entry axis')
    if cfg.exterior_offset_m<10*cfg.blend_radius_m:raise ValueError('Blend needs exterior clearance')
    return final-direction/np.linalg.norm(direction)*cfg.exterior_offset_m

def validate_timing(times,velocities):
    t=np.asarray(times,float);v=np.asarray(velocities,float)
    if len(t)<3 or v.shape!=(len(t),6) or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid trajectory times or velocities')
    if np.any(np.max(np.abs(v[1:-1]),axis=1)<1e-5):raise ValueError('Interior stop: no unblended fallback allowed')

def route_candidates(final,object_position,approach_axis,cfg):
    """Vary exterior descent corridors; keep final insertion along the tool axis."""
    from dataclasses import replace
    routes=[]
    for retreat in cfg.retreat_candidates_m:
        low=waypoints(final,object_position,replace(cfg,exterior_offset_m=retreat),approach_axis)
        direction=(np.asarray(final)-low)/retreat;lateral=np.array([-direction[1],direction[0],0.])
        for height in [.080,.120]:
            for azimuth in [0.,15.,-15.]:
                high=low+np.array([0.,0.,height])+lateral*retreat*np.tan(np.deg2rad(azimuth))
                routes.append(dict(high=high,low=low,retreat_m=retreat,height_m=height,azimuth_deg=azimuth))
    return routes
