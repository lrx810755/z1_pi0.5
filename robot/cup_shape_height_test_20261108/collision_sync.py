"""Private MoveIt collider adapter; all ACM and synchronization behavior retained."""
import os
from experiment_core import geometry
from z1_isaac_bridge import object_collision_sync as m
shape=os.environ['CUP_EXPERIMENT_SHAPE'];spec=geometry(shape)
original=m.ObjectCollisionSync._world_collision

def world(self,name,position,orientation):
 if name!='cup' or shape=='frustum':return original(self,name,position,orientation)
 collision=m.CollisionObject();collision.header.frame_id=self.world_frame;collision.id=self._object_id(name)
 primitive=m.SolidPrimitive();primitive.type=m.SolidPrimitive.CYLINDER;primitive.dimensions=[spec['height_m'],spec['radius_m']]
 offset=m.rotate_vector(orientation,(0.,0.,spec['height_m']/2));pose=m.Pose()
 pose.position.x=position[0]+offset[0];pose.position.y=position[1]+offset[1];pose.position.z=position[2]+offset[2]
 pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=orientation
 collision.primitives=[primitive];collision.primitive_poses=[pose];collision.operation=m.CollisionObject.ADD
 return collision
m.ObjectCollisionSync._world_collision=world
if __name__=='__main__':m.main()
