import unittest
from unittest.mock import Mock
from combined_transport import CombinedBottleTransport
class Parent:
 def _open_gripper_for_box_release(self,target):self.opened=True
class Node(CombinedBottleTransport,Parent):
 def __init__(self,z):self.z=z;self.box_center=(.32,.14);self.opened=False
 def _current_object_position(self,*_):return [.32,.14,self.z]
class ReleaseRange(unittest.TestCase):
 def test_both_range_boundaries_allow_open(self):
  for z in [.115,.1225,.130]:
   n=Node(z);n._open_gripper_for_box_release('beverage_bottle');self.assertTrue(n.opened)
 def test_outside_range_never_opens(self):
  for z in [.1149,.1301,.18]:
   n=Node(z)
   with self.assertRaisesRegex(RuntimeError,'释放高度'):n._open_gripper_for_box_release('beverage_bottle')
   self.assertFalse(n.opened)
