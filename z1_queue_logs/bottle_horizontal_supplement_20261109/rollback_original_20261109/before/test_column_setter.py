import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace as N
from set_collection_column import bounded_call,set_column,ParameterType
class Tests(unittest.TestCase):
 def test_timeout_cancels_request(self):
  f=Mock();f.done.return_value=False;c=Mock();c.wait_for_service.return_value=True;c.call_async.return_value=f
  with patch('set_collection_column.rclpy.spin_until_future_complete'):
   with self.assertRaisesRegex(RuntimeError,'超时'):bounded_call(Mock(),c,Mock(),timeout=.1)
  f.cancel.assert_called_once();f.result.assert_not_called()
 def test_missing_service_does_not_send(self):
  c=Mock();c.wait_for_service.return_value=False
  with self.assertRaises(RuntimeError):bounded_call(Mock(),c,Mock())
  c.call_async.assert_not_called()
 def test_mismatched_readback_rejected(self):
  replies=[N(results=[N(successful=True)]),N(values=[N(type=ParameterType.PARAMETER_INTEGER,integer_value=2)])]
  with patch('set_collection_column.bounded_call',side_effect=replies):
   with self.assertRaisesRegex(RuntimeError,'回读'):set_column(Mock(),1)
 def test_correct_readback_accepted(self):
  replies=[N(results=[N(successful=True)]),N(values=[N(type=ParameterType.PARAMETER_INTEGER,integer_value=1)])]
  with patch('set_collection_column.bounded_call',side_effect=replies):set_column(Mock(),1)
if __name__=='__main__':unittest.main()
