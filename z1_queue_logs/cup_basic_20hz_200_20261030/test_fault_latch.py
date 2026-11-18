import tempfile,unittest,json
from pathlib import Path
from fault_latch import latch_fault,check_fault
class FaultTests(unittest.TestCase):
 def test_latched_fault_survives_later_clear_or_error(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fault.json';check_fault(p);latch_fault(p,'camera missing',3);latch_fault(p,'',4);latch_fault(p,'later physical failure',4)
   self.assertEqual(json.loads(p.read_text())['reason'],'camera missing')
   with self.assertRaises(RuntimeError):check_fault(p)
if __name__=='__main__':unittest.main()
