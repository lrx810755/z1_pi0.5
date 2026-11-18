import json,tempfile,unittest
from pathlib import Path
from training_gate import training_ready
class TrainingGateTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.r=Path(self.temp.name);self.c=self.r/'checkpoints/240000/pretrained_model';self.c.mkdir(parents=True)
 def tearDown(self):self.temp.cleanup()
 def write(self,name,obj):(self.r/name).write_text(json.dumps(obj))
 def complete(self):
  self.write('workflow_status.json',{'stage':'completed','optimizer_updates':30000,'microsteps':240000})
  self.write('completed.json',{'optimizer_updates':30000,'microsteps':240000})
  self.write('final_checkpoint_reload_verified.json',{'passed':True,'checkpoint':str(self.c),'tests':[{'prediction_shape':[1,50,7],'finite':True}]})
  (self.c/'config.json').write_text(json.dumps({'chunk_size':50}));(self.c/'adapter_model.safetensors').write_bytes(b'weight')
 def test_waits_while_training(self):
  self.write('workflow_status.json',{'stage':'training'});self.assertFalse(training_ready(self.r,self.c,False))
 def test_waits_until_training_service_exits(self):
  self.complete();self.assertFalse(training_ready(self.r,self.c,True))
 def test_accepts_only_completed_final_model(self):
  self.complete();self.assertTrue(training_ready(self.r,self.c,False))
 def test_rejects_wrong_horizon(self):
  self.complete();(self.c/'config.json').write_text(json.dumps({'chunk_size':30}))
  with self.assertRaises(ValueError):training_ready(self.r,self.c,False)
 def test_rejects_unverified_final_checkpoint(self):
  self.complete();self.write('final_checkpoint_reload_verified.json',{'passed':False})
  with self.assertRaises(ValueError):training_ready(self.r,self.c,False)
if __name__=='__main__':unittest.main()
