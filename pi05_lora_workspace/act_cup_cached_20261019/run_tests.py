import unittest
import test_baseline,test_data
for name in sorted(dir(test_baseline)):
 if name.startswith('test_'):
  getattr(test_baseline,name)();print(name,'PASS',flush=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_data))
if not result.wasSuccessful():raise SystemExit(1)

from prepare import write
write('recipe_verified.json',dict(passed=True,bottle_baseline=str(test_baseline.BOTTLE),
    optimizer_architecture_loader_seed_equal=True,sampling_rule_equal=True,
    original_mass=.7,correction_mass=.3,key_weight=2,other_weight=1,
    close_first_frame_extra_weight=False,chunk_size=16,optimizer_updates=50000))
