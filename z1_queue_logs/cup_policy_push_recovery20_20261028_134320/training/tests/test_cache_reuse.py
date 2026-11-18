import tempfile,unittest
from pathlib import Path
import build_cache

class CacheReuse(unittest.TestCase):
 def test_changed_source_image_requires_matching_original_digest(self):
  self.assertTrue(hasattr(build_cache,'verify_source_image'),'Raw provenance needs unchanged-source verification')
  with tempfile.TemporaryDirectory() as temp:
   image=Path(temp)/'image.jpg';image.write_bytes(b'original')
   digest=build_cache.sha(image);build_cache.verify_source_image(image,digest,0)
   image.write_bytes(b'changed')
   with self.assertRaises(ValueError):build_cache.verify_source_image(image,digest,0)
