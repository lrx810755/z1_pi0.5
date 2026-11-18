"""Private runner: outer HTTP wait covers10 complete layout searches."""
import z1_isaac_bridge.continuous_training as original
OriginalClient=original.JsonApiClient
class CompleteSearchClient(OriginalClient):
 def post(self,path,payload=None,timeout_s=30.):
  if path=='/api/demonstration/auto':timeout_s=max(timeout_s,21600.)
  return super().post(path,payload,timeout_s)
original.JsonApiClient=CompleteSearchClient
if __name__=='__main__':raise SystemExit(original.main())
