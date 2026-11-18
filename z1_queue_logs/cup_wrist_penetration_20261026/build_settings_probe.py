from pathlib import Path
r=Path(__file__).parent;s=(r/'depth_probe.py').read_text().replace("R=G/'depth_probe'","R=G/'settings_probe'")
needle=" capture('original')\n"
insert=""" capture('original')
 import carb
 settings=carb.settings.get_settings();keys=['/rtx/translucency/worldEps','/rtx/raytracing/fractionalCutoutOpacity','/rtx/material/translucencyAsOpacity','/rtx/translucency/virtualDepth','/rtx/rendermode'];values={k:settings.get(k) for k in keys};print('ORIGINAL_SETTINGS',values,flush=True);(R/'original_settings.json').write_text(json.dumps(values,indent=2))
 for name,key,value in [('world_eps_0p0001','/rtx/translucency/worldEps',.0001),('world_eps_zero','/rtx/translucency/worldEps',0.),('fractional_cutout','/rtx/raytracing/fractionalCutoutOpacity',True),('translucency_as_opacity','/rtx/material/translucencyAsOpacity',True),('virtual_depth','/rtx/translucency/virtualDepth',True)]:
  old=settings.get(key);settings.set(key,value);capture(name)
  if old is not None:settings.set(key,old)
  else:settings.destroy_item(key)
 settings.set('/rtx/rendermode','PathTracing');settings.set('/rtx/pathtracing/spp',32);settings.set('/rtx/pathtracing/totalSpp',32);capture('original_path_tracing');settings.set('/rtx/rendermode','RaytracedLighting')
"""
assert needle in s;s=s.replace(needle,insert);(r/'settings_probe.py').write_text(s)
