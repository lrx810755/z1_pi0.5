"""Keep all tabletop glass behind opaque geometry in RTX real-time RGB."""


CLOSE_RANGE_REFRACTION_WORLD_EPS = 0.0001


def author_close_range_refraction(stage):
    """Persist the all-object setting in Kit's serialized render-settings metadata.

    Scene builders and saved scenes need this in addition to the runtime call:
    RTX can overwrite its settings while initializing the first render pass.
    Preserve all other task metadata and render settings; do not edit materials.
    The caller owns saving the root layer.
    """
    layer = stage.GetRootLayer()
    metadata = dict(layer.customLayerData)
    render_settings = dict(metadata.get("renderSettings", {}))
    render_settings["rtx:translucency:worldEps"] = CLOSE_RANGE_REFRACTION_WORLD_EPS
    metadata["renderSettings"] = render_settings
    layer.customLayerData = metadata


def configure_close_range_refraction(settings):
    """Avoid image-based refraction reprojection at centimetre-scale grasps.

    RTX's default worldEps=1.0 can composite hidden glass (cups, bottles, liquid,
    or condensation) over opaque fingers despite correct depth/semantic AOVs.
    A 0.1 mm threshold retains glass materials and real-time rendering while
    tracing refraction accurately at the task's metre-based scene scale.
    Call after the first warm-up render, when RTX has initialized its pass.
    """
    key = "/rtx/translucency/worldEps"
    settings.set_float(key, CLOSE_RANGE_REFRACTION_WORLD_EPS)
    return {key: settings.get(key)}
