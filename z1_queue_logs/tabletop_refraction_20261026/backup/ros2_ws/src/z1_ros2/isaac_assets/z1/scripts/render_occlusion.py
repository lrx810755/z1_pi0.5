"""Keep close-range glass behind opaque robot geometry in RTX real-time RGB."""


CLOSE_RANGE_REFRACTION_WORLD_EPS = 0.0001


def configure_close_range_refraction(settings):
    """Avoid image-based refraction reprojection at centimetre-scale grasps.

    RTX's default worldEps=1.0 can composite a hidden glass cup over an opaque
    finger even though depth and semantic AOVs correctly see the finger.
    A 0.1 mm threshold retains glass materials and real-time rendering while
    tracing refraction accurately at the task's metre-based scene scale.
    Call after the first warm-up render, when RTX has initialized its pass.
    """
    key = "/rtx/translucency/worldEps"
    settings.set_float(key, CLOSE_RANGE_REFRACTION_WORLD_EPS)
    return {key: settings.get(key)}
