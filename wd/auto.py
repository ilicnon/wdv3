def tpu_available():
    devices = jax.devices()
    for device in devices:
        if "TPU" in str(device).upper():
            return True
    return False


try:
    import jax

    if tpu_available():
        from wd.jax import get_infer_batch
    else:
        from wd.timm import get_infer_batch
except ImportError:
    from wd.timm import get_infer_batch
