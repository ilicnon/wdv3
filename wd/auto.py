def tpu_available():
    devices = jax.devices()
    for device in devices:
        if "TPU" in str(device).upper():
            return True
    return False


try:
    import jax

    if tpu_available():
        print("Using jax")
        from wd.jax import get_infer_batch
    else:
        print("Using timm")
        from wd.timm import get_infer_batch
except ImportError:
    print("Using timm")
    from wd.timm import get_infer_batch
