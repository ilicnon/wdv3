from wd.models.convnext import convnext_base, convnext_small, convnext_tiny
from wd.models.swinv2 import swinv2_base, swinv2_large, swinv2_tiny
from wd.models.vit import vit_base, vit_large, vit_small

model_registry = {
    "swinv2_tiny": swinv2_tiny,
    "swinv2_base": swinv2_base,
    "swinv2_large": swinv2_large,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
}
