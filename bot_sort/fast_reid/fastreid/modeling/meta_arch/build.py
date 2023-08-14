# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch

from fast_reid.fastreid.utils.registry import Registry

from torchvision import models
from torchsummary import summary

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
   
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    print(f' Device in the build_model = {torch.device(cfg.MODEL.DEVICE)}')
    model.to(torch.device(cfg.MODEL.DEVICE))
    model.eval()
     
    
    return model
