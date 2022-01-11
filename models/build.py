# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

from .dat import DAT

def build_model(config):

    model_type = config.MODEL.TYPE
    if model_type == 'dat':
        model = DAT(**config.MODEL.DAT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
