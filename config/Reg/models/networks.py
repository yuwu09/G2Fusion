import logging

import torch

from models import modules as M

logger = logging.getLogger("base")

# Generator
def define_Match(opt):
    opt_net = opt["network_Match"]
    which_model = opt_net["which_model_Match"]
    setting = opt_net["setting"]
    net_match = getattr(M, which_model)(**setting)
    return net_match

def define_dino(opt):
    opt_net = opt["network_dino"]
    which_model = opt_net["which_model_dino"]
    setting = opt_net["setting"]
    net_dino = getattr(M, which_model)(**setting)
    return net_dino
