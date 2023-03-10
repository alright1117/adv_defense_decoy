import torch

from .XDeception import XDeception
from .Xception import return_Xception
from .MesoDeception import MesoDeception
from .Meso import MesoInception4

def load_model(model_name, pretrained_weight, logger):

    if model_name == "Xception":
        logger.info('Model Xception.')
        model = return_Xception()
        input_size = 299
    elif model_name == "XDeception":
        logger.info('Model XDeception.')
        model = XDeception()
        input_size = 299
    elif model_name == "Meso":
        logger.info('Model MesoInception.')
        model = MesoInception4()
        input_size = 256
    elif model_name == "MesoDeception":
        logger.info('Model MesoDeception.')
        model = MesoDeception()
        input_size = 256
    else:
        raise ValueError('Wrong model name.')

    if pretrained_weight:
        model.load_state_dict(torch.load(pretrained_weight, map_location="cpu"))

    return model, input_size