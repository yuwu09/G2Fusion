import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]

    if model == 'Matcher':
        from .match_model import match_model as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
