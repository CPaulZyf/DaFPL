from Dassl.dassl.utils import Registry, check_availability
from trainers.dafpl import PromptFL, DaFPL

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(DaFPL)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    # print("avai_trainers",avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
