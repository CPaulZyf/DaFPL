from Dassl.dassl.utils import Registry, check_availability

from datasets.domainnet import DomainNet
from datasets.office import Office
from datasets.pacs import PACS
from datasets.officehome import OfficeHome


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.register(DomainNet)
DATASET_REGISTRY.register(Office)
DATASET_REGISTRY.register(OfficeHome)
DATASET_REGISTRY.register(PACS)

print(DATASET_REGISTRY.registered_names())
def build_dataset(cfg):
    print(DATASET_REGISTRY.registered_names())
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
