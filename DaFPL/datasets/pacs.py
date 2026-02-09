import os
import pickle
import random
from collections import defaultdict
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum

def split_list(input_list, ratios):

    total = sum(ratios)
    sizes = [int(len(input_list) * ratio / total) for ratio in ratios]


    shuffled = input_list[:]
    random.shuffle(shuffled)

    result = []
    start = 0
    for size in sizes:
        result.append(shuffled[start:start + size])
        start += size


    if start < len(shuffled):
        result[0].extend(shuffled[start:])

    return result

class PACS():
    dataset_dir = "PACS"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10
        self.lab2cname={'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        self.classnames ={'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'}
        self.site_domian = {'art_painting':0, 'cartoon':1, 'photo':2, 'sketch':3}
        self.federated_train_x = []
        self.federated_test_x = []
        self.clientid2domain = defaultdict(list)

        for domain, _ in self.site_domian.items():
            image_dir = os.path.join(self.dataset_dir, domain)
            file_name = image_dir + '_' + str(cfg.DATASET.DOMAINUSERS) + 'client_' + cfg.DATASET.PARTITION + '.pkl'
            with open(file_name, "rb") as file:
                result, traindata_cls_counts, data_distributions = pickle.load(file)
            data_user = defaultdict(list)
            for i in range(len(result)):
                for sample in result[i]:
                    s = Datum(impath=sample[0], label=sample[1], domain=sample[2], classname=sample[3])
                    data_user[i].append(s)

            for i in range(len(data_user)):
                data = data_user[i]
                ratios = [6, 2, 2]
                train, val, test =  split_list(data, ratios)
                self.federated_train_x.append(train)
                self.federated_test_x.append(val + test)
                self.clientid2domain[len(self.federated_train_x) - 1] = self.site_domian[domain]

        # self.federated_train_x = train_set
        # self.federated_test_x = test_set
        # self.lab2cname = lab2cname
        # self.classnames = classnames



