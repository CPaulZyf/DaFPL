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

class DomainNet:
    dataset_dir = "DomainNet"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10
        self.lab2cname = {'The_Eiffel_Tower': 0, 'The_Great_Wall_of_China': 1, 'The_Mona_Lisa': 2, 'aircraft_carrier': 3, 'airplane': 4, 'alarm_clock': 5, 'ambulance': 6, 'angel': 7, 'animal_migration': 8, 'ant': 9, 'anvil': 10, 'apple': 11, 'arm': 12, 'asparagus': 13, 'axe': 14, 'backpack': 15, 'banana': 16, 'bandage': 17, 'barn': 18, 'baseball': 19, 'baseball_bat': 20, 'basket': 21, 'basketball': 22, 'bat': 23, 'bathtub': 24, 'beach': 25, 'bear': 26, 'beard': 27, 'bed': 28, 'bee': 29, 'belt': 30, 'bench': 31, 'bicycle': 32, 'binoculars': 33, 'bird': 34, 'birthday_cake': 35, 'blackberry': 36, 'blueberry': 37, 'book': 38, 'boomerang': 39, 'bottlecap': 40, 'bowtie': 41, 'bracelet': 42, 'brain': 43, 'bread': 44, 'bridge': 45, 'broccoli': 46, 'broom': 47, 'bucket': 48, 'bulldozer': 49, 'bus': 50, 'bush': 51, 'butterfly': 52, 'cactus': 53, 'cake': 54, 'calculator': 55, 'calendar': 56, 'camel': 57, 'camera': 58, 'camouflage': 59, 'campfire': 60, 'candle': 61, 'cannon': 62, 'canoe': 63, 'car': 64, 'carrot': 65, 'castle': 66, 'cat': 67, 'ceiling_fan': 68, 'cell_phone': 69, 'cello': 70, 'chair': 71, 'chandelier': 72, 'church': 73, 'circle': 74, 'clarinet': 75, 'clock': 76, 'cloud': 77, 'coffee_cup': 78, 'compass': 79, 'computer': 80, 'cookie': 81, 'cooler': 82, 'couch': 83, 'cow': 84, 'crab': 85, 'crayon': 86, 'crocodile': 87, 'crown': 88, 'cruise_ship': 89, 'cup': 90, 'diamond': 91, 'dishwasher': 92, 'diving_board': 93, 'dog': 94, 'dolphin': 95, 'donut': 96, 'door': 97, 'dragon': 98, 'dresser': 99}

        self.classnames = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser']
        self.site_domian = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
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



