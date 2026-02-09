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

class OfficeHome():
    dataset_dir = "OfficeHome"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10
        self.lab2cname={'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
        self.classnames = {'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'}
        self.site_domian = {'Art':0, 'Clipart':1, 'Product':2, 'Real World':3}
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



