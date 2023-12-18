import random
from collections import deque
import numpy as np
import torch

class Store:
    def __init__(self, total_num_classes, items_per_class, items_per_adding=5, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.items_per_adding = items_per_adding
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        new_items = []
        new_class_ids = []
        for class_id in range(self.total_num_classes):
            _items = items[class_ids == class_id]
            _class_ids = class_ids[class_ids == class_id]

            samples = torch.randperm(len(_items))[:min(self.items_per_class, len(_items))]

            new_items.append(_items[samples])
            new_class_ids.append(_class_ids[samples])

        items = torch.cat(new_items)
        class_ids = torch.cat(new_class_ids)

        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])
