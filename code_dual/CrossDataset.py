import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from functions import get_uir

class CrossDataset(object):
    def __init__(self, data_path):
        '''
        Constructor
        '''
        with np.load(data_path) as data:
            self.target_train_m = data['target_train_m']
            self.target_train = data['target_train']
            self.target_vali = data['target_vali']
            self.target_test  = data['target_test']
            self.source_train_m  = data['source_train_m']
            self.source_train  = data['source_train']
            self.source_vali  = data['source_vali']
            self.source_test  = data['source_test']

    def get_train_dataloader(self, batch_size):        
        target_user, target_item, target_rating = get_uir(self.target_train, self.target_train_m)
        source_ind_rand = np.random.randint(len(self.source_train), size=(self.target_train.shape[0],))
        source_train_sample = self.source_train[source_ind_rand]
        source_user = self.source_train_m[source_train_sample[:,0].astype(int),:]
        source_item = self.source_train_m[:, source_train_sample[:,1].astype(int)].T
        source_rating = source_train_sample[:,2]
        source_user, source_item, source_rating = torch.FloatTensor(source_user), torch.FloatTensor(source_item), torch.FloatTensor(source_rating)

        train_data = TensorDataset(source_user, source_item, source_rating, target_user, target_item, target_rating)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader

    def get_test_dataloader(self, batch_size, mode = 'test', data_source = 'target'):        
        if mode == 'validation':
            if data_source == 'source':
                user, item, rating = get_uir(self.source_vali, self.source_train_m)
            elif data_source == 'target':
                user, item, rating = get_uir(self.target_vali, self.target_train_m)
        elif mode == 'test':
            if data_source == 'source':
                user, item, rating = get_uir(self.source_test, self.source_train_m)
            elif data_source == 'target':
                user, item, rating = get_uir(self.target_test, self.target_train_m)

        test_data = TensorDataset(user, item, rating)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return test_loader

