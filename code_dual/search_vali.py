import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/qzhang7/Data/Documents/python/dual/')

from CrossDataset import CrossDataset
from Model import Dual
from functions import LossMSE, LossOrth


model_root = 'model'
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCH = 4
source_predict_scheme = 'MF'

lr = 1e-4
batch_size = 128
weight_decay = 0

num_dim_user_1 = 40
num_dim_item_1 = 40
num_dim_user_2 = 10
num_dim_item_2 = 10
num_dim_hidden = 10
alpha = 1
beta = 0.1
gamma = 1
active_domain_loss_step = 1

data_path = '/home/qzhang7/Data/Documents/python/dual/data/douban_movie2book_same.npz'
save_path = './result_batch'
code_name = '/douban_movie2book_same_tt'

if not os.path.exists(save_path):
    os.makedirs(save_path)
dataset = CrossDataset(data_path)
num_user_source, num_item_source = dataset.source_train_m.shape
num_user_target, num_item_target = dataset.target_train_m.shape

dual_net = Dual(num_user_source, num_item_source, num_user_target, num_item_target, \
        num_dim_user_1, num_dim_item_1, num_dim_user_2, num_dim_item_2, num_dim_hidden).to(DEVICE)
optimizer = torch.optim.Adam([{'params':dual_net.parameters(),'lr':lr},], weight_decay=weight_decay)

train_loader = dataset.get_train_dataloader(batch_size)
vali_target_loader = dataset.get_test_dataloader(batch_size, mode = 'validation', data_source = 'target')
test_target_loader = dataset.get_test_dataloader(batch_size, mode = 'test', data_source = 'target')
del(dataset)
num_batch = len(train_loader)
dann_epoch = np.floor(active_domain_loss_step / num_batch * 1.0)

loss_recon = LossMSE().to(DEVICE)
loss_diff = LossOrth().to(DEVICE)
loss_adver = nn.CrossEntropyLoss().to(DEVICE)

rmse_target_train = []
rmse_target_vali = []
rmse_target_test = []
cc = 0
for epoch in range(EPOCH):                                                                                                                 
    batch_step = 0
    rmse_train_epoch = 0    
    dual_net.train()
    for batch_ind, (source_user, source_item, source_rating, target_user, target_item, target_rating) in enumerate(train_loader):
        source_user, source_item, source_rating, target_user, target_item, target_rating = source_user.to(DEVICE), source_item.to(DEVICE), \
            source_rating.to(DEVICE), target_user.to(DEVICE), target_item.to(DEVICE), target_rating.to(DEVICE)
        dual_net.zero_grad()
        loss = 0
        # if batch_step > active_domain_loss_step:
        p = float(batch_ind + (epoch - dann_epoch) * num_batch / (EPOCH - dann_epoch) / num_batch)
        p = 2. / (1. + np.exp(-10 * p)) - 1
        embeds_source, results_source = dual_net(source_user, source_item, p, source_scheme = 'source', rec_scheme = 'all', predict_scheme = source_predict_scheme)
        embeds_target, results_target = dual_net(target_user, target_item, p, source_scheme = 'target', rec_scheme = 'all', predict_scheme = 'MF')
        [source_user_embeds_p, source_item_embeds_p, source_user_embeds_s, source_item_embeds_s] = embeds_source
        [source_user_label, source_item_label, source_predict] = results_source
        [target_user_embeds_p, target_item_embeds_p, target_user_embeds_s, target_item_embeds_s] = embeds_target
        [target_user_label, target_item_label, target_predict] = results_target
        # Rating prediction loss
        if source_predict_scheme == 'recon':
            loss_source = 0.5 * (loss_recon(source_predict[0], source_user) + loss_recon(source_predict[1], source_item))
        elif source_predict_scheme == 'MF':
            loss_source = loss_recon(source_predict, source_rating)
        loss_target = loss_recon(target_predict, target_rating)
        rmse_train_epoch += loss_target.item()
        loss += (loss_source * alpha + loss_target)
        # Orthorganal loss between private coding and sharing coding
        loss_source_user_diff = loss_diff(source_user_embeds_p, source_user_embeds_s)
        loss_source_item_diff = loss_diff(source_item_embeds_p, source_item_embeds_s)
        loss_target_user_diff = loss_diff(target_user_embeds_p, target_user_embeds_s)
        loss_target_item_diff = loss_diff(target_item_embeds_p, target_item_embeds_s)
        loss += beta * (loss_source_user_diff + loss_source_item_diff + loss_target_user_diff + loss_target_item_diff) 
        # Classification loss during adversarial training
        if batch_step > active_domain_loss_step:
            source_true_label = torch.zeros(len(source_user_label), dtype = torch.long).to(DEVICE)
            target_true_label = torch.ones(len(target_user_label), dtype = torch.long).to(DEVICE)
            loss_source_user_class = loss_adver(source_user_label, source_true_label)
            loss_source_item_class = loss_adver(source_item_label, source_true_label)
            loss_target_user_class = loss_adver(target_user_label, target_true_label)
            loss_target_item_class = loss_adver(target_item_label, target_true_label)
            loss += gamma * (loss_source_user_class + loss_source_item_class + loss_target_user_class + loss_target_item_class)
        
        loss.backward()
        optimizer.step()
        batch_step += 1
            
    rmse_target_train.append(math.sqrt(rmse_train_epoch/(batch_ind+1)))
    dual_net.eval()
    rmse_vali_epoch = 0
    rmse_test_epoch = 0
    with torch.no_grad():
        for batch_ind, (target_user_vali, target_item_vali, target_rating_vali) in enumerate(vali_target_loader):
            target_user_vali, target_item_vali, target_rating_vali = target_user_vali.to(DEVICE), target_item_vali.to(DEVICE), target_rating_vali.to(DEVICE)
            _, results_target_vali = dual_net(target_user_vali, target_item_vali, p = 0, source_scheme = 'target', rec_scheme = 'all', predict_scheme = 'MF')
            [_, _, target_predict_vali] = results_target_vali
            loss_target_vali = loss_recon(target_predict_vali, target_rating_vali)
            rmse_vali_epoch += loss_target_vali.item()
        rmse_target_vali.append(math.sqrt(rmse_vali_epoch/len(vali_target_loader)))                
        for batch_ind, (target_user_test, target_item_test, target_rating_test) in enumerate(test_target_loader):
            target_user_test, target_item_test, target_rating_test = target_user_test.to(DEVICE), target_item_test.to(DEVICE), target_rating_test.to(DEVICE)
            _, results_target_test = dual_net(target_user_test, target_item_test, p = 0, source_scheme = 'target', rec_scheme = 'all', predict_scheme = 'MF')
            [_, _, target_predict_test] = results_target_test
            loss_target_test = loss_recon(target_predict_test, target_rating_test)
            rmse_test_epoch += loss_target_test.item()
        rmse_target_test.append(math.sqrt(rmse_test_epoch/len(test_target_loader)))
        print('Draw_iter %d: RMSE train: %f, validation:%f, test:%f'% (cc+1, rmse_target_train[cc], rmse_target_vali[cc], rmse_target_test[cc]))
    cc +=1
    print('Epoch %d is finished'% (epoch+1))


with open(save_path + code_name + '.txt', 'w') as f:
    for ii in range(cc):
        f.write(str(rmse_target_train[ii]) + '\t' + str(rmse_target_vali[ii]) + '\t' + str(rmse_target_test[ii]) + '\n')

# x = np.arange(1,len(rmse_target_train)+1,1)
# y1 = np.array(rmse_target_train)
# y2 = np.array(rmse_target_vali)
# y3 = np.array(rmse_target_test)

# plt.plot(x, y1,'r--',label='Target Training Loss')
# plt.plot(x, y2,'b--',label='Validation Loss')
# plt.plot(x, y3,'g--',label='Test Loss')

# # # plt.xticks(np.arange(0,220,20),fontsize=14)
# # # plt.yticks(np.arange(1.20,1.40,0.05),fontsize=14)

# plt.xlabel('Iteration(mini-batch)',fontsize=12)
# plt.ylabel('MSE Error',fontsize=12)


# plt.legend(loc='upper right')
# plt.savefig('loss_dual.png')
# plt.show()






