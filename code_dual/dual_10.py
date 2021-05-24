import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import statistics

from CrossDataset import CrossDataset
from Model import Dual
from functions import LossMSE, LossOrth, LossMAE, evaluation_an_epoch

model_root = 'model'
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
show_epoch = True
epoch_num = 20
source_predict_scheme = 'MF'   # 'recon'


lr = 1e-2
batch_size = 64
weight_decay = 1e-6

num_dim_user_1 = 100
num_dim_item_1 = 100
num_dim_user_2 = 40
num_dim_item_2 = 40
num_dim_hidden = 10
alpha = 0.5
beta = 0.1
gamma = 1
active_domain_loss_step = 0

data_path = './data/amazon_book2movie_same.npz'
save_path = './result/amazon_same'
if not os.path.exists(save_path):
    os.makedirs(save_path)
dataset = CrossDataset(data_path)
num_user_source, num_item_source = dataset.source_train_m.shape
num_user_target, num_item_target = dataset.target_train_m.shape

train_loader = dataset.get_train_dataloader(batch_size)
vali_target_loader = dataset.get_test_dataloader(batch_size, mode = 'validation', data_source = 'target')
test_target_loader = dataset.get_test_dataloader(batch_size, mode = 'test', data_source = 'target')
vali_source_loader = dataset.get_test_dataloader(batch_size, mode = 'validation', data_source = 'source')
test_source_loader = dataset.get_test_dataloader(batch_size, mode = 'test', data_source = 'source')
del(dataset)
num_batch = len(train_loader)
dann_epoch = np.floor(active_domain_loss_step / num_batch * 1.0)

loss_recon = LossMSE().to(device)
loss_diff = LossOrth().to(device)
loss_adver = nn.CrossEntropyLoss().to(device)
loss_mae = LossMAE().to(device)


rmse_source_10, rmse_target_10, mae_source_10, mae_target_10 = [], [], [], []

for ii in range(10):
    dual_net = Dual(num_user_source, num_item_source, num_user_target, num_item_target, \
        num_dim_user_1, num_dim_item_1, num_dim_user_2, num_dim_item_2, num_dim_hidden).to(device)
    optimizer = optim.SGD([{'params':dual_net.parameters(),'lr':lr},], weight_decay=weight_decay)

    min_rmse_vali_source = 100
    min_rmse_vali_target = 100
    min_test = [0,0,0,0]

    for epoch in range(epoch_num):                                                                                                                 
        batch_step = 0
        rmse_target_train_epoch, rmse_source_train_epoch = 0, 0
        dual_net.train()
        for batch_ind, (source_user, source_item, source_rating, target_user, target_item, target_rating) in enumerate(train_loader):
            source_user, source_item, source_rating, target_user, target_item, target_rating = source_user.to(device), source_item.to(device), \
                source_rating.to(device), target_user.to(device), target_item.to(device), target_rating.to(device)
            dual_net.zero_grad()
            
            p = float(batch_ind + (epoch - dann_epoch) * num_batch / (epoch_num - dann_epoch) / num_batch)
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
            rmse_target_train_epoch += loss_target.item()
            rmse_source_train_epoch += loss_source.item()
            loss = 0
            loss += (loss_source * alpha + loss_target)
            # Orthorganal loss between private coding and sharing coding
            loss_source_user_diff = loss_diff(source_user_embeds_p, source_user_embeds_s)
            loss_source_item_diff = loss_diff(source_item_embeds_p, source_item_embeds_s)
            loss_target_user_diff = loss_diff(target_user_embeds_p, target_user_embeds_s)
            loss_target_item_diff = loss_diff(target_item_embeds_p, target_item_embeds_s)
            loss += beta * (loss_source_user_diff + loss_source_item_diff + loss_target_user_diff + loss_target_item_diff) 
            # Classification loss during adversarial training
            if batch_step > active_domain_loss_step:
                source_true_label = torch.zeros(len(source_user_label), dtype = torch.long).to(device)
                target_true_label = torch.ones(len(target_user_label), dtype = torch.long).to(device)
                loss_source_user_class = loss_adver(source_user_label, source_true_label)
                loss_source_item_class = loss_adver(source_item_label, source_true_label)
                loss_target_user_class = loss_adver(target_user_label, target_true_label)
                loss_target_item_class = loss_adver(target_item_label, target_true_label)
                loss += gamma * (loss_source_user_class + loss_source_item_class + loss_target_user_class + loss_target_item_class)
            
            loss.backward()
            optimizer.step()
            batch_step += 1

        dual_net.eval()
        rmse_target_vali_epoch, _ = evaluation_an_epoch(dual_net, vali_target_loader, source_scheme = 'target', predict_scheme = 'MF', device = device)                    
        rmse_source_vali_epoch, _ = evaluation_an_epoch(dual_net, vali_source_loader, source_scheme = 'source', predict_scheme = source_predict_scheme, device = device)
        rmse_target_test_epoch, mae_target_test_epoch = evaluation_an_epoch(dual_net, test_target_loader, source_scheme = 'target', predict_scheme = 'MF', device = device)
        rmse_source_test_epoch, mae_source_test_epoch = evaluation_an_epoch(dual_net, test_source_loader, source_scheme = 'source', predict_scheme = source_predict_scheme, device = device)
        if epoch == 1:
            min_test = [rmse_target_test_epoch, mae_target_test_epoch, rmse_source_test_epoch, mae_source_test_epoch]

        print('Epoch %d: Target RMSE train: %f, validation:%f, test:%f.\t Source RMSE train: %f, validation:%f, test:%f'% (epoch+1, \
            math.sqrt(rmse_target_train_epoch/num_batch), rmse_target_vali_epoch, rmse_target_test_epoch, \
                math.sqrt(rmse_source_train_epoch/num_batch), rmse_source_vali_epoch, rmse_source_test_epoch))
        if rmse_source_vali_epoch <= min_rmse_vali_source:
            min_rmse_vali_source = rmse_source_vali_epoch
            min_test[2], min_test[3] = rmse_source_test_epoch, mae_source_test_epoch
        if rmse_target_vali_epoch <= min_rmse_vali_target:
            min_rmse_vali_target = rmse_target_vali_epoch
            min_test[0], min_test[1] = rmse_target_test_epoch, mae_target_test_epoch
        if rmse_source_vali_epoch > min_rmse_vali_source and rmse_target_vali_epoch > min_rmse_vali_target:
            print('Iter %d, Para: alpha:%f, batch_size: %d, learning rate: %f. \n Target RMSE: %f, MAE: %f.\t Source RMSE: %f, MAE: %f.'% \
                (ii+1, alpha, batch_size, lr, min_test[0], min_test[1], min_test[2], min_test[3]))
            rmse_target_10.append(min_test[0])
            mae_target_10.append(min_test[1])
            rmse_source_10.append(min_test[2])
            mae_source_10.append(min_test[3])
            break
        
        if epoch == epoch_num -1:
            print('Epoch finished.')
            print('Para: alpha: %f, batch_size: %d, learning rate: %f. \n Target RMSE: %f, MAE: %f.\t Source RMSE: %f, MAE: %f.'% \
                    (alpha, batch_size, lr, rmse_target_test_epoch, mae_target_test_epoch, rmse_source_test_epoch, mae_source_test_epoch))
            rmse_target_10.append(rmse_target_test_epoch)
            mae_target_10.append(mae_target_test_epoch)
            rmse_source_10.append(rmse_source_test_epoch)
            mae_source_10.append(mae_source_test_epoch)
        
torch.save(dual_net.state_dict(), save_path + '/db_m2b_dual_' + str(lr) + '_' +str(batch_size) + '.pth')

target_mae_fin_mean = statistics.mean(mae_target_10)
target_mae_fin_std = statistics.stdev(mae_target_10)
target_rmse_fin_mean = statistics.mean(rmse_target_10)
target_rmse_fin_std = statistics.stdev(rmse_target_10)
print('FINAL Target MAE: mean: %f, std: %f, RMSE: mean: %f, std: %f ' % (target_mae_fin_mean, target_mae_fin_std, target_rmse_fin_mean, target_rmse_fin_std))

source_mae_fin_mean = statistics.mean(mae_source_10)
source_mae_fin_std = statistics.stdev(mae_source_10)
source_rmse_fin_mean = statistics.mean(rmse_source_10)
source_rmse_fin_std = statistics.stdev(rmse_source_10)
print('FINAL Source MAE: mean: %f, std: %f, RMSE: mean: %f, std: %f ' % (source_mae_fin_mean, source_mae_fin_std, source_rmse_fin_mean, source_rmse_fin_std))


with open(save_path + '/db_m2b_dual_' + str(lr) + '_' +str(batch_size) +'.txt', 'w') as f:
    for ii in range(10):
        f.write(str(rmse_target_10[ii]) + '\t' + str(mae_target_10[ii]) + '\t' + str(rmse_source_10[ii]) + '\t' + str(mae_source_10[ii]) + '\n')


# x = np.arange(1,len(rmse_target_train)+1,1)
# y1 = np.array(rmse_target_train)
# y2 = np.array(rmse_target_vali)
# y3 = np.array(rmse_target_test)

# plt.plot(x, y1,'r--',label='Target Training Loss')
# plt.plot(x, y2,'b--',label='Validation Loss')
# plt.plot(x, y3,'g--',label='Test Loss')

# # # plt.xticks(np.arange(0,220,20),fontsize=14)
# # # plt.yticks(np.arange(1.20,1.40,0.05),fontsize=14)

# # plt.xlabel('Iteration(mini-batch)',fontsize=12)
# # plt.ylabel('MSE Error',fontsize=12)


# plt.legend(loc='upper right')
# plt.savefig('loss_dual.png')
# plt.show()





