from torch.autograd import Function
import torch.nn as nn
import torch
import math


def get_uir(target, target_m):
    num_target = target.shape[0]
    user = target_m[target[:,0].astype(int), :]
    item = target_m[:, target[:,1].astype(int)].T
    rating = target[:,2]
    user, item, rating = torch.FloatTensor(user), torch.FloatTensor(item), torch.FloatTensor(rating)
    return user, item, rating

def get_uir_five(target, target_m):
    num_target = target.shape[0]
    userid = target[:,0].astype(int)
    itemid = target[:,1].astype(int)
    user = target_m[userid, :]
    item = target_m[:, itemid].T
    rating = target[:,2]

    user, item, userid, itemid, rating = torch.FloatTensor(user), torch.FloatTensor(item), torch.LongTensor(userid), torch.LongTensor(itemid), torch.FloatTensor(rating)
    return user, item, userid, itemid, rating

def clip_pred(a):
    # vec is pytorch tensor
    a[a>5] = 5
    a[a<1] = 1
    return a

def evaluation_an_epoch(dual_net, vali_target_loader, source_scheme, predict_scheme, device):
    rmse_vali_epoch = 0
    mae_vali_epoch = 0
    loss_recon = LossMSE().to(device)
    loss_mae = LossMAE().to(device)

    with torch.no_grad():
        for batch_ind, (target_user_vali, target_item_vali, target_rating_vali) in enumerate(vali_target_loader):
            target_user_vali, target_item_vali, target_rating_vali = target_user_vali.to(device), target_item_vali.to(device), target_rating_vali.to(device)
            _, results_target_vali = dual_net(target_user_vali, target_item_vali, p = 0, source_scheme = source_scheme, rec_scheme = 'all', predict_scheme = predict_scheme)
            [_, _, target_predict_vali] = results_target_vali
            if predict_scheme == 'recon':
                target_predict_vali[0] = clip_pred(target_predict_vali[0])
                target_predict_vali[1] = clip_pred(target_predict_vali[1])
                loss_target_vali = 0.5 * (loss_recon(target_predict_vali[0], target_user_vali) + loss_recon(target_predict_vali[1], target_item_vali))
                loss_target_mae = 0.5 * (loss_mae(target_predict_vali[0], target_user_vali) + loss_mae(target_predict_vali[1], target_item_vali))
            elif predict_scheme == 'MF':
                target_predict_vali = clip_pred(target_predict_vali)
                loss_target_vali = loss_recon(target_predict_vali, target_rating_vali)
                loss_target_mae = loss_mae(target_predict_vali, target_rating_vali)         
            rmse_vali_epoch += loss_target_vali.item()
            mae_vali_epoch += loss_target_mae.item()
        rmse_vali_epoch = math.sqrt(rmse_vali_epoch/len(vali_target_loader))
        mae_vali_epoch = mae_vali_epoch/len(vali_target_loader)
    return rmse_vali_epoch, mae_vali_epoch


def one_evaluation_an_epoch(dual_net, vali_target_loader, source_scheme, share_scheme, device):
    rmse_vali_epoch = 0
    mae_vali_epoch = 0
    loss_recon = LossMSE().to(device)
    loss_mae = LossMAE().to(device)

    with torch.no_grad():
        for batch_ind, (target_user_vali, target_item_vali, target_rating_vali) in enumerate(vali_target_loader):
            target_user_vali, target_item_vali, target_rating_vali = target_user_vali.to(device), target_item_vali.to(device), target_rating_vali.to(device)
            _, results_target_vali = dual_net(target_user_vali, target_item_vali, p = 0, source_scheme = source_scheme, share_scheme = share_scheme)
            [_, _, target_predict_vali] = results_target_vali

            target_predict_vali = clip_pred(target_predict_vali)
            loss_target_vali = loss_recon(target_predict_vali, target_rating_vali)
            loss_target_mae = loss_mae(target_predict_vali, target_rating_vali)         
            rmse_vali_epoch += loss_target_vali.item()
            mae_vali_epoch += loss_target_mae.item()
        rmse_vali_epoch = math.sqrt(rmse_vali_epoch/len(vali_target_loader))
        mae_vali_epoch = mae_vali_epoch/len(vali_target_loader)
    return rmse_vali_epoch, mae_vali_epoch

def da_evaluation_an_epoch(darec_net, vali_target_loader, source_scheme, device):
    rmse_vali_epoch = 0
    mae_vali_epoch = 0
    loss_recon = LossMSE().to(device)
    loss_mae = LossMAE().to(device)

    with torch.no_grad():
        for batch_ind, target_user_vali in enumerate(vali_target_loader):
            target_user_vali = target_user_vali[0].to(device)
            _, target_predict_vali, _ = darec_net(target_user_vali, source_scheme = source_scheme, p = 0)
            target_predict_vali = clip_pred(target_predict_vali)
            loss_target_vali = loss_recon(target_predict_vali, target_user_vali)
            rmse_vali_epoch += loss_target_vali.item()
            loss_target_mae = loss_mae(target_predict_vali, target_user_vali)
            mae_vali_epoch += loss_target_mae.item()
        rmse_vali_epoch = math.sqrt(rmse_vali_epoch/len(vali_target_loader))
        mae_vali_epoch = mae_vali_epoch/len(vali_target_loader)
    return rmse_vali_epoch, mae_vali_epoch


def recon_evaluation_an_epoch(dual_net, vali_target_loader, source_scheme, predict_scheme, device):
    rmse_vali_epoch = 0
    mae_vali_epoch = 0
    loss_recon = LossMSE().to(device)
    loss_mae = LossMAE().to(device)

    with torch.no_grad():
        for batch_ind, (target_user_vali, target_item_vali, target_userid, target_itemid, target_rating_vali) in enumerate(vali_target_loader):
            target_user_vali, target_item_vali, target_rating_vali = target_user_vali.to(device), target_item_vali.to(device), target_rating_vali.to(device)
            target_userid, target_itemid = target_userid.to(device), target_itemid.to(device)
            _, results_target_vali = dual_net(target_user_vali, target_item_vali, p = 0, source_scheme = source_scheme, rec_scheme = 'all', predict_scheme = predict_scheme)
            [_, _, target_predict_vali] = results_target_vali

            target_predict_u = target_predict_vali[0].gather(1, target_itemid.view(-1,1))
            target_predict_i = target_predict_vali[1].gather(1, target_userid.view(-1,1))
            target_predict_vali = 0.5 * (target_predict_u + target_predict_i)
            target_predict_vali = torch.squeeze(target_predict_vali)
                
            target_predict_vali = clip_pred(target_predict_vali)
            loss_target_vali = loss_recon(target_predict_vali, target_rating_vali)
            loss_target_mae = loss_mae(target_predict_vali, target_rating_vali)         
            rmse_vali_epoch += loss_target_vali.item()
            mae_vali_epoch += loss_target_mae.item()
        rmse_vali_epoch = math.sqrt(rmse_vali_epoch/len(vali_target_loader))
        mae_vali_epoch = mae_vali_epoch/len(vali_target_loader)
    return rmse_vali_epoch, mae_vali_epoch



class LossMSE(nn.Module):
    def __init__(self):
        super(LossMSE, self).__init__()

    def forward(self, pred, real):
        pred_loss=pred.clone()
        pred_loss[real==0]=0
        diffs = torch.add(real, -pred_loss)
        n = len(torch.nonzero(real))
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class LossOrth(nn.Module):

    def __init__(self):
        super(LossOrth, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class LossMAE(nn.Module):
    def __init__(self):
        super(LossMAE, self).__init__()

    def forward(self, pred, real):

        pred_loss=pred.clone()
        pred_loss[real==0]=0
        diffs = torch.add(real, -pred_loss)
        n = len(torch.nonzero(real))
        mae = torch.sum(torch.abs(diffs)) / n

        return mae




