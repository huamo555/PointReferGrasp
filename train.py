import argparse
import torch
import numpy as np
import os
import logging
from utils.util import seed_torch, set_gpu_devices, read_yaml
from utils.logger import logger as loggger

parser = argparse.ArgumentParser()

parser.add_argument("-v", type=str, required=True, help="version", default="try")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=48)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-tlr", type=float, action="store", help="learning rate for text encoder", default=5e-6)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=50)

parser.add_argument("-n_groups", type=int, action="store", help="num of queries", default=40)
parser.add_argument("-gpu", type=int, help="set gpu id", default=0) 
parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-3]')
parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
parser.add_argument('--save_dir', type=str, default='runs/train/', help='path to save .pt model while training')
parser.add_argument('--name', type=str, default='PointRefer', help='training name to classify each training process')
parser.add_argument('--resume', type=str, default=False, help='start training from previous epoch')
parser.add_argument('--checkpoint_path', type=str, default='runs/train/PointRefer/best.pt', help='checkpoint path')
parser.add_argument('--log_name', type=str, default='train.log', help='the name of current training')
parser.add_argument('--loss_cls', type=float, default=0.3, help='cls loss scale')
parser.add_argument('--storage', type=bool, default=False, help='whether to storage the model during training')
parser.add_argument('--yaml', type=str, default='config/default.yaml', help='yaml path')

opt = parser.parse_args()

seed_torch(seed=42)
set_gpu_devices(opt.gpu)

import torch.nn as nn
from torch.utils.data import DataLoader
from model.PointRefer import get_PointRefer
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from eval_lyc import evaluate, print_metrics_in_table 
from data_utils.shapenetpart import AffordQ
from sklearn.metrics import roc_auc_score


def main(opt, dict):
    logger, sign = loggger(opt)

    if opt.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    save_path = opt.save_dir + opt.name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = dict['bs']

    logger.debug('Start loading train data---')
    train_dataset = AffordQ('train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 ,shuffle=True, drop_last=True)
    logger.debug(f'train data loading finish, loading data files:{len(train_dataset)}')

    logger.debug('Start loading val data---')
    val_dataset = AffordQ('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'val data loading finish, loading data files:{len(val_dataset)}')

    logger.debug('Start loading test data---')
    test_dataset = AffordQ('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'test data loading finish, loading data files:{len(test_dataset)}')

    model = get_PointRefer(emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'], n_groups=opt.n_groups)

    # move model and loss to device FIRST
    model = model.to(device)
    criterion_hm = HM_Loss().to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # optimizer AFTER .to(device)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": opt.tlr}
    ]
    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=dict['lr'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if opt.resume:
        model_checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        start_epoch = -1

    best_IOU = 0

    '''
    Training
    '''
    for epoch in range(start_epoch+1, dict['Epoch']):
        logger.debug(f'Epoch:{epoch} strat-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        logger.debug(f'lr_rate:{learning_rate}')

        num_batches = len(train_loader)
        loss_sum = 0
        model = model.train()
        for i,(point, cls, gt_mask, question, aff_label) in enumerate(train_loader):
            
            optimizer.zero_grad()      

            if(opt.use_gpu):
                point = point.to(device)
                question = question   # FIXED
                gt_mask = gt_mask.to(device)
                aff_label = aff_label.to(device)
                cls = cls.to(device)

            _3d = model(question, point)
            loss_hm = criterion_hm(_3d, gt_mask)
            temp_loss = loss_hm 

            print(f'Epoch:{epoch}| iteration:{i}|{len(train_loader)} | loss:{temp_loss.item()}')
            temp_loss.backward()
            optimizer.step()
            loss_sum += temp_loss.item()

        mean_loss = loss_sum / (num_batches*dict['pairing_num'])
        logger.debug(f'Epoch:{epoch} | mean_loss:{mean_loss}')

        if(opt.storage):
            model_path = save_path + f'/Epoch_{epoch+1}.pt'
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Epoch': epoch
            }
            torch.save(checkpoint, model_path)
            logger.debug(f'model saved at {model_path}')
        
        '''
        Evalization
        '''
        if((epoch+1)%1 == 0):
            num = 0
            with torch.no_grad():
                logger.debug(f'EVALUATION strat-------')
                total_MAE = 0
                total_point = 0
                model = model.eval()
                results = torch.zeros((len(val_dataset), 2048, 1))
                targets = torch.zeros((len(val_dataset), 2048, 1))
                for i,(point, _, label, question,aff_label) in enumerate(val_loader):
                    print(f'iteration: {i}|{len(val_loader)} start----')
                    point, label = point.float(), label.float()
                    if(opt.use_gpu):
                        point = point.to(device)
                        question = question   # FIXED
                        label = label.to(device)
                    
                    _3d = model(question, point)
                    mae, point_nums = evaluating(_3d, label)
                    total_point += point_nums
                    total_MAE += mae.item()
                    pred_num = _3d.shape[0]
                    results[num : num+pred_num, :, :] = _3d.cpu().unsqueeze(-1)
                    targets[num : num+pred_num, :, :] = label.cpu().unsqueeze(-1)
                    num += pred_num

                mean_mae = total_MAE / total_point
                results = results.numpy()
                targets = targets.numpy()

                SIM_matrix = np.zeros(targets.shape[0])
                for i in range(targets.shape[0]):
                    SIM_matrix[i] = SIM(results[i], targets[i])
                
                sim = np.mean(SIM_matrix)
                AUC = np.zeros((targets.shape[0], targets.shape[2]))
                IOU = np.zeros((targets.shape[0], targets.shape[2]))
                IOU_thres = np.linspace(0, 1, 20)
                targets = targets >= 0.5
                targets = targets.astype(int)
                for i in range(AUC.shape[0]):
                    t_true = targets[i]
                    p_score = results[i]

                    if np.sum(t_true) == 0:
                        AUC[i] = np.nan
                        IOU[i] = np.nan
                    else:
                        auc = roc_auc_score(t_true, p_score)
                        AUC[i] = auc

                        temp_iou = []
                        for thre in IOU_thres:
                            p_mask = (p_score >= thre).astype(int)
                            intersect = np.sum(p_mask & t_true)
                            union = np.sum(p_mask | t_true)
                            temp_iou.append(1.*intersect/union)
                        IOU[i] = np.mean(temp_iou)
                
                AUC = np.nanmean(AUC)
                IOU = np.nanmean(IOU)

                logger.debug(f'AUC:{AUC} | IOU:{IOU} | SIM:{sim} | MAE:{mean_mae}')

                if IOU > best_IOU:
                    best_IOU = IOU
                    best_model_path = save_path + f'/best_model-{sign}.pt'
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'Epoch': epoch
                    }
                    torch.save(checkpoint, best_model_path)
                    logger.debug(f'best model saved at {best_model_path}')
        scheduler.step()
    logger.debug(f'Best Val IOU:{best_IOU}')

    category_metrics, affordance_metrics, overall_metrics = evaluate(model, test_loader, device, 3)
    print_metrics_in_table(category_metrics, affordance_metrics, overall_metrics, logger)


if __name__=='__main__':
    dict = read_yaml(opt.yaml)
    dict['bs'] = opt.bs
    dict['lr'] = opt.lr
    dict['Epoch'] = opt.epoch
    main(opt, dict)
