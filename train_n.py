import argparse
import torch
import numpy as np
import os
import logging
from utils.util import seed_torch, set_gpu_devices, read_yaml
from utils.logger import logger as loggger
import time

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", type=str, required=True, help="version", default="try")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=64)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-tlr", type=float, action="store", help="learning rate for text encoder", default=5e-6)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=60)
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

# Set deterministic behavior and GPU device
seed_torch(seed=42)
set_gpu_devices(opt.gpu)

# Local imports for training components
import torch.nn as nn
from torch.utils.data import DataLoader
from model.PointRefer import get_PointRefer_Iterative_Einsum
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from eval_lyc import evaluate, print_metrics_in_table
from data_utils.shapenetpart import AffordQ
from sklearn.metrics import roc_auc_score

def evaluate_on_loader(model, loader, device):
    """
    Evaluate model on a single dataloader.
    Returns: dict with keys 'AUC', 'IOU', 'SIM', 'MAE'
    This duplicates the metric computations used during training to keep evaluation consistent.
    """
    model = model.eval()
    total_MAE = 0.0
    total_point = 0
    num = 0

    dataset_len = len(loader.dataset)
    results = torch.zeros((dataset_len, 2048, 1))
    targets = torch.zeros((dataset_len, 2048, 1))

    with torch.no_grad():
        for i, (point, _, label, question, aff_label) in enumerate(loader):
            # Debug print for iteration
            print(f'[EVAL] iteration: {i}/{len(loader)} start----')

            point, label = point.float(), label.float()
            if opt.use_gpu:
                point = point.to(device)
                question = question  # text kept as-is (fixed in original code)
                label = label.to(device)

            # Model forward
            all_mask_logits = model(question, point) 
            
            final_mask_logits = all_mask_logits[-1]
            _3d = torch.sigmoid(final_mask_logits)

            # Per-batch MAE and point count (evaluating returns (mae_tensor, point_count))
            mae, point_nums = evaluating(_3d, label)

            total_point += point_nums
            total_MAE += mae.item()

            # Collect predictions and targets for global metrics
            pred_num = _3d.shape[0]
            results[num: num + pred_num, :, :] = _3d.cpu().unsqueeze(-1)
            targets[num: num + pred_num, :, :] = label.cpu().unsqueeze(-1)
            num += pred_num

    # Finalize metrics
    mean_mae = total_MAE / total_point if total_point > 0 else float('nan')
    results = results.numpy()
    targets = targets.numpy()

    # SIM per-sample
    SIM_matrix = np.zeros(targets.shape[0])
    for i in range(targets.shape[0]):
        SIM_matrix[i] = SIM(results[i], targets[i])
    sim = np.mean(SIM_matrix)

    # AUC and IOU calculations
    AUC = np.zeros((targets.shape[0], targets.shape[2]))
    IOU = np.zeros((targets.shape[0], targets.shape[2]))
    IOU_thres = np.linspace(0, 1, 20)

    targets_bin = (targets >= 0.5).astype(int)

    for i in range(AUC.shape[0]):
        t_true = targets_bin[i]
        p_score = results[i]

        if np.sum(t_true) == 0:
            AUC[i] = np.nan
            IOU[i] = np.nan
        else:
            try:
                auc = roc_auc_score(t_true, p_score)
            except ValueError:
                auc = np.nan
            AUC[i] = auc

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union != 0 else 0.0)
            IOU[i] = np.mean(temp_iou)

    mean_auc = np.nanmean(AUC)
    mean_iou = np.nanmean(IOU)

    return {'AUC': mean_auc, 'IOU': mean_iou, 'SIM': sim, 'MAE': mean_mae}


def main(opt, config_dict):
    """
    Main training loop with validation and test evaluation.
    Validation is used to pick the best model (based on IOU).
    Test evaluation is executed each epoch and only reported (not used to save models).
    """
    logger, sign = loggger(opt)

    # Device selection
    if opt.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create save directory
    save_path = os.path.join(opt.save_dir, opt.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = config_dict['bs']

    # Load datasets and dataloaders
    logger.debug('Start loading train data---')
    train_dataset = AffordQ('train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    logger.debug(f'train data loading finish, number of samples: {len(train_dataset)}')

    logger.debug('Start loading val data---')
    val_dataset = AffordQ('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'val data loading finish, number of samples: {len(val_dataset)}')

    logger.debug('Start loading test data---')
    test_dataset = AffordQ('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'test data loading finish, number of samples: {len(test_dataset)}')

    # Build model
    model = get_PointRefer_Iterative_Einsum(
                           emb_dim=config_dict['emb_dim'],
                           proj_dim=config_dict['proj_dim'],
                           num_heads=config_dict['num_heads'],
                           N_raw=config_dict['N_raw'],
                           num_affordance=config_dict['num_affordance'],
                           n_groups=opt.n_groups,
                           num_stages=2 
                           )

    # Move model and losses to device
    model = model.to(device)
    criterion_hm = HM_Loss().to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # Set up optimizer with differential lr for text encoder parameters
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": opt.tlr}
    ]
    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=config_dict['lr'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Resume from checkpoint if required
    if opt.resume:
        model_checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        start_epoch = -1

    best_IOU = 0.0
    best_model_path = None

    # Training loop
    for epoch in range(start_epoch + 1, config_dict['Epoch']):
        logger.debug(f'Epoch:{epoch} start-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        logger.debug(f'lr_rate:{learning_rate}')

        num_batches = len(train_loader)
        loss_sum = 0.0
        model.train()

        for i, (point, cls, gt_mask, question, aff_label) in enumerate(train_loader):
            optimizer.zero_grad()

            if opt.use_gpu:
                point = point.to(device)
                question = question  # text tensor/strings remain as provided
                gt_mask = gt_mask.to(device)
                aff_label = aff_label.to(device)
                cls = cls.to(device)

            # forward
            
            start_time = time.time()
            all_mask_logits = model(question, point) 
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Model running time: {elapsed_time:.4f} seconds")
            
            total_loss = 0

            loss_weights = [1.0, 0.5] 
            
            for stage_idx, logits in enumerate(all_mask_logits):
                pred_mask_sigmoid = torch.sigmoid(logits)
                
                loss_hm = criterion_hm(pred_mask_sigmoid, gt_mask)
                
                total_loss += loss_hm * loss_weights[stage_idx]

            loss = total_loss
            
            print(f'Epoch:{epoch}| iteration:{i}/{num_batches} | loss:{loss.item():.6f}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        mean_loss = loss_sum / (num_batches * config_dict.get('pairing_num', 1))
        logger.debug(f'Epoch:{epoch} | mean_loss:{mean_loss:.6f}')

        # Optionally save per-epoch model
        if opt.storage:
            model_path = os.path.join(save_path, f'Epoch_{epoch+1}.pt')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Epoch': epoch
            }
            torch.save(checkpoint, model_path)
            logger.debug(f'model saved at {model_path}')

        # Validation and Test evaluation stage (runs every epoch here)
        if (epoch + 1) % 1 == 0:
            logger.debug('EVALUATION start-------')

            # Validate on validation set and compute metrics
            val_metrics = evaluate_on_loader(model, val_loader, device)
            logger.debug(f'VAL--- AUC:{val_metrics["AUC"]:.6f} | IOU:{val_metrics["IOU"]:.6f} | SIM:{val_metrics["SIM"]:.6f} | MAE:{val_metrics["MAE"]:.6f}')

            # Test on test set and compute metrics (do not use test metrics to choose best model)
            test_metrics = evaluate_on_loader(model, test_loader, device)
            logger.debug(f'TEST--- AUC:{test_metrics["AUC"]:.6f} | IOU:{test_metrics["IOU"]:.6f} | SIM:{test_metrics["SIM"]:.6f} | MAE:{test_metrics["MAE"]:.6f}')

            # Save best model based on validation IOU
            if val_metrics['IOU'] > best_IOU:
                best_IOU = val_metrics['IOU']
                best_model_path = os.path.join(save_path, f'best_model-{sign}.pt')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'Epoch': epoch
                }
                torch.save(checkpoint, best_model_path)
                logger.debug(f'best model saved at {best_model_path}')

        # Step scheduler
        scheduler.step()

    logger.debug(f'Best Val IOU:{best_IOU:.6f}')

    # After training, optionally run final test evaluation using the final model
    # Here we use the current model (which may or may not be the saved best model).
    category_metrics, affordance_metrics, overall_metrics = evaluate(model, test_loader, device, 3)
    print_metrics_in_table(category_metrics, affordance_metrics, overall_metrics, logger)


if __name__ == '__main__':
    # Read YAML config and inject CLI-specified overrides
    config = read_yaml(opt.yaml)
    config['bs'] = opt.bs
    config['lr'] = opt.lr
    config['Epoch'] = opt.epoch

    main(opt, config)
