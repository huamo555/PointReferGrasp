import torch
import numpy as np
from torch.utils.data import DataLoader
from model.PointRefer import get_PointRefer_Iterative_Einsum
from utils.util import seed_torch, read_yaml
from data_utils.shapenetpart import AffordQ
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd


def calculate_batch_iou_auc(results, targets):
    auc = np.zeros(results.shape[0])
    iou = np.zeros(results.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    targets = targets >= 0.5
    targets = targets.astype(int)
    for i in range(results.shape[0]):
        t_true = targets[i]
        p_score = results[i]

        if np.sum(t_true) == 0:
            auc_ = np.nan
            iou_ = np.nan
        else:
            try:
                auc_ = roc_auc_score(t_true, p_score)
            except ValueError:
                auc_ = np.nan # Handle cases where only one class is present in y_true

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1.*intersect/union if union != 0 else 0.0)
            iou_ = np.mean(temp_iou)
        auc[i]=auc_
        iou[i]=iou_
    return iou, auc

def calculate_batch_sim(pred, target):
    # Adding a small epsilon to prevent division by zero
    pred_norm = pred / (np.sum(pred, axis=1, keepdims=True) + 1e-12)
    target_norm = target / (np.sum(target, axis=1, keepdims=True) + 1e-12)
    sim = np.sum(np.minimum(pred_norm, target_norm), axis=1)
    return sim

def calculate_batch_mae(pred, true):
    mae = np.mean(np.abs(pred - true), axis=1)
    return mae


def evaluate(model, test_loader, device, num_votes=3):
    # Initialize the structures for holding the predictions and targets
    category_iou = np.zeros(23)
    category_auc = np.zeros(23)
    category_sim = np.zeros(23)
    category_mae = np.zeros(23)
    category_count = np.zeros(23)
    category_metrics = {}
    
    affordance_iou = np.zeros(18)
    affordance_auc = np.zeros(18)
    affordance_sim = np.zeros(18)
    affordance_mae = np.zeros(18)
    affordance_count = np.zeros(18)
    affordance_metrics = {}
    

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (point, cls, label, question, aff_label) in tqdm(enumerate(test_loader), total=len(test_loader), ascii=True):
            point, label = point.float(), label.float()
            point = point.to(device)
            label = label.to(device)

          
            aggregated_fused_logits = torch.zeros_like(label)

         
            for v in range(num_votes):
                seed_torch(v) 
                
              
                stage_outputs = model(question, point)
                stage1_logits = stage_outputs[0]
                stage2_logits = stage_outputs[1]
                

                current_vote_fused_logits = (stage1_logits + stage2_logits)
                
                
                aggregated_fused_logits += current_vote_fused_logits
                
           
            final_averaged_logits = aggregated_fused_logits / num_votes
           
            _3d = torch.sigmoid(final_averaged_logits)
            # ########## MODIFICATION END ##########


            """ 
            ########### Evaluation ##########
            """
            _3d = _3d.cpu().numpy()
            label = label.cpu().numpy()
            cls = cls.numpy()
            aff_label = aff_label.numpy()

            iou, auc = calculate_batch_iou_auc(_3d, label)
            sim = calculate_batch_sim(_3d, label)
            mae = calculate_batch_mae(_3d, label)

            for index, (cls_id, aff_id) in enumerate(zip(cls, aff_label)):
                 # For object class
                category_iou[cls_id.item()] += iou[index]  
                category_auc[cls_id.item()] += auc[index]
                category_sim[cls_id.item()] += sim[index]
                category_mae[cls_id.item()] += mae[index]
                category_count[cls_id.item()] += 1

                # For affordance class
                affordance_iou[aff_id.item()] += iou[index]  
                affordance_auc[aff_id.item()] += auc[index]
                affordance_sim[aff_id.item()] += sim[index]
                affordance_mae[aff_id.item()] += mae[index]
                affordance_count[aff_id.item()] += 1


        overall_metrics = {
            'IOU': np.nansum(category_iou) / category_count.sum(),
            'AUC': np.nansum(category_auc) / category_count.sum(),
            'SIM': np.nansum(category_sim) / category_count.sum(),
            'MAE': np.nansum(category_mae) / category_count.sum()
        }

        # Calculate metrics for each category
        # Use np.divide to handle division by zero safely, resulting in NaN
        category_iou = np.divide(category_iou, category_count, out=np.full_like(category_iou, np.nan), where=category_count!=0)
        category_auc = np.divide(category_auc, category_count, out=np.full_like(category_auc, np.nan), where=category_count!=0)
        category_sim = np.divide(category_sim, category_count, out=np.full_like(category_sim, np.nan), where=category_count!=0)
        category_mae = np.divide(category_mae, category_count, out=np.full_like(category_mae, np.nan), where=category_count!=0)

        affordance_iou = np.divide(affordance_iou, affordance_count, out=np.full_like(affordance_iou, np.nan), where=affordance_count!=0)
        affordance_auc = np.divide(affordance_auc, affordance_count, out=np.full_like(affordance_auc, np.nan), where=affordance_count!=0)
        affordance_sim = np.divide(affordance_sim, affordance_count, out=np.full_like(affordance_sim, np.nan), where=affordance_count!=0)
        affordance_mae = np.divide(affordance_mae, affordance_count, out=np.full_like(affordance_mae, np.nan), where=affordance_count!=0)

        for cls_id in range(len(category_iou)):
            category_metrics[cls_id]={'IOU': category_iou[cls_id], 'AUC': category_auc[cls_id], \
                                      'SIM': category_sim[cls_id], 'MAE': category_mae[cls_id]}

        for aff_id in range(len(affordance_iou)):
            affordance_metrics[aff_id]={'IOU': affordance_iou[aff_id], 'AUC': affordance_auc[aff_id], \
                                      'SIM': affordance_sim[aff_id], 'MAE': affordance_mae[aff_id]}


    return category_metrics, affordance_metrics, overall_metrics



import pandas as pd
from model.PointRefer import get_PointRefer_Iterative_Einsum

def print_metrics_in_table(category_metrics, affordance_metrics, overall_metrics, logger = None):
    # Define the class and affordance names
    classes = ["Bag", "Bed", "Bowl", "Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
               "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
               "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]
    
    affordances = ['lay', 'sit', 'support', 'grasp', 'lift', 'contain', 'open', 'wrap_grasp', 'pour', 
                   'move', 'display', 'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']

    # Set the precision for floating point numbers in pandas
    pd.set_option('display.float_format', '{:.3f}'.format)

    # Convert the metrics dictionaries into DataFrames
    category_df = pd.DataFrame(category_metrics)
    affordance_df = pd.DataFrame(affordance_metrics)

    # Rename the indices with the actual class and affordance names
    category_df.rename(index={i: classes[i] for i in range(len(classes))}, inplace=True)
    affordance_df.rename(index={i:affordances[i] for i in range(len(affordances))}, inplace=True)
    if logger:
        # Print the metrics for each category in a tabular format
        logger.debug("\nCategory Metrics:")
        logger.debug(category_df.to_string())  # Using to_string to print the entire DataFrame without truncation

        logger.debug("\nAffordance Metrics:")
        logger.debug(affordance_df.to_string())  # Using to_string to print the entire DataFrame without truncation

        # Print overall metrics
        logger.debug("\nOverall Metrics:")

    else:
        # Print the metrics for each category in a tabular format
        print("\nCategory Metrics:")
        print(category_df.to_string())  # Using to_string to print the entire DataFrame without truncation

        print("\nAffordance Metrics:")
        print(affordance_df.to_string())  # Using to_string to print the entire DataFrame without truncation

        # Print overall metrics
        print("\nOverall Metrics:")
    # Formatting overall metrics for consistent display
    overall_metrics_formatted = {key: f"{value:.3f}" for key, value in overall_metrics.items()}
    for metric, value in overall_metrics_formatted.items():
        if logger:
            logger.debug(f"{metric}: {value}")
        else:
            print(f"{metric}: {value}")


if __name__=='__main__':

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    test_dataset = AffordQ('val')
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=8, shuffle=False)
    
    # dict = read_yaml('config/config_seen.yaml')
    model = get_PointRefer_Iterative_Einsum(emb_dim=512,
                       proj_dim=512, num_heads=4, N_raw=2048,
                       num_affordance =18)
    model_checkpoint = torch.load("/data3/gaoyuming/Task_Oriented_Grasping/LASO-main_PWCA+mutilstage/runs/train/PointRefer/best_model-try_at_10.19_17.43.12.pt", map_location='cuda:3')
    model.load_state_dict(model_checkpoint['model'])
    model.to(device)
   
    category_metrics, affordance_metrics, overall_metrics = evaluate(model, test_loader, device, 3)
    print_metrics_in_table(category_metrics, affordance_metrics, overall_metrics)
