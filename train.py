import time
import math
from datetime import timedelta
import sys

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from base.east_dataset import EASTDataset
from base.model import EAST
from data_loader.dataset import SceneTextDataset, CORDDataset, MergedDataset
from data_loader.transform import get_train_transform, get_val_transform

from deteval import calc_deteval_metrics

from utils.custom import *
from utils.argeParser import parse_args
from utils.accuracy_metric import get_gt_bboxes, get_pred_bboxes, get_lang_pred_bboxes, get_lang_gt_bboxes

def do_training(dataset, train_lang_list, valid, resume, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, entity, project_name, model_name):
    
    '''
    dataset
    '''
    if dataset == 'Base':
        train_dataset = SceneTextDataset(
            data_dir,
            split='train',
            train_lang_list=train_lang_list,
            image_size=image_size,
            crop_size=input_size,
            transforms=get_train_transform()
        )
        if valid: 
            val_dataset = SceneTextDataset(
                data_dir,
                split='val',
                image_size=image_size,
                crop_size=input_size,
                transforms=get_val_transform()
            )

    elif dataset == 'CORD':
        train_dataset = CORDDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        transforms=get_train_transform()
        )
        if valid:
            val_dataset = CORDDataset(
            data_dir,
            split='val',
            image_size=image_size,
            crop_size=input_size,
            transforms=get_val_transform()
            )

    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if valid:
        val_dataset = EASTDataset(val_dataset)
        val_num_batches = math.ceil(len(val_dataset) / batch_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


    '''
    model
    '''
    model = EAST()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[75], gamma=0.1)

    model_save_and_delete = ModelSaveAndDelete(model, model_dir,3)     
    my_wandb = MyWandb(entity, project_name, model_name)
    loss_names = my_wandb.loss_names
    my_wandb.init(learning_rate, batch_size, max_epoch, image_size, input_size)


    '''
    resume
    '''
    if resume != 'None':
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint)
    

    '''
    train
    '''
    train_step, val_step = 0, 0  # 초기화

    for epoch in range(max_epoch):
        # ========== train ===========
        model.train()
        train_epoch_losses = [0.,0.,0.,0.]
        epoch_start = time.time()   

        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description(f'[Epoch {epoch + 1}]')
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_losses = [loss.item(), extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss']]
                
                for i, iter_loss in enumerate(iter_losses):
                    # None인 경우 0으로 대체하여 더할 수 있도록 처리
                    if iter_loss is None:
                        iter_losses[i] = 0.0
                    train_epoch_losses[i] += iter_losses[i]

                pbar.update(1)
                val_dict = dict(zip(loss_names, iter_losses))
                pbar.set_postfix(val_dict)

                my_wandb.save_iter('train', iter_losses, train_step)   
                train_step += 1                 

        scheduler.step()

        mean_losses = [loss / train_num_batches for loss in train_epoch_losses]
        print(f'Mean loss: {mean_losses[0]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
        
        my_wandb.save_epoch('train', epoch, optimizer.param_groups[0]['lr'], mean_losses)

        # ======== validation =========   
        if valid:
            with torch.no_grad():
                model.eval()
                val_epoch_losses = [0., 0., 0., 0.]
                epoch_start = time.time()
                
                with tqdm(total=val_num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                        pbar.set_description('Validate : [Epoch {}]'.format(epoch + 1))
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        
                        iter_losses = [loss.item(), extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss']]
                        
                        for i, iter_loss in enumerate(iter_losses):
                            # None인 경우 0으로 대체하여 더할 수 있도록 처리
                            if iter_loss is None:
                                iter_losses[i] = 0.0
                            val_epoch_losses[i] += iter_losses[i]

                        pbar.update(1)
                        val_dict = dict(zip(loss_names, iter_losses))
                        pbar.set_postfix(val_dict)

                        my_wandb.save_iter('val', iter_losses, val_step)
                        val_step += 1

                mean_losses = [loss / val_num_batches for loss in val_epoch_losses]
                print(f'Mean Validation loss: {mean_losses[0]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')

                ''' epoch 당 valid accuracy 구할 때 '''
                epoch_start = time.time()
                gt_bboxes = get_gt_bboxes(data_dir)
                pred_bboxes = get_pred_bboxes(model, data_dir, input_size, batch_size)

                result = calc_deteval_metrics(pred_bboxes, gt_bboxes)['total']
                precision, recall, f1_score = result['precision'], result['recall'], result['hmean']
                total_accuracies = [precision, recall, f1_score]

                print(f'Precision: {total_accuracies[0]:.4f}, Recall: {total_accuracies[1]:.4f}, F1_score: {total_accuracies[2]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
                my_wandb.save_epoch('val', epoch, optimizer.param_groups[0]['lr'], mean_losses, total_accuracies=total_accuracies)


                ''' lang별로 accuracy 구할 때 '''
                
                '''
                epoch_start = time.time()
                p_sum, r_sum, num_gt, num_det = 0.0, 0.0, 0, 0
                for lang in ['chinese', 'japanese', 'thai', 'vietnamese']:
                    gt_bboxes = get_lang_gt_bboxes(data_dir, lang)
                    pred_bboxes = get_lang_pred_bboxes(model, lang, data_dir, input_size, batch_size)
                    result = calc_deteval_metrics(pred_bboxes, gt_bboxes)
                    group_result = result['groupMetrics']
                    total_result = result['total']

                    p_sum += group_result['precisionsum']
                    r_sum += group_result['recallsum']
                    num_gt += group_result['numGt']
                    num_det += group_result['numDet']

                    precision, recall, f1_score = total_result['precision'], total_result['recall'], total_result['hmean']
                    accuracies = [precision, recall, f1_score]

                    print(f'{lang}_Precision: {accuracies[0]:.4f}, {lang}_Recall: {accuracies[1]:.4f}, {lang}_F1_score: {accuracies[2]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
                    my_wandb.save_epoch('val', epoch, optimizer.param_groups[0]['lr'], mean_losses, accuracies=accuracies, lang=lang)

                total_precision = 0 if num_det==0 else p_sum / num_det
                total_recall = 0 if num_gt==0 else r_sum / num_gt
                total_f1_score = 0 if total_precision+total_recall==0 else (2*total_precision*total_recall) / (total_precision+total_recall)
                total_accuracies = [total_precision, total_recall, total_f1_score]

                print(f'TOTAL_Precision: {total_accuracies[0]:.4f}, TOTAL_Recall: {total_accuracies[1]:.4f}, TOTAL_F1_score: {total_accuracies[2]:.4f}')
                my_wandb.save_epoch('val', epoch, optimizer.param_groups[0]['lr'], mean_losses, total_accuracies=total_accuracies)
                '''

        model_save_and_delete(total_accuracies[2], epoch) 

    my_wandb.finish()


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args('train')
    main(args)
