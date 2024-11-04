import os
import os.path as osp
import time
import math
import yaml
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from base.east_dataset import EASTDataset
from base.model import EAST
from data_loader.dataset import SceneTextDataset, MergedDataset

from data_loader.transform import get_train_transform, get_val_transform

from utils.custom import *

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--dataset', type=str, choices=['Base', 'CORD'])
    parser.add_argument('--valid', type=bool, default=True)

    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(dataset, valid, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, resume):

    '''
    train_dataset
    '''
    # dataset = SceneTextDataset(
    #     data_dir,
    #     split='train',
    #     image_size=image_size,
    #     crop_size=input_size,
    #     transforms=get_train_transform()
    # )
    train_dataset = MergedDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        transforms=get_train_transform()
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

    '''
    val_dataset
    '''
    val_dataset = MergedDataset(
        data_dir,
        split='val',
        image_size=image_size,
        crop_size=input_size,
        transforms=get_val_transform()
    )
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model_save_and_delete = ModelSaveAndDelete(model, model_dir,3)     
    my_wandb = MyWandb('kaeh3403-personal', 'Data-Centric', 'cord_dataset')
    loss_names = my_wandb.loss_names
    my_wandb.init(learning_rate, batch_size, max_epoch, image_size, input_size)

    '''
    Resume 
    '''
    if args.resume != None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    train_step, val_step = 0, 0  # 초기화

    for epoch in range(max_epoch):
        # ======== train =========
        model.train()
        train_epoch_losses = [0., 0., 0., 0.]
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
        print(f'Mean Train loss: {mean_losses[0]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
        
        my_wandb.save_epoch('train', epoch, optimizer.param_groups[0]['lr'], mean_losses)
        
        # ======== validation =========   
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
                    # Iteration 단위로 val_step 증가 및 기록
                    my_wandb.save_iter('val', iter_losses, val_step)
                    val_step += 1

        mean_losses = [loss / val_num_batches for loss in val_epoch_losses]
        print(f'Mean Validation loss: {mean_losses[0]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
    
        my_wandb.save_epoch('val', epoch, optimizer.param_groups[0]['lr'], mean_losses)
        model_save_and_delete(mean_losses[0], epoch) 

    my_wandb.finish()

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)