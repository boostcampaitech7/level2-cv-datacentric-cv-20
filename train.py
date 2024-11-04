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
from data_loader.dataset import SceneTextDataset
from data_loader.transform import get_train_transform

from utils.custom import *

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):

    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        transforms=get_train_transform()
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    model_save_and_delete = ModelSaveAndDelete(model, model_dir,3)     

    my_wandb = MyWandb('ocr baseline', 'baselineMaxEpoch200')
    loss_names = my_wandb.loss_names
    my_wandb.init(learning_rate, batch_size, max_epoch, image_size, input_size)

    for epoch in range(max_epoch):
        epoch_losses = [0.,0.,0.,0.]
        epoch_start = time.time()   
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:

                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                iter_losses = [loss_val, extra_info['cls_loss'], extra_info['angle_loss'], extra_info['iou_loss']]
                for i in range(len(epoch_losses)):
                    epoch_losses[i] += iter_losses[i]

                pbar.update(1)
                val_dict = dict(zip(loss_names, iter_losses))
                pbar.set_postfix(val_dict)
                my_wandb.save_iter(iter_losses)   

        scheduler.step()
        mean_losses = [loss / num_batches for loss in epoch_losses]
        print(f'Mean loss: {mean_losses[0]:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')
        my_wandb.save_epoch(epoch,optimizer.param_groups[0]['lr'],mean_losses)
        model_save_and_delete(mean_losses[0], epoch)     
    my_wandb.finish()

def main(args):
    do_training(**args)

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/default.yaml"
    args = load_config(config_path)['train']
    main(args)