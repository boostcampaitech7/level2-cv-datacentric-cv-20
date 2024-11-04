import time
import math
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

from util.custom import *

def parse_args():
    parser = ArgumentParser()

    # Conventional args
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
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


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
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)