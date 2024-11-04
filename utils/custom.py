import os
import os.path as osp
import torch
import wandb

class MyWandb:
    def __init__(self, project, name):
        self.project = project
        self.name = name
        self.loss_names = ['loss', 'Cls loss', 'Angle loss', 'IoU loss']
        self.epoch_loss_names = ['mean loss', 'mean Cls loss', 'mean Angle loss', 'mean IoU loss']
        self.iter = 0
        
    def init(self,learning_rate, batch_size, max_epoch, image_size, input_size):
        wandb.init(project = self.project, name = self.name, config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "image_size": image_size,
        "input_size": input_size,
         })
    
    def finish():
        wandb.finish()

    def save_iter(self,iter_losses):
        self.iter +=1
        wandb.log({"iter": self.iter, **dict(zip(self.loss_names, iter_losses))})

    def save_epoch(self,epoch,lr,mean_losses):
        wandb.log({"epoch": epoch + 1, "learning rate": lr, **dict(zip(self.epoch_loss_names, mean_losses))})

class ModelSaveAndDelete:
    def __init__(self, model, model_dir, num_save):
        self.model = model
        self.model_dir = model_dir
        self.best_losses = []      
        self.num_save = num_save
    def save_model(self, epoch, mean_loss):
        ckpt_fpath = osp.join(self.model_dir, f'epoch_{epoch}_loss_{mean_loss:.4f}.pth')
        torch.save(self.model.state_dict(), ckpt_fpath)
        print(f'Saved model for epoch {epoch} with loss: {mean_loss:.4f}')

    def delete_model(self, worst_loss_epoch):
        worst_loss_path = osp.join(self.model_dir, f'epoch_{worst_loss_epoch[1]}_loss_{worst_loss_epoch[0]:.4f}.pth')                
        if os.path.exists(worst_loss_path):
            os.remove(worst_loss_path)
            print(f'Deleted {worst_loss_path}')  

    def __call__(self, mean_loss, epoch):
        epoch+=1
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if (epoch) in [10, 30, 50, 100, 150, 200]:
            self.save_model(epoch, mean_loss)
        else:
            self.best_losses.append((mean_loss,epoch))
            if len(self.best_losses) > self.num_save:
                self.best_losses.sort()
                worst_loss_epoch = self.best_losses[-1]
                self.delete_model(worst_loss_epoch)
                self.best_losses = self.best_losses[:self.num_save]

            if (mean_loss, epoch) in self.best_losses:
                self.save_model(epoch, mean_loss)
