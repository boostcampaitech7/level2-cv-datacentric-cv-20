import os
import os.path as osp
import torch
import wandb

class MyWandb:
    def __init__(self, entity, project, name):
        self.entity = entity
        self.project = project
        self.name = name
        self.loss_names = ['loss', 'Cls loss', 'Angle loss', 'IoU loss']
        self.epoch_loss_names = ['mean loss', 'mean Cls loss', 'mean Angle loss', 'mean IoU loss']
        self.accuracy_names = ['Precision', 'Recall', 'F1_score']
        self.iter = 0

    def init(self,learning_rate, batch_size, max_epoch, image_size, input_size):
        wandb.init(project = self.project, 
                    name = self.name, 
                    entity=self.entity,
                    config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "image_size": image_size,
                "input_size": input_size,
                })
    
    def finish(self):
        wandb.finish()

    def save_iter(self, type, iter_losses, step=None):
        if step is not None:
            self.iter = step 
        else:
            self.iter += 1       
       
        wandb.log({"iter": self.iter, **dict(zip(self.prefix_loss_names(self.loss_names,type), iter_losses))})

    def save_epoch(self, type, epoch, lr, mean_losses, accuracies=None, total_accuracies=None, lang=None):
        log_dict = {"epoch": epoch + 1, "learning rate": lr, **dict(zip(self.prefix_loss_names(self.epoch_loss_names,type), mean_losses))}
        if type == 'val':
            if accuracies is not None and lang is not None:
                lang_accuracy_names = [f"{lang}_{name}" for name in self.accuracy_names]
                log_dict.update(dict(zip(lang_accuracy_names, accuracies)))
            if total_accuracies is not None:
                total_accuracy_names = [f"TOTAL_{name}" for name in self.accuracy_names]
                log_dict.update(dict(zip(total_accuracy_names, total_accuracies)))
        wandb.log(log_dict)

    def prefix_loss_names(self, loss_names, mode):        
        return [mode + "_" + name for name in loss_names]

class ModelSaveAndDelete:
    def __init__(self, model, model_dir, num_save):
        self.model = model
        self.model_dir = model_dir
        # self.best_losses = []     
        self.best_acc = [] 
        self.num_save = num_save

    def save_model(self, epoch, f1_score):
        ckpt_fpath = osp.join(self.model_dir, f'epoch_{epoch}_acc_{f1_score:.4f}.pth')
        torch.save(self.model.state_dict(), ckpt_fpath)
        print(f'Saved model for epoch {epoch} with f1_score: {f1_score:.4f}')

    def delete_model(self, worst_acc_epoch):
        worst_acc_path = osp.join(self.model_dir, f'epoch_{worst_acc_epoch[1]}_acc_{worst_acc_epoch[0]:.4f}.pth')                
        if os.path.exists(worst_acc_path):
            os.remove(worst_acc_path)
            print(f'Deleted {worst_acc_path}')

    def __call__(self, f1_score, epoch):
        epoch+=1
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if (epoch) in [10, 30, 50, 100, 150, 200]:
            self.save_model(epoch, f1_score)
        else:
            self.best_acc.append((f1_score,epoch))
            if len(self.best_acc) > self.num_save:
                self.best_acc.sort(reverse=True)
                worst_acc_epoch = self.best_acc[-1]
                self.delete_model(worst_acc_epoch)
                self.best_acc = self.best_acc[:self.num_save]

            if (f1_score, epoch) in self.best_acc:
                self.save_model(epoch, f1_score)


    # def save_model(self, epoch, mean_loss):
    #     ckpt_fpath = osp.join(self.model_dir, f'epoch_{epoch}_loss_{mean_loss:.4f}.pth')
    #     torch.save(self.model.state_dict(), ckpt_fpath)
    #     print(f'Saved model for epoch {epoch} with loss: {mean_loss:.4f}')

    # def delete_model(self, worst_loss_epoch):
    #     worst_loss_path = osp.join(self.model_dir, f'epoch_{worst_loss_epoch[1]}_loss_{worst_loss_epoch[0]:.4f}.pth')                
    #     if os.path.exists(worst_loss_path):
    #         os.remove(worst_loss_path)
    #         print(f'Deleted {worst_loss_path}')  

    # def __call__(self, mean_loss, epoch):
    #     epoch+=1
    #     if not osp.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #     if (epoch) in [10, 30, 50, 100, 150, 200]:
    #         self.save_model(epoch, mean_loss)
    #     else:
    #         self.best_losses.append((mean_loss,epoch))
    #         if len(self.best_losses) > self.num_save:
    #             self.best_losses.sort()
    #             worst_loss_epoch = self.best_losses[-1]
    #             self.delete_model(worst_loss_epoch)
    #             self.best_losses = self.best_losses[:self.num_save]

    #         if (mean_loss, epoch) in self.best_losses:
    #             self.save_model(epoch, mean_loss)