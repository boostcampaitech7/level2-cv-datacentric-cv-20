import albumentations as A

def get_train_transform():
    return A.Compose([
        A.ColorJitter(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

def get_val_transform():
    return A.Compose([
        A.ColorJitter(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])