import glob
import config
import albumentations
import cv2
import numpy as np
import torch

from utils.helpers import label_colors_list, get_label_mask
from utils.helpers import ALL_CLASSES, visualize_from_path
from utils.helpers import visualize_from_dataloader
from torch.utils.data import Dataset, DataLoader

train_images = glob.glob(f"{config.ROOT_PATH}/train/*")
train_images.sort()
train_segs = glob.glob(f"{config.ROOT_PATH}/train_labels/*")
train_segs.sort()
valid_images = glob.glob(f"{config.ROOT_PATH}/val/*")
valid_images.sort()
valid_segs = glob.glob(f"{config.ROOT_PATH}/val_labels/*")
valid_images.sort()

if config.DEBUG:
    visualize_from_path(train_images, train_segs)

class CamVidDataset(Dataset):
    CLASSES = ALL_CLASSES

    def __init__(self, path_images, path_segs, transform, label_colors_list, classes):
        print(f"TRAINING ON CLASSES: {classes}")

        self.path_images = path_images
        self.path_segs = path_segs
        self.label_colors_list = label_colors_list
        self.transform = transform
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
    def __len__(self):
        return len(self.path_images)
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.path_segs[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
          
        ##### THIS WORKS TOO #####
        """
        image = np.array(Image.open(self.path_images[index]).convert('RGB'))
        mask = np.array(Image.open(self.path_segs[index]).convert('RGB'))
        """
        #####     ######     #####               
        
        image = self.transform(image=image)['image']
        mask = self.transform(image=mask)['image']
        
        # get the colored mask labels
        mask = get_label_mask(mask, self.class_values)
       
        image = np.transpose(image, (2, 0, 1))
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(mask, dtype=torch.long) 

transform = albumentations.Compose([
    albumentations.Resize(224, 224, always_apply=True),
])
        
train_dataset = CamVidDataset(train_images, train_segs, transform, 
                              label_colors_list, 
                              classes=config.CLASSES_TO_TRAIN)
valid_dataset = CamVidDataset(valid_images, valid_segs, transform,
                              label_colors_list, 
                              classes=config.CLASSES_TO_TRAIN)

train_data_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
valid_data_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)

if config.DEBUG:
    visualize_from_dataloader(train_data_loader)