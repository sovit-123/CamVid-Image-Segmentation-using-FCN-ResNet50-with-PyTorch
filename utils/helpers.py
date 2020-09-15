import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

label_colors_list = [
        (64, 128, 64), # animal
        (192, 0, 128), # archway
        (0, 128, 192), # bicyclist
        (0, 128, 64), #bridge
        (128, 0, 0), # building
        (64, 0, 128), #car
        (64, 0, 192), # car luggage pram...???...
        (192, 128, 64), # child
        (192, 192, 128), # column pole
        (64, 64, 128), # fence
        (128, 0, 192), # lane marking driving
        (192, 0, 64), # lane maring non driving
        (128, 128, 64), # misc text
        (192, 0, 192), # motor cycle scooter
        (128, 64, 64), # other moving
        (64, 192, 128), # parking block
        (64, 64, 0), # pedestrian
        (128, 64, 128), # road
        (128, 128, 192), # road shoulder
        (0, 0, 192), # sidewalk
        (192, 128, 128), # sign symbol
        (128, 128, 128), # sky
        (64, 128, 192), # suv pickup truck
        (0, 0, 64), # traffic cone
        (0, 64, 64), # traffic light
        (192, 64, 128), # train
        (128, 128, 0), # tree
        (192, 128, 192), # truck/bus
        (64, 0, 64), # tunnel
        (192, 192, 0), # vegetation misc.
        (0, 0, 0),  # 0=background/void
        (64, 192, 0), # wall
    ]

# all the classes that are present in the dataset
ALL_CLASSES = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving', 
        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk', 
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall']

def get_label_mask(mask, class_values): 
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def draw_seg_maps(output, label_colors_nparray, epoch, i):
    """
    This function color codes the segmentation maps that is generated while
    validating (if the user passes the argument while executing 
    `train.py`).
    """
    sample = output[0] # use only one output from the batch
    sample = torch.argmax(sample.squeeze(), dim=0).detach().cpu().numpy()

    r = np.zeros_like(sample).astype(np.uint8)
    g = np.zeros_like(sample).astype(np.uint8)
    b = np.zeros_like(sample).astype(np.uint8)
    
    for l in range(0, 32):
        idx = sample == l
        r[idx] = np.array(label_colors_list)[l, 0]
        g[idx] = np.array(label_colors_list)[l, 1]
        b[idx] = np.array(label_colors_list)[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    plt.imshow(rgb)
    plt.axis('off')
    plt.savefig(f"train_seg_maps/e{epoch}_b{i}.jpg")
    plt.close()


def visualize_from_dataloader(data_loader): 
    """
    Helper function to visualzie the data from 
    dataloaders
    """
    data = iter(data_loader)   
    images, labels = data.next()
    # print(images.size(), labels.size())
    image = images[1]
    label = labels[1]
    image = np.array(image, dtype='uint8')
    image = np.transpose(image, (1, 2, 0))
    images = [image, label.squeeze()]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image, cmap='gray')
    plt.show()

def visualize_from_path(image_path, seg_path):
    train_sample_img = cv2.imread(image_path[0])
    train_sample_img = cv2.cvtColor(train_sample_img, cv2.COLOR_BGR2RGB)
    train_sample_seg = cv2.imread(seg_path[0])
    train_sample_seg = cv2.cvtColor(train_sample_seg, cv2.COLOR_BGR2RGB)
    images = [train_sample_img, train_sample_seg]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image)
    plt.show()

def save_model_dict(model, epochs, optimizer, criterion):
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"checkpoints/model_{epochs}.pth")
