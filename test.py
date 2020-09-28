"""
Use this python script to apply semantic segmentation any images
of your choice.

USAGE: python test.py --input <path to image> --weights <path to saved checkpoint/weight file>
"""

import torchvision
import numpy
import torch
import argparse
import cv2
import torchvision.transforms as transforms

from utils.helpers import draw_test_segmentation_map, image_overlay
from PIL import Image
from model import model

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, 
                    help='path to input image')
parser.add_argument('-w', '--weights', required=True, 
                    help='path to the trained weight file')
args = vars(parser.parse_args())

# image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# load the model
model = model.to(device)
checkpoint = torch.load(args['weights'])
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
# set the model to eval model
model.eval()

# read the image
image = Image.open(args['input'])
# keep a copy of the original image
orig_image = image.copy()
# apply image transform
image = transform(image).to(device)
# add an extra batch dimension
image = image.unsqueeze(0)
# forward pass
outputs = model(image)
# get the output tensors
outputs = outputs['out']
# draw the segmentation map on top of the original image
segmented_image = draw_test_segmentation_map(outputs)
# get the final output
final_image = image_overlay(orig_image, segmented_image)

save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# show the segmented image and save to disk
cv2.imshow('Segmented image', final_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{save_name}.jpg", final_image)