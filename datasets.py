from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torch
import numpy as np
import os

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CelebaBinaryCalssificationPairwise(Dataset):
    def __init__(self, images, attributes_list, annots, class_name):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
        self.class_name = class_name
        self.positives_indexes = []
        self.negative_indexes = []
        self.opposite_indexes = []
        self.score = []
        for i in range(len(self.images)):
            im_name = self.images[i]
            target = self.annots[im_name.split('/')[-1]][self.attributes_list.index(self.class_name)]
            target = int((target + 1) // 2)
            if target == 1:
                self.positives_indexes.append(i)
            else:
                self.negative_indexes.append(i)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):  
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index(self.class_name)]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2 # celeba sizes
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)
        image = transform(image).float()
        target = int((target + 1) // 2)
        if target == 1:
            if self.opposite_indexes:
                idx_opposite = self.opposite_indexes[idx]
            else:
                idx_opposite = np.random.choice(self.negative_indexes)
            image_name_opposite = self.images[idx_opposite]
            image_opposite = np.zeros((256, 256, 3), dtype = np.uint8)
            image_opposite[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(image_name_opposite)
            image_opposite = transform(image_opposite).float()
            return image, image_opposite, target, idx, idx_opposite
        else:
            if self.opposite_indexes:
                idx_opposite = self.opposite_indexes[idx]
            else:
                idx_opposite = np.random.choice(self.positives_indexes)
            image_name_opposite = self.images[idx_opposite]
            image_opposite = np.zeros((256, 256, 3), dtype = np.uint8)
            image_opposite[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(image_name_opposite)
            image_opposite = transform(image_opposite).float()
            return image_opposite, image, target, idx, idx_opposite

class CelebaSegmentation(Dataset):
    def __init__(self, images, class_name):
        self.images = sorted(images)
        self.class_name = class_name   
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        num = im_name.split('/')[-1].split('.')[0]
        num = int(num)
        # check
        segm_name = "/home/dmartynov/shad/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_" + self.class_name + ".png"
        image = cv2.imread(im_name)  
        image = transform(image).float()
        if os.path.isfile(segm_name):
            segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        else:
            segm_image = torch.zeros((1, 256, 256))
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()
        return image, segm_image
