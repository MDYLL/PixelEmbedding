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
            # print (1, idx, idx_opposite)
            image_name_opposite = self.images[idx_opposite]
            image_opposite = np.zeros((256, 256, 3), dtype = np.uint8)
            image_opposite[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(image_name_opposite)
            image_opposite = transform(image_opposite).float()
            # print(idx, idx_opposite)
            return image, image_opposite, target, idx, idx_opposite
        else:
            if self.opposite_indexes:
                idx_opposite = self.opposite_indexes[idx]
            else:
                idx_opposite = np.random.choice(self.positives_indexes)
            # print (0, idx, idx_opposite)
            image_name_opposite = self.images[idx_opposite]
            image_opposite = np.zeros((256, 256, 3), dtype = np.uint8)
            image_opposite[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(image_name_opposite)
            image_opposite = transform(image_opposite).float()
            # print(idx_opposite, idx)
            return image_opposite, image, target, idx, idx_opposite

class CelebaBinaryCalssification(Dataset):
    def __init__(self, images, attributes_list, annots, class_name, transform=False):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
        self.class_name = class_name
        self.transfom = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index(self.class_name)]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2 # celeba sizes
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)
        if self.transfom:
             image = transform1(image).float()
        else:
            image = transform(image).float()
        return image, int((target + 1) // 2)

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
            segm_image = None #TODO remove
        if segm_image is None: # file not found == no mask
            segm_image = torch.zeros((1, 256, 256))
            # print("no mask")
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()
        return image, segm_image

# LEGACY
class CelebaMale(Dataset):
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Male')]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)       
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaSmile(Dataset):
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Smiling')]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)     
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaYoung(Dataset): # derm3.py
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Young')]      
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)    
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaIdentity(Dataset): # derm5.ipynb
    def __init__(self, images, identity):
        self.images = sorted(images)    
        self.identity = identity
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        ident =  self.identity[im_name.split('/')[-1]]        
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)       
        image = transform(image).float()
        return image, ident

class CelebaLandmarks(Dataset): # derm9.ipynb
    def __init__(self, images, landmarks):
        self.images = sorted(images)    
        self.landmarks = landmarks
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        im_landmarks = self.landmarks[im_name.split('/')[-1]]        
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)        
        image = transform(image).float()
        return image, np.float32(im_landmarks / 256.)

class CelebaNose(Dataset):
    def __init__(self, images):
        self.images = sorted(images)   
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        # print(im_name, idx)
        num = im_name.split('/')[-1].split('.')[0]
        num = int(num)
        segm_name = "/root/dmartynov/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_nose.png"
        image = cv2.imread(im_name)  
        image = transform(image).float()
        try:
            segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        except:
            segm_image = None
        if segm_image is None: # file not found == no nose
            segm_image = torch.zeros((1, 256, 256))
            print("no nose")
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()
        return image, segm_image

class CelebaSkin(Dataset):
    def __init__(self, images):
        self.images = sorted(images)   
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        # print(im_name, idx)
        num = im_name.split('/')[-1].split('.')[0]
        num = int(num)
        segm_name = "/root/dmartynov/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_skin.png"
        image = cv2.imread(im_name)  
        image = transform(image).float()
        try:
            segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        except:
            segm_image = None
        if segm_image is None: # file not found == no nose
            segm_image = torch.zeros((1, 256, 256))
            print("no skin")
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()
        return image, segm_image

