import cv2
import os
import random
import glob

import torch
from torch.utils.data import Dataset

class TestFaceForensics(Dataset):
    def __init__(self, root_dir, file_path, img_batch, input_size, transform = None):
        super().__init__()  
        self.root_dir = root_dir
        with open(file_path, "r") as f:
            lines = f.read()
            self.video_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(self.video_list)
        self.img_batch = img_batch
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_name, label = self.video_list[index]
        all_img, labels, label_name = self._read_image(video_name, label)
            
        return all_img, labels, label_name
    
    def _read_image(self, video_name, label):
        random.seed(1)
        img_path = os.path.join(self.root_dir, video_name, '*.png')
        img_list = glob.glob(img_path)

        label_name = video_name.split('/')[-2]

        sample_img = random.sample(img_list, self.img_batch)

        img_tensor = torch.zeros((self.img_batch, 3, self.input_size, self.input_size))
        
        for idx, img_path in enumerate(sample_img):
            img = cv2.imread(img_path)

            if self.transform is not None:
                img = self.transform(img)
            img_tensor[idx] = img[:]
        
        if label == '1':
            labels = torch.ones(self.img_batch, 1).type(torch.LongTensor)
        else:
            labels = torch.zeros(self.img_batch, 1).type(torch.LongTensor)

        return img_tensor, labels, label_name