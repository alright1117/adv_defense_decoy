import cv2
import os
import random
import glob

import torch
from torch.utils.data import Dataset

class TrainFaceForensics(Dataset):
    def __init__(self, data_name, path, img_batch, input_size, transform = None):
        super().__init__()  
        self.data_path = os.path.join(path, data_name)
        self.fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self.video_list = os.listdir(os.path.join(self.data_path, 'Real'))
        self.img_batch = img_batch
        self.input_size = input_size
        self.transform = transform


    def __len__(self):
        return len(self.video_list)
    

    def __getitem__(self, index):
        video_name = self.video_list[index]
        all_img, labels = self._read_image(video_name)
            
        return all_img, labels
    

    def _read_image(self, video_name):
        real_path = os.path.join(self.data_path, 'Real', video_name)
        img_name =  glob.glob(os.path.join(real_path, '*.png'))
        sample_img = random.sample(img_name, self.img_batch)
        true_img = torch.zeros((self.img_batch, 3, self.input_size, self.input_size))
        fake_img = torch.zeros((self.img_batch, 3, self.input_size, self.input_size))
        
        for idx in range(len(sample_img)):
            img_path = os.path.join(sample_img[idx])
            img = cv2.imread(img_path)

            if self.transform is not None:
                img = self.transform(img)
                
            true_img[idx] = img[:]
        
        count = 0
        for init, fake_name in enumerate(self.fake_list):
            fake_path = os.path.join(self.data_path, fake_name, video_name + '*')
            fake_path = glob.glob(fake_path)[0]
            img_name =  glob.glob(os.path.join(fake_path, '*.png'))
            sample_size = self.img_batch // len(self.fake_list)
            sample_img = random.sample(img_name, sample_size)
            for idx in range(len(sample_img)):
                img_path = os.path.join(sample_img[idx])
                img = cv2.imread(img_path)
                if img is None:
                    print(img_path)
                if self.transform is not None:
                    img = self.transform(img)

                fake_img[count] = img[:]
                count += 1
        
        all_img = torch.cat((true_img,fake_img))

        labels = torch.cat((torch.zeros(self.img_batch, 1), torch.ones(self.img_batch, 1))).type(torch.LongTensor)
        
        return all_img, labels


class TrainFaceForensicsDeception(Dataset):
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