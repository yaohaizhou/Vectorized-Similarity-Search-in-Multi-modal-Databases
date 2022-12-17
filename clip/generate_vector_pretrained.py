import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
from PIL import Image
import os
from torch.utils.data import DataLoader
import pdb
import clip
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
# model = torch.load("new_80000_new_model_4.pth")

NUMEXAMPLE = 80000

class image_text_dataset(Dataset):
    def __init__(self, mode, image_path, annotation_file, num_examples):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        self.captions = []
        self.img_names = []
        for annot in annotations['annotations']:
            caption = '[SOS] ' + annot['caption'] + ' [EOS]'
            image_id = annot['image_id']
            full_coco_image_path = image_path + 'COCO_'+mode+'2014_' + '%012d.jpg' % (image_id)
            
            self.img_names.append(full_coco_image_path)
            self.captions.append(caption)
        print(mode+" captions size : "+str(len(self.captions)))
        self.img_names, self.captions = shuffle(self.img_names,self.captions,random_state=1)
        self.num_examples = num_examples
        self.img_names = self.img_names[:self.num_examples]
        self.captions = self.captions[:self.num_examples]
        self.captions_token = clip.tokenize(self.captions)
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # image = self.img_names[idx]
        # caption = self.captions[idx]
        image = preprocess(Image.open(self.img_names[idx]).convert("RGB"))
        caption = self.captions_token[idx]
        return image,caption

#####DATASET
CURDIR = os.path.abspath('.')
dataset = image_text_dataset(mode='train',image_path=CURDIR+'/train2014/', annotation_file=CURDIR+'/annotations/captions_train2014.json',num_examples=NUMEXAMPLE)
dataset_val = image_text_dataset(mode='val',image_path=CURDIR+'/val2014/', annotation_file=CURDIR+'/annotations/captions_val2014.json',num_examples=NUMEXAMPLE)


data = []
for i in tqdm(range(len(dataset.img_names))):
    image = Image.open(dataset.img_names[i]).convert("RGB")
    text = dataset.captions[i]
    with torch.no_grad():
        image_feature = model.encode_image(torch.tensor(preprocess(image)).cuda().unsqueeze(0)).float()
        text_token = clip.tokenize(text).cuda()
        text_feature = model.encode_text(text_token).float()
    data.append({'name':dataset.img_names[i].split('/')[-1],'text':text,'image_feature':image_feature.cpu().detach().numpy(),'text_feature':text_feature.cpu().detach().numpy()})
output = open("full_train.pkl", 'wb')
pickle.dump(data,output)
output.close()

data = []
for i in tqdm(range(len(dataset_val.img_names))):
    image = Image.open(dataset_val.img_names[i]).convert("RGB")
    text = dataset_val.captions[i]
    with torch.no_grad():
        image_feature = model.encode_image(torch.tensor(preprocess(image)).cuda().unsqueeze(0)).float()
        text_token = clip.tokenize(text).cuda()
        text_feature = model.encode_text(text_token).float()
    data.append({'name':dataset_val.img_names[i].split('/')[-1],'text':text,'image_feature':image_feature.cpu().detach().numpy(),'text_feature':text_feature.cpu().detach().numpy()})
output = open("val.pkl", 'wb')
pickle.dump(data,output)
output.close()

