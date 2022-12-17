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

EPOCH = 5
BATCHSIZE = 32
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
train_dataloader = DataLoader(dataset,batch_size = BATCHSIZE, shuffle=True,num_workers=4)

# dataset_val = image_text_dataset(mode='val',image_path=CURDIR+'/val2014/', annotation_file=CURDIR+'/annotations/captions_val2014.json',num_examples=NUMEXAMPLE)
# val_dataloader = DataLoader(dataset_val,batch_size = 2*BATCHSIZE, shuffle=True,num_workers=4)
# pdb.set_trace()
######TRAINING
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-8,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
loss_list = []

######FUNCTIONS
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

# ######TEST FIRST
# n_correct = 0
# with torch.no_grad():
#     model.eval()
#     for batch in tqdm(val_dataloader):
#         images,texts = batch 
#         images= images.to(device)
#         texts = texts.to(device)
#         logits_per_image, logits_per_text = model(images, texts)
#         # pdb.set_trace()
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#         ground_truth = np.arange(len(images))
#         n_correct += np.sum(np.argmax(probs,axis=1)==ground_truth)

#     print("Initial Val Precision: "+str(n_correct/NUMEXAMPLE))


for epoch in range(EPOCH):
    print("EPOCH: ",epoch)

    model.train()
    for i, batch in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()

        images,texts = batch 

        images= images.to(device)
        texts = texts.to(device)
        # pdb.set_trace()
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()

        # convert_models_to_fp32(model)
        optimizer.step()
        # clip.model.convert_weights(model)

        if i % 20 == 0:
            print(total_loss)
            loss_list.append(total_loss.cpu().detach().numpy())
    
    n_correct = 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            images,texts = batch 
            images= images.to(device)
            texts = texts.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            # pdb.set_trace()
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            ground_truth = np.arange(len(images))
            n_correct += np.sum(np.argmax(probs,axis=1)==ground_truth)

        print("Val Precision: "+str(n_correct/NUMEXAMPLE))

    torch.save(model, "new_"+str(NUMEXAMPLE)+"_new_model_"+str(epoch)+".pth")
    np.save("new_"+str(NUMEXAMPLE)+"_loss_"+str(epoch)+".npy",np.array(loss_list))
    data = []
    for i in tqdm(range(len(dataset.img_names))):
        image = Image.open(dataset.img_names[i]).convert("RGB")
        text = dataset.captions[i]
        with torch.no_grad():
            image_feature = model.encode_image(torch.tensor(preprocess(image)).cuda().unsqueeze(0)).float()
            text_token = clip.tokenize(text).cuda()
            text_feature = model.encode_text(text_token).float()
        data.append({'name':dataset.img_names[i].split('/')[-1],'text':text,'image_feature':image_feature.cpu().detach().numpy(),'text_feature':text_feature.cpu().detach().numpy()})
    output = open("new_"+str(NUMEXAMPLE)+"_finetune_"+str(epoch)+".pkl", 'wb')
    pickle.dump(data,output)
    output.close()
