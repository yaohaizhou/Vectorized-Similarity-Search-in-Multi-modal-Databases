import os
import cv2

# import faiss
import pickle
import requests
import pathlib

# current working directory
# print(pathlib.Path().absolute())
import time
import streamlit as st

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
from io import StringIO
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import clip
import pdb
from knn import get_topk_similar_indices
from faiss_ann import *
import pyheif

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_knn = torch.device("cpu")
MORE_IMAGE_TIMES = 2

@st.cache
def load_models():
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model_finetune = torch.load("/home/ubuntu/zyh/result/new_80000_new_model_4.pth", map_location="cpu")
    return model, preprocess, model_finetune


@st.cache(allow_output_mutation=True)
def load_embeddings():
    STORE_DIR = "/home/ubuntu/zyh/result/"
    name = np.load(STORE_DIR+"train_name.npy")
    text = np.load(STORE_DIR+"train_text.npy")
    text_embeddings_pretrained = torch.FloatTensor(np.load(STORE_DIR+"train_text_feature.npy"))
    image_embeddings_pretrained = torch.FloatTensor(np.load(STORE_DIR+"train_image_feature.npy"))
    text_embeddings_finetuned = torch.FloatTensor(np.load(STORE_DIR+"train_finetune_text_feature.npy"))
    image_embeddings_finetuned = torch.FloatTensor(np.load(STORE_DIR+"train_finetune_image_feature.npy"))
    return (
        name,
        text,
        text_embeddings_pretrained.squeeze(1),
        image_embeddings_pretrained.squeeze(1),
        text_embeddings_finetuned.squeeze(1),
        image_embeddings_finetuned.squeeze(1),
    )

@st.cache(allow_output_mutation=True)
def load_faiss_index(
        # text_embeddings_pretrained,
        # image_embeddings_pretrained,
        # text_embeddings_finetuned,
        # image_embeddings_finetuned,
        # nlist = 100,
        # nprobe = 10,
        method = "cosine", 
    ):
    
    # text_embeddings_pretrained = text_embeddings_pretrained.cpu().detach().numpy()
    # image_embeddings_pretrained = image_embeddings_pretrained.cpu().detach().numpy()
    # text_embeddings_finetuned = text_embeddings_finetuned.cpu().detach().numpy()
    # image_embeddings_finetuned = image_embeddings_finetuned.cpu().detach().numpy()
    
    # text_faiss_index_pretrained = \
    #     create_index(
    #             text_embeddings_pretrained, 
    #             nlist, 
    #             nprobe, 
    #             method,
    #         )
    # image_faiss_index_pretrained = \
    #     create_index(
    #             image_embeddings_pretrained, 
    #             nlist, 
    #             nprobe, 
    #             method,
    #         )
    # text_faiss_index_finetuned = \
    #     create_index(
    #             text_embeddings_finetuned, 
    #             nlist, 
    #             nprobe, 
    #             method,
    #         )
    # image_faiss_index_finetuned = \
    #     create_index(
    #             image_embeddings_finetuned, 
    #             nlist, 
    #             nprobe, 
    #             method,
    #         )
    # faiss.write_index(text_faiss_index_pretrained, "index/text_faiss_index_pretrained_"+method+".index")
    # faiss.write_index(image_faiss_index_pretrained, "index/image_faiss_index_pretrained_"+method+".index")
    # faiss.write_index(text_faiss_index_finetuned, "index/text_faiss_index_finetuned_"+method+".index")
    # faiss.write_index(image_faiss_index_finetuned, "index/image_faiss_index_finetuned_"+method+".index")
    text_faiss_index_pretrained = faiss.read_index("index/text_faiss_index_pretrained_"+method+".index")
    image_faiss_index_pretrained = faiss.read_index("index/image_faiss_index_pretrained_"+method+".index")
    text_faiss_index_finetuned = faiss.read_index("index/text_faiss_index_finetuned_"+method+".index")
    image_faiss_index_finetuned = faiss.read_index("index/image_faiss_index_finetuned_"+method+".index")
    return (
        text_faiss_index_pretrained,
        image_faiss_index_pretrained,
        text_faiss_index_finetuned,
        image_faiss_index_finetuned,
    )

# Function define
# Input text to vector
@torch.no_grad()
def text2vector(model, text):
    text_token = clip.tokenize(text).to(device)
    text_feature = model.encode_text(text_token).float()
    return text_feature


# Input image to vector
@torch.no_grad()
def image2vector(model, image):
    image_feature = model.encode_image(
        preprocess(image).clone().detach().to(device).unsqueeze(0)
    ).float()
    # image_feature = model.encode_image(
    #     torch.tensor(preprocess(image)).to(device).unsqueeze(0)
    # ).float()
    return image_feature


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# pdb.set_trace()
# load models
model, preprocess, model_finetune = load_models()
model_finetune = model_finetune.to(device)

# load embeddings
(
    name,
    text,
    text_embeddings_pretrained,
    image_embeddings_pretrained,
    text_embeddings_finetuned,
    image_embeddings_finetuned,
) = load_embeddings()

# sidebar
st.sidebar.title("Please check the boxes before generating")

# Search Strategy Options
st.sidebar.subheader("Search Strategy Options")
search_method = st.sidebar.selectbox("Please select Search Strategy Options",("KNN", "ANN"),label_visibility="collapsed")

# Distance Options
st.sidebar.subheader("Distance Options")
dist_method = st.sidebar.selectbox("Please select distance function",("cosine", "l2", "dot"),label_visibility="collapsed")


if(search_method=='ANN'):
    # load faiss index
    if(dist_method=='l2'):
        (
            text_faiss_index_pretrained_l2,
            image_faiss_index_pretrained_l2,
            text_faiss_index_finetuned_l2,
            image_faiss_index_finetuned_l2,
        ) = load_faiss_index(
                # text_embeddings_pretrained,
                # image_embeddings_pretrained,
                # text_embeddings_finetuned,
                # image_embeddings_finetuned,
                # 100,
                # 10,
                "l2", 
            )
    elif(dist_method=='dot'):
        (
            text_faiss_index_pretrained_dot,
            image_faiss_index_pretrained_dot,
            text_faiss_index_finetuned_dot,
            image_faiss_index_finetuned_dot,
        ) = load_faiss_index(
                # text_embeddings_pretrained,
                # image_embeddings_pretrained,
                # text_embeddings_finetuned,
                # image_embeddings_finetuned,
                # 100,
                # 10,
                "dot", 
            )
    else:
        (
            text_faiss_index_pretrained_cosine,
            image_faiss_index_pretrained_cosine,
            text_faiss_index_finetuned_cosine,
            image_faiss_index_finetuned_cosine,
        ) = load_faiss_index(
                # text_embeddings_pretrained,
                # image_embeddings_pretrained,
                # text_embeddings_finetuned,
                # image_embeddings_finetuned,
                # 100,
                # 10,
                "cosine", 
            )

# time options
st.sidebar.subheader("Time Consuming")
Time = st.sidebar.checkbox("show",value=True)
# if time:
# Finetune Option
st.sidebar.subheader("Finetune")
finetune = st.sidebar.checkbox("Finetune",value=True)

#####CPU ONLY#####
if(finetune and device == torch.device("cpu")):
    model_finetune = model_finetune.float()

# K value
st.sidebar.subheader("K Nearest Neighbor")
k= st.sidebar.number_input(min_value=1, max_value=15,value=5, step=1, label="Input the value of K (min:1, max:15)" )

# if third:
st.title("Upload Image")
uploaded_file1 = st.file_uploader("", type=["jpg","png","jpeg","heic"])

# predict_caption(image)
#   st.success("Click again to retry or try a different image by uploading")
#  st.balloons()

# # predict_image(captions)
# #   st.success("Click again to retry or try a different image by uploading")
# #   st.balloons()
# st.title("Input image to search for image")
# # uploaded_file2=st.file_uploader("", type="jpg")
st.title("Input Text")
caption_input = st.text_input("Input the text you want to search")

st.title("Image to Text")
if st.button("Generate captions from image!"):
    start_i2t = time.time()
    # Step 1.
    # forward model to get embedding
    target_model = model_finetune if finetune else model
    # st.markdown(uploaded_file1.type)
    if(uploaded_file1.type=="image/heic"):
        image = pyheif.read(uploaded_file1)
        image = Image.frombytes(mode=image.mode, size=image.size, data=image.data)
    else:
        image = Image.open(uploaded_file1).convert("RGB")
    image = ImageOps.exif_transpose(image)
    
    st.image(image)
    
    image_feature = image2vector(target_model, image)
    # image_feature = image_feature.to(device_knn)
    # st.markdown(image_feature) # device='cuda:0'

    # Step 2.
     # get top-k similar items
    if(search_method=='KNN'):
        candidate_features = text_embeddings_finetuned if finetune else text_embeddings_pretrained
        candidate_features = candidate_features.to(device)
        similarities, indices = get_topk_similar_indices(image_feature, candidate_features, k, dist_method)
        # candidate_features = candidate_features.cpu()
        # release gpu ram
        # useless since candidate_features is cached in st
        # st.markdown(torch.cuda.memory_summary())
        # print(torch.cuda.memory_summary())
        del candidate_features
        torch.cuda.empty_cache()
    
    if(search_method=='ANN'):
        ## zoey test for faiss ann
        if dist_method == "l2":
            candidate_faiss_index = text_faiss_index_finetuned_l2 if finetune else text_faiss_index_pretrained_l2
        elif dist_method == "dot":
            candidate_faiss_index = text_faiss_index_finetuned_dot if finetune else text_faiss_index_pretrained_dot
        else:
            candidate_faiss_index = text_faiss_index_finetuned_cosine if finetune else text_faiss_index_pretrained_cosine
        # st.markdown(type(candidate_faiss_index))
        similarities,indices = get_topk_similar_indices_faiss_ivf(image_feature.cpu().detach().numpy(), 
                                                                candidate_faiss_index, 
                                                                k)
    # Step 3.
    # show top-k similar items
    for i in range(len(indices)):

        st.markdown(str(i+1)+". "+"Similarity: "+str(similarities[i]))
        st.markdown(text[indices[i]][6:-6])
    end_i2t = time.time()
    if Time:

        st.write((round((end_i2t-start_i2t)*1000)), "ms for executing")
        #st.markdown('The time of execution of above program is :'
        #(end_t2t-start_t2t) * 10**3, 'ms')
    
    # release gpu ram
    del image_feature
    torch.cuda.empty_cache()

st.title("Text to Image")
if st.button("Generate images from text!"):
    start_t2i = time.time()
    # Step 1.
    # forward model to get embedding
    target_model = model_finetune if finetune else model
    text_feature = text2vector(target_model, caption_input)
    # text_feature = text_feature.to(device_knn)
    
    # st.markdown(text_feature)

    # Step 2.
    # get top-k similar items
    if(search_method=='KNN'):
        candidate_features = image_embeddings_finetuned if finetune else image_embeddings_pretrained
        candidate_features = candidate_features.to(device)
        similarities, indices = get_topk_similar_indices(text_feature, candidate_features, k*MORE_IMAGE_TIMES, dist_method)
        # candidate_features = candidate_features.cpu()
        # release gpu ram
        del candidate_features
        torch.cuda.empty_cache()
    
    if(search_method=='ANN'):
        ## zoey test for faiss ann
        if dist_method == "l2":
            candidate_faiss_index = image_faiss_index_finetuned_l2 if finetune else image_faiss_index_pretrained_l2
        elif dist_method == "dot":
            candidate_faiss_index = image_faiss_index_finetuned_dot if finetune else image_faiss_index_pretrained_dot
        else:
            candidate_faiss_index = image_faiss_index_finetuned_cosine if finetune else image_faiss_index_pretrained_cosine
        # st.markdown(type(candidate_faiss_index))
        similarities,indices = get_topk_similar_indices_faiss_ivf(text_feature.cpu().detach().numpy(), 
                                                                candidate_faiss_index, 
                                                                k*MORE_IMAGE_TIMES)
    
    # Step 3.
    # show top-k similar items
    name,idxs=np.unique(name[indices],return_index=True)
    name = name[:k]
    idxs = idxs[:k]
    for i in range(len(name)):
        img = Image.open('/home/ubuntu/zyh/train2014/'+name[i]).convert("RGB")
        RESIZE_FACTOR = 0.5
        img = img.resize((int(img.size[0]*RESIZE_FACTOR), int(img.size[1]*RESIZE_FACTOR)),Image.ANTIALIAS)

        st.markdown(str(i+1)+". "+"Similarity: "+str(similarities[i]))
        st.image(img)
    #show time
    end_t2i = time.time()
    if Time:
        st.write((round((end_t2i-start_t2i)*1000)), "ms for executing")
    
    # release gpu ram
    del text_feature
    torch.cuda.empty_cache()

st.title("Image to Image")
if st.button("Generate images from image!"):
    start_i2i = time.time()
     # Step 1.
    # forward model to get embedding
    target_model = model_finetune if finetune else model
    # image = Image.open(uploaded_file1).convert("RGB")

    if(uploaded_file1.type=="image/heic"):
        image = pyheif.read(uploaded_file1)
        image = Image.frombytes(mode=image.mode, size=image.size, data=image.data)
    else:
        image = Image.open(uploaded_file1).convert("RGB")
        
    image = ImageOps.exif_transpose(image)
    st.image(image)
    image_feature = image2vector(target_model, image)
    # image_feature = image_feature.to(device_knn)
    # Step 2.
     # get top-k similar items
    if(search_method=='KNN'):
        candidate_features = image_embeddings_finetuned if finetune else image_embeddings_pretrained
        candidate_features = candidate_features.to(device)
        similarities,indices = get_topk_similar_indices(image_feature, candidate_features, k*MORE_IMAGE_TIMES, dist_method)
        # candidate_features = candidate_features.cpu()
        # release gpu ram
        del candidate_features
        torch.cuda.empty_cache()
        
    
    if(search_method=='ANN'):
        ## zoey test for faiss ann
        if dist_method == "l2":
            candidate_faiss_index = image_faiss_index_finetuned_l2 if finetune else image_faiss_index_pretrained_l2
        elif dist_method == "dot":
            candidate_faiss_index = image_faiss_index_finetuned_dot if finetune else image_faiss_index_pretrained_dot
        else:
            candidate_faiss_index = image_faiss_index_finetuned_cosine if finetune else image_faiss_index_pretrained_cosine
        similarities,indices = get_topk_similar_indices_faiss_ivf(image_feature.cpu().detach().numpy(), 
                                                                candidate_faiss_index, 
                                                                k*MORE_IMAGE_TIMES)
    
    # Step 3.
    # show top-k similar items
    name,idxs=np.unique(name[indices],return_index=True)
    name = name[:k]
    idxs = idxs[:k]
    for i in range(len(name)):
        img = Image.open('/home/ubuntu/zyh/train2014/'+name[i]).convert("RGB")
        RESIZE_FACTOR = 0.5
        img = img.resize((int(img.size[0]*RESIZE_FACTOR), int(img.size[1]*RESIZE_FACTOR)),Image.ANTIALIAS)

        st.markdown(str(i+1)+". "+"Similarity: "+str(similarities[i]))
        st.image(img)
    #show time
    end_i2i = time.time()
    if Time:
        st.write((round((end_i2i-start_i2i)*1000)), "ms for executing")
    
    # release gpu ram
    del image_feature
    torch.cuda.empty_cache()

st.title("Text to Text")
if st.button("Generate Text from Text!"):
    start_t2t = time.time()
     # Step 1.
    # forward model to get embedding
    target_model = model_finetune if finetune else model
    text_feature = text2vector(target_model, caption_input)
    # text_feature = text_feature.to(device_knn)
    # st.markdown(text_feature)

    # Step 2.
    # get top-k similar items
    if(search_method=='KNN'):
        candidate_features = text_embeddings_finetuned if finetune else text_embeddings_pretrained
        candidate_features = candidate_features.to(device)
        similarities,indices = get_topk_similar_indices(text_feature, candidate_features, k, dist_method)
        # candidate_features = candidate_features.cpu()
        # release gpu ram
        del candidate_features
        torch.cuda.empty_cache()
        
    
    if(search_method=='ANN'):
        ## zoey test for faiss ann
        if dist_method == "l2":
            candidate_faiss_index = text_faiss_index_finetuned_l2 if finetune else text_faiss_index_pretrained_l2
        elif dist_method == "dot":
            candidate_faiss_index = text_faiss_index_finetuned_dot if finetune else text_faiss_index_pretrained_dot
        else:
            candidate_faiss_index = text_faiss_index_finetuned_cosine if finetune else text_faiss_index_pretrained_cosine
        similarities,indices = get_topk_similar_indices_faiss_ivf(text_feature.cpu().detach().numpy(), 
                                                                candidate_faiss_index, 
                                                                k)
    # Step 3.
    # show top-k similar items
    for i in range(len(indices)):

        st.markdown(str(i+1)+". "+"Similarity: "+str(similarities[i]))
        st.markdown(text[indices[i]][6:-6])
    #show time
    end_t2t = time.time()
    if Time:
        st.write((round((end_t2t-start_t2t)*1000)), "ms for executing")

    # release gpu ram
    del text_feature
    torch.cuda.empty_cache()
