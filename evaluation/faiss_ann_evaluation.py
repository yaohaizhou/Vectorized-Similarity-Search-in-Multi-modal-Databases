import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from faiss_ann import *

device = torch.device("cuda:0")  # cuda:0 #CPU

def evaluate_error_rate(k_testing_list, testing_number, query_vector, key_vector, method, testing_result, type, path):
    
    # create faiss index
    key_vector = key_vector.cpu().detach().numpy()
    index = create_index(keys = key_vector, nlist = 100, nprobe = 3, method = method)
    
    # query (loop)
    initial_time = datetime.now()
    error_list = []
    for k in k_testing_list:
        print(f"k = {k}, method = {method}, type = {type}, path = {path}")
        error = 0

        for i in tqdm(range(testing_number)):
            q = query_vector[i].unsqueeze(0)
            q = q.cpu().detach().numpy()
            similarties, indices = get_topk_similar_indices_faiss_ivf(q, index, k)

            if i not in indices:
                error += 1

        error_list.append(error)

    sub_testing_result = pd.DataFrame(
        {
            "k": k_testing_list,
            "error_counts": error_list,
            "dist_method": [method] * len(k_testing_list),
            "type": [type] * len(k_testing_list),
            "path": [path] * len(k_testing_list), 
            "testing_time": [datetime.now() - initial_time] * len(k_testing_list),
        }
    )
    testing_result = testing_result.append(sub_testing_result)
    # print(sub_testing_result)
    return testing_result

#### COCO data #############################################################################################
#### initial setting
# paths = ['/home/ubuntu/zyh/result/train.pkl', '/home/ubuntu/zyh/result/val.pkl', 
#          '/home/ubuntu/zyh/result/train_finetune.pkl', '/home/ubuntu/zyh/result/val_finetune.pkl']
paths = ['/home/ubuntu/zyh/result/train.pkl']
testing_number = 80000
k_testing_list = [100,1000]

testing_result = pd.DataFrame()
for path in paths:
    f = open(path, "rb")
    data_load = pickle.load(f)
    N = len(data_load)

    ## preprocess image feature
    image_vector = np.concatenate([data["image_feature"] for data in data_load])
    image_vector = torch.FloatTensor(image_vector).to(device)

    ## preprocess text feature
    text_vector = np.concatenate([data["text_feature"] for data in data_load])
    text_vector = torch.FloatTensor(text_vector).to(device)

    ## try query
    for dist_method in ["cosine"]:
        testing_result = \
            evaluate_error_rate(
                k_testing_list, testing_number, image_vector, text_vector, dist_method, testing_result, "image > text", path,
            )

        testing_result = \
            evaluate_error_rate(
                k_testing_list, testing_number, text_vector, image_vector, dist_method, testing_result, "text > image", path,
            )
            
print(testing_result)
testing_result.to_csv("faiss_ann_testing_results_100_3_1206.csv", index = False)
