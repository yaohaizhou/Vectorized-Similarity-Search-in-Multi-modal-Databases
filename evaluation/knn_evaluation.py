import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from knn import get_topk_similar_indices

device = torch.device("cuda:0")  # cuda:0 #CPU

#### COCO data #############################################################################################
## read data
paths = ['/home/ubuntu/zyh/result/train.pkl', '/home/ubuntu/zyh/result/val.pkl', 
'/home/ubuntu/zyh/result/train_finetune.pkl', '/home/ubuntu/zyh/result/val_finetune.pkl']

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

    #### initial setting
    testing_number = N
    k_testing_list = [1,10,100,1000]
    # dist_method = "cosine"
    testing_result = pd.DataFrame()

    #### query caption by image

    ## try query
    def evaluate_error_rate(
        k_testing_list, testing_number, query_vector, key_vector, method, testing_result, type
    ):
        initial_time = datetime.now()
        error_list = []
        for k in k_testing_list:
            print(f"[Query caption by image] Test k = {k}")
            error = 0

            for i in tqdm(range(testing_number)):
                q = query_vector[i].unsqueeze(0)
                similarties, indices = get_topk_similar_indices(q, key_vector, topk=k, method=method)

                if i not in indices:
                    # print(k, i)
                    error += 1

            error_list.append(error)

        sub_testing_result = pd.DataFrame(
            {
                "k": k_testing_list,
                "error_counts": error_list,
                "dist_method": [method] * len(k_testing_list),
                "type": [type] * len(k_testing_list),
                "testing_time": [datetime.now() - initial_time] * len(k_testing_list),
            }
        )
        testing_result = testing_result.append(sub_testing_result)
        # print(sub_testing_result)
        return testing_result

    for dist_method in ["cosine", "dot", "l2"]:
        testing_result = \
            evaluate_error_rate(
                k_testing_list, testing_number, image_vector, text_vector, dist_method, testing_result, "image > text"
            )

        testing_result = \
            evaluate_error_rate(
                k_testing_list, testing_number, text_vector, image_vector, dist_method, testing_result, "text > image"
            )
    print(path)
    print(testing_result)
