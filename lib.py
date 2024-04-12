import argparse
import concurrent.futures
import json
import logging
import math
import os
import random
import sys
import time
import shutil
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

actual_path= '/root/autodl-tmp/projects/PromptHate/Dataset/harmc/test.jsonl'

# 可以处理 harmc 和 harmp 的 xxx.jsonl
def read_harm_actual_result():
    path = actual_path
    if not os.path.isfile(path):
        raise Exception(f'{path} is not valid result file')
    dic = {}
    with open(path) as f:
        for line in f.readlines():
            obj = json.loads(line)
            labels = obj.get('labels')
            for label in labels:
                if label == 'not harmful':
                    dic[obj.get('image')] = 0
                elif label == 'somewhat harmful':
                    dic[obj.get('image')] = 1
                elif label == 'very harmful':
                    dic[obj.get('image')] = 1
                else:
                    continue
    return dic

# 获取 path 下的所有文件的绝对地址
def recursive_get_image_paths(path):
    if not os.path.exists(path):
        raise Exception(f"图片路径有错，路径:{path}")
    images = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            temp = recursive_get_image_paths(file_path)
            images = images + temp
        elif file_path.find('.png') != -1 or file_path.find('.jpg') != -1:
            images.append(file_path)
    return images


# extract_gpt_answer 提取 chat gpt 回复的 [Q1], [Q2] 等答案
def extract_gpt_answer(s, dic):
    for line in s.split('\n'):
        # 使用split方法以冒号分割行，得到问题和答案
        qa = line.split(':')
        # 检查问题是否以Q开头，如果是，打印答案
        if len(qa) > 1 and qa[0].find('[Q') != -1:
            dic[qa[0].strip()] = qa[1]


def get_img_name_from_path(image_path):
    # 根据 image path 获取 image file name
    splits = image_path.split('/')
    if len(splits) == 1:
        return splits[0]
    return splits[len(splits) - 1]


# get_result_from_gpt_response:
# non-hateful -> 0, hateful -> 1
# non-harmful -> 0, harmful -> 1
def get_predict_from_gpt_response(s):
    # flag_idx = s.find(']')
    # if flag_idx == -1:
        # return -1
    # start = flag_idx + len(']')
    # content = s[start:].lower()
    content = s.lower()
    # hateful 的情况
    if 'non-hateful' in content:
        return 0
    elif 'non hateful' in content:
        return 0
    elif 'nonhateful' in content:
        return 0
    elif 'hateful' in content:
        return 1
    # harmful 的情况
    elif 'non-harmful' in content:
        return 0
    elif 'non harmful' in content:
        return 0
    elif 'nonharmful' in content:
        return 0
    elif 'harmless' in content:
        return 0
    elif 'harmful' in content:
        return 1
    else:
        return -1

# extract_gpt_answer 提取 chat gpt 回复的 [Q1], [Q2] 等答案
def extract_gpt_answer(s, dic):
    for line in s.split('\n'):
        # 使用split方法以冒号分割行，得到问题和答案
        qa = line.split(':')
        # 检查问题是否以Q开头，如果是，打印答案
        if len(qa) > 1 and qa[0].find('[Q') != -1:
            dic[qa[0].strip()] = qa[1]


def save_to_excel(result_map):
    t = datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S')
    result_map_list = []
    for k, v in result_map.items():
        result_map_list.append(v)
    df = pd.DataFrame(result_map_list)
    df.to_csv(f"./result-{t}.csv", index=False)

def calculate_accuracy(result_map):
    predict_list = []
    actual_list = []
    ignore = 0
    for img_name, img_map in result_map.items():
        if 'predict' not in img_map or img_map['predict'] == -1:
            # 跳过无效的数据计算
            ignore += 1
            continue
        predict_list.append(img_map['predict'])
        actual_list.append(img_map['actual'])
    logging.info(f"{ignore} 个无效数据，本次结果统计不包含这些无效数据\n")
    logging.info(f"predict_list={predict_list}\n")
    logging.info(f"actual_list={actual_list}\n")
    if len(predict_list) == 0 or len(actual_list) == 0:
        logging.warning(f"数据为空，跳过统计 accuracy 和 f1\n")
        return
    accuracy = accuracy_score(actual_list, predict_list)
    f1 = f1_score(actual_list, predict_list, average='macro')
    logging.info(f"结果统计(不包含无效gpt预测数据)：accuracy={accuracy},fi={f1}\n")
    calculate_wrong_predict(predict_list, actual_list)

def calculate_wrong_predict(predict, actual):
    harm_to_not_harm = 0
    not_harm_to_harm = 0
    for i in range(len(predict)):
        if predict[i] == -1:
            continue
        if actual[i] != predict[i]:
            # 预测错了
            if actual[i] == 0:
                not_harm_to_harm += 1
            if actual[i] == 1:
                harm_to_not_harm += 1
    logging.info(f"预测错的情况 - [harmful -> non-harmful] count={harm_to_not_harm}, "
                 f"占比总数：{harm_to_not_harm / len(predict)}\n")
    logging.info(f"预测错的情况 - [non-harmful -> harmful] count={not_harm_to_harm}, "
                 f"占比总数：{not_harm_to_harm / len(predict)}\n")