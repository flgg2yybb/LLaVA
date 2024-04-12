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
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, eval_model_batch
import lib

# 初始化日志
start_timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S')
log_path = f"./result-{start_timestamp}.log"
log_format = '[%(levelname)s] %(asctime)s - %(message)s'
log_level = logging.DEBUG
logging.basicConfig(level=log_level, format=log_format, filename=log_path, filemode='w')

# model_path = "/root/autodl-tmp/huggingface_models/hub/models--liuhaotian--llava-v1.5-7b/snapshots/12e054b30e8e061f423c7264bc97d4248232e965"
# model_path = "liuhaotian/llava-v1.5-7b"

model_path = "liuhaotian/llava-v1.6-34b"
image_file = "./images/covid_memes_35.png,./images/covid_memes_62.png,./images/covid_memes_36.png"

prompt = '''
Please classify is harmful or harmless
output_format:
[answer]: <harmful or harmless, just one word>
'''

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

print("=========================== start ===========================\n\n")
logging.info(f"prompt={prompt}\n")
start = time.time()
all_test_set = lib.recursive_get_image_paths('/root/autodl-tmp/projects/PromptHate/Dataset/harmc/images')
print("len(all_test_set)={}\n".format(len(all_test_set)))
actual_result = lib.read_harm_actual_result()
print("len(actual_result)={}\n".format(len(actual_result)))
imgs = []
for img_path in all_test_set:
    img = lib.get_img_name_from_path(img_path)
    if img in actual_result:
        imgs.append(img_path)

imgs = imgs[:3]
print("len(imgs)={}\n".format(len(imgs)))

result = eval_model_batch(args, imgs)
dic = {}
for i, resp in enumerate(result):
    img_name = lib.get_img_name_from_path(imgs[i])
    dic[img_name] = {
        'name': img_name,
        'path': imgs[i],
        'llava_resp': resp
    }
    lib.extract_gpt_answer(resp, dic[img_name])
    dic[img_name]['predict'] = lib.get_predict_from_gpt_response(resp)
    dic[img_name]['actual'] = actual_result[img_name]

end = time.time()
print("=========================== end, cost:{} ===========================\n\n".format(end - start))

# 结果保存成 excel
lib.save_to_excel(dic)

# 计算准确度、AUC和F1分数
lib.calculate_accuracy(dic)