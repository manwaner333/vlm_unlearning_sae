import os
import sys
import torch
# import wandb
import json
import plotly.express as px
# from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from functools import partial
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
import io

import os
import sys
import torch
# import wandb
import json
import pickle
import plotly.express as px
# from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from torch.utils.data import DataLoader 
from datasets import Dataset, Features, Value
from datasets import Image as dataset_Image 
import json
from tqdm import tqdm, trange
import torch
import random
import numpy as np

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data
from sae_training.utils import ViTSparseAutoencoderSessionloader
from sae_training.hooked_vit import HookedVisionTransformer, Hook

import os
import sys
import torch
import wandb
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial
from vit_sae_analysis.dashboard_fns import get_feature_data   # FeatureData

import gzip
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
from datasets import load_dataset
from tqdm import trange
from eindex import eindex
from IPython.display import HTML, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor, topk
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_activations_store import ViTActivationsStore
import torchvision.transforms as transforms
from PIL import Image
from sae_training.utils import ViTSparseAutoencoderSessionloader
import shutil


from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data    # FeatureData
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data     # FeatureData
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from plotly import express as xp
import torch
import plotly.io as pio
from typing import Union, List, Optional
import torch


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.path.append("..")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

def plot_alignment_fig(cos_sims):
    example_fig = px.line(cos_sims.to('cpu'))
    example_fig.show() 

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()
    

def create_scatter_plot(data, x_label = "", y_label = "", title="", colour_label = ""):
    num_columns = data.size()[0]
    assert num_columns in [2,3], "data should be of size (2,n) to create a scatter plot"
    data = data.transpose(0,1)
    
    if num_columns ==2:
        assert colour_label == "", "You can't submit a colour label when no colour variable is submitted"
        # Convert the torch tensor to a pandas DataFrame
        df = pd.DataFrame(data.numpy(), columns=[x_label, y_label])
        # Create the scatter plot with marginal histograms
        fig = px.scatter(df, x=x_label, y=y_label,
                        marginal_x='histogram', marginal_y='histogram',
                        )
    else:
        # Convert the torch tensor to a pandas DataFrame
        df = pd.DataFrame(data.numpy(), columns=[x_label, y_label, colour_label])
        # Create the scatter plot with marginal histograms
        fig = px.scatter(df, x=x_label, y=y_label, color=colour_label,
                        marginal_x='histogram', marginal_y='histogram',
                        color_continuous_scale=px.colors.sequential.Bluered,  # Optional: specify color scale     Bluered
                        )
        
    # Show the plot
    fig.update_layout(title={
        'text': title,
        'x': 0.5,  # Centering the title
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'family': 'Arial',
            'size': 24,
            'color': '#333333'
        }})
    # fig.show()
    fig.write_image("aaa222.png")


def get_sae_activations(model_activaitons, sparse_autoencoder):
    hook_name = "hook_hidden_post"
    max_batch_size = sparse_autoencoder.cfg.max_batch_size_for_vit_forward_pass # Use this for the SAE too
    number_of_mini_batches = model_activaitons.size()[0] // max_batch_size
    remainder = model_activaitons.size()[0] % max_batch_size
    sae_activations = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: obtaining sae activations"):
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size])[1][hook_name])
    
    if remainder>0:
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[-remainder:])[1][hook_name])
        
    sae_activations = torch.cat(sae_activations, dim = 0)
    sae_activations = sae_activations.to(sparse_autoencoder.cfg.device)
    return sae_activations  


def conversation_form(key):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": key},  # "What's in the picture?"
            {"type": "image"},
            ],
        },
    ]
    return conversation


def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    
    activations = activations[:,-7,:]

    return activations


def get_all_model_activations(model, images, conversations, cfg):
    max_batch_size = cfg.max_batch_size_for_vit_forward_pass
    number_of_mini_batches = len(images) // max_batch_size
    remainder = len(images) % max_batch_size
    sae_batches = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: forward pass images through ViT"):
        image_batch = images[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        conversation_batch = conversations[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        # inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "What is shown in this image?"},
        #         {"type": "image"},
        #         ],
        #     },
        # ]
        # prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch_of_prompts = []
        for ele in conversation_batch:
            batch_of_prompts.append(model.processor.apply_chat_template(ele, add_generation_prompt=True))
        
        inputs = model.processor(images=image_batch, text=batch_of_prompts, padding=True, return_tensors="pt").to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
    
    if remainder>0:
        image_batch = images[-remainder:]
        conversation_batch = conversations[-remainder:]
        # inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": "What is shown in this image?"},
        #         {"type": "image"},
        #         ],
        #     },
        # ]
        # prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch_of_prompts = []
        for ele in conversation_batch:
            batch_of_prompts.append(model.processor.apply_chat_template(ele, add_generation_prompt=True))
            
        inputs = model.processor(images=image_batch, text=batch_of_prompts, padding=True, return_tensors="pt").to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
        
    sae_batches = torch.cat(sae_batches, dim = 0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches


def load_sae_model(sae_path):
    sae_path = sae_path
    loaded_object = torch.load(sae_path)
    cfg = loaded_object['cfg']
    state_dict = loaded_object['state_dict']

    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()

    loader = ViTSparseAutoencoderSessionloader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)
    
    return sparse_autoencoder, model
    
    

def save_sae_activations(sparse_autoencoder, model, dataset_path, label_index, label, directory):
    torch.cuda.empty_cache()
    try:
        with open(dataset_path, "r") as f:
            data_json = json.load(f)
    except:
        with open(dataset_path, "r") as f:
            data_json = [json.loads(line) for line in f.readlines()]
            
    dataset_dict = {
        "image": [item["image"] for item in data_json],
        "label": [item["label"] for item in data_json]
    }

    features = Features({
        "image": dataset_Image(),
        "label": Value("string")
    })

    dataset_all = Dataset.from_dict(dataset_dict, features=features)
    dataset = dataset_all.filter(lambda example: example['label'] == str(label_index[label]))
    
    print(f"Total data quantity: {len(dataset)}")
    
    image_key = 'image'
    image_label = 'label'
    
    number_of_images_processed = 0
    max_number_of_images_per_iteration = len(dataset)
    while number_of_images_processed < len(dataset):
        torch.cuda.empty_cache()
        try:
            images = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_key]
            labels = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_label]
            conversations = [conversation_form(str(ele)) for ele in labels]
        except StopIteration:
            print('All of the images in the dataset have been processed!')
            break
        
        model_activations = get_all_model_activations(model, images, conversations, sparse_autoencoder.cfg) 
        sae_activations = get_sae_activations(model_activations, sparse_autoencoder).transpose(0,1) 
        torch.save(sae_activations, f'{directory}/sae_activations/emotion/sae_activations_{label_index[label]}.pt')
        number_of_images_processed += max_number_of_images_per_iteration
        

def save_selected_features(directory, label, compare_label, k):
    sae_activations_0 = torch.load(f'{directory}/sae_activations/emotion/sae_activations_{label}.pt').to('cpu')
    sae_activations_200 = torch.load(f'{directory}/sae_activations/emotion/sae_activations_{compare_label}.pt').to('cpu') 

    sum_0 = sae_activations_0.mean(dim=1, keepdim=True)
    sum_200 = sae_activations_200.mean(dim=1, keepdim=True)

    epsilon = torch.finfo(sum_0.dtype).eps
    total_sum_0 = sum_0.sum() + epsilon 
    total_sum_200 = sum_200.sum() + epsilon

    ratio_0 = sum_0 / total_sum_0 
    ratio_200 = sum_200 / total_sum_200 


    diff_ratio = ratio_0 - ratio_200
    values_ratio, indices_ratio = torch.topk(diff_ratio, k, dim=0)
    values_sum_0, indices_sum_0 = torch.topk(sum_0, k, dim=0)
    
    # indices = indices_ratio[torch.isin(indices_ratio, indices_sum_0)]
    indices = indices_ratio
    values = sum_0[indices]
    
    torch.save(indices, f'{directory}/feature_indices/emotion/feature_indices_{label}.pt')
    torch.save(values, f'{directory}/feature_values/emotion/feature_values_{label}.pt')
    
    print("value of epsion:", epsilon)
    print("total_sum:", total_sum_0)
    print("a shape:", sum_0.shape)        
    print("ratio shape:", ratio_0.shape)
    print("Top 10 indices:", indices.squeeze().tolist())
    print("Top 10 values:", values.squeeze().tolist())


def save_figures(sparse_autoencoder, directory, dataset_path, split, label):
    
    dataset_path = dataset_path
    split = split
    dataset_all = load_dataset(dataset_path, split=split)
    
    if sparse_autoencoder.cfg.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
    
    label = label
    start_index = label * 50 
    if start_index >= 1000:
        start_index = start_index - 1000
    end_index = (label + 1 ) * 50 + 1000
    
    dataset_5000 = dataset_all.select(range(start_index, end_index))
    dataset = dataset_5000.filter(lambda example: example['label'] == label)
    print(f"Total data quantity: {len(dataset)}")
    
    first_image = dataset[3][image_key]
    if isinstance(first_image, Image.Image):
        first_image.save(f"{directory}/images/image_{label}.png")  
        print(f"Image saved as image_{label}.png")
    elif isinstance(first_image, dict) and "bytes" in first_image:
        img = Image.open(io.BytesIO(first_image["bytes"]))
        img.save(f"{directory}/images/image_{label}.png")
        print(f"Image saved as image_{label}.png")
    else:
        print("Unsupported image format:", type(first_image))

def generate_adj_sae_outputs(sparse_autoencoder, model, directory, label, max_token):
    image_file = f"{directory}/images/emotion/image_{label}.jpg"
    raw_image = Image.open(image_file)
    
    res_file = f"{directory}/emotion_adj_sae_outputs.json"
    
    existing_labels = []
    if os.path.exists(res_file):
        with open(res_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                existing_labels.append(line['label'])

        if label in existing_labels:
            print(f"Label '{label}' Already exists, skip adding.")
            return
            
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please identify the category that best describes the facial expressions of the characters in the image from the following options:[happy, sad, angry, surprised, scared, disgusted, excited, relaxed, confused, bored]"},   # "What are these?" "Please describe this picture." What is the main color of the object in this picture?
            {"type": "image"},
            ],
        },
    ]
    
    prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True)
    model_inputs = model.processor(images=raw_image, text=prompt, return_tensors='pt').to(sparse_autoencoder.device, torch.float16)
    
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    pixel_values = model_inputs.pixel_values
    # image_sizes = model_inputs.image_sizes

    tokenizer = model.processor.tokenizer
    prompt_str_tokens = tokenizer.convert_ids_to_tokens(input_ids[0]) 
    # answer_str_tokens = tokenizer.convert_ids_to_tokens(answer_tokens[0])

    max_token = max_token
    generated_ids = input_ids.clone()


    def sae_hook(activations):
        activations[:,0:576,:] = sparse_autoencoder(activations[:,0:576,:], label)[0]    
        return (activations,)

    sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=True)]

    new_tokens = []
    with torch.no_grad():
        for ele in range(max_token):
            # print(f"Token: {ele}")
            outputs = model.run_with_hooks(
                sae_hooks,
                return_type='output',
                input_ids=generated_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                # image_sizes=image_sizes,
            )
            
            logits = outputs.logits[:, -1, :]  
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_tokens.append(next_token)
            new_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            torch.cuda.empty_cache()

    output_texts = model.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_texts)
    res = {"label": label, "instruction": "Please identify the category that best describes the facial expressions of the characters in the image from the following options:[happy, sad, angry, surprised, scared, disgusted, excited, relaxed, confused, bored]", "output_texts": output_texts}
    
    with open(res_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n") 

        
    
               
    
seed = 42 
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
dataset_path = "dataset/emotion"
directory = "dashboard_2621440"

### 1. load model
sparse_autoencoder, model = load_sae_model(sae_path)

### 2. 生成存储数据的文件
# files = os.listdir(dataset_path)
# res = []
# for file in files:
#     if "happy" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 0}
#     elif "sad" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 1}
#     elif "angry" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 2}
#     elif "surprised" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 3}
#     elif "scared" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 4}
#     elif "disgusted" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 5}
#     elif "excited" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 6}
#     elif "relaxed" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 7}
#     elif "confused" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 8}
#     elif "bored" in file.lower():
#         ele = {'image': f'dataset/emotion/{file.lower()}', "label": 9}
#     res.append(ele)
# res_file= f"dataset/emotion.json"
# with open(res_file, "a", encoding="utf-8") as f_new:
#     for entry in res:
#         f_new.write(json.dumps(entry, ensure_ascii=False) + "\n")      



### 3. geneate sae_activations
# data_path = f"dataset/emotion.json"
# label_index = {"happy":0, "sad":1, "angry":2, "surprised":3, "scared":4, "disgusted":5, "excited":6, "relaxed":7, "confused":8, "bored":9}
# for label in tqdm (["happy", "sad", "angry", "surprised", "scared", "disgusted", "excited", "relaxed", "confused", "bored"], desc="Processing labels"):
#     file_path = f'{directory}/sae_activations/emotion/sae_activations_{label_index[label]}.pt'
#     if not os.path.exists(file_path):
#         print(f"label: {label}")
#         save_sae_activations(sparse_autoencoder, model, data_path, label_index, label=label, directory=directory)
#     else:
#         print(f"File already exists: {file_path}")


### 4. save figures
# for label in tqdm(range(500, 700, 2), desc="Processing Figures"):
#     save_figures(sparse_autoencoder, directory, dataset_path, split="val", label=label)


### 5. select features 
# pairs = {0: 5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
# for ele in tqdm(range(0, 10), desc="Processing Features"):
#     compare_label = pairs[ele]
#     print(compare_label)
#     save_selected_features(directory, ele, compare_label, k=10) 
        

### 5. generate adj_sae_outputs
for ele in tqdm(range(0, 10), desc="Processing Features"):
    print(f"Emotion: {ele}")
    generate_adj_sae_outputs(sparse_autoencoder, model, directory, label=ele, max_token=306)

