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
from functools import partial
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
import io

import os
import sys
import torch
import wandb
import json
import pickle
import plotly.express as px
from transformer_lens import utils
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


def generate_no_adj_sae_outputs(sae_path, sparse_autoencoder, model, directory, label, max_token=306):
    
    image_file = f"{directory}/images/sport/image_{label}.jpg"
    raw_image = Image.open(image_file)
    
    res_file = f"{directory}/sport_no_adj_sae_outputs.json"
    
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
            {"type": "text", "text": "Please identify the sport depicted in the image by selecting one of the following categories: [soccer, basketball, football, tennis, baseball, volleyball, golf, hockey, rugby, badminton]."},   # "What are these?"
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
        activations[:,0:576,:] = sparse_autoencoder(activations[:,0:576,:])[0]    
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
    
    res = {"label": label, "instruction": "Please identify the sport depicted in the image by selecting one of the following categories: [soccer, basketball, football, tennis, baseball, volleyball, golf, hockey, rugby, badminton].", "output_texts": output_texts}
    with open(res_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n") 
    
    
    
seed = 42 
sae_path = "checkpoints/models--jiahuimbzuai--sae_64/snapshots/e44861c762f4a32084ac448f31cd7264800610df/2621440_sae_image_model_activations_7.pt"
# dataset_path = "evanarlian/imagenet_1k_resized_256"
directory = "dashboard_2621440"

### 1. load model
sparse_autoencoder, model = load_sae_model(sae_path)
for ele in tqdm(range(0, 10), desc="Processing Features"):
    print(f"sport: {ele}")
    generate_no_adj_sae_outputs(sae_path, sparse_autoencoder, model, directory, label=ele, max_token=306)