import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate approximately 2000-3000 data points to match the density in the plot
n_points = 2500

# Generate log10 sparsity values (x-axis: approximately -4.5 to 0.5)
log10_sparsity = np.random.uniform(-4.5, 0.5, n_points)

# Create a nonlinear relationship between sparsity and mean activation
# The plot shows a curved relationship with some scatter
base_relationship = 2.5 * log10_sparsity + 3.0 * (log10_sparsity + 3)**0.8 - 6.5

# Add noise to create the scatter
noise = np.random.normal(0, 0.3, n_points)
log10_sae_mean_acts = base_relationship + noise

# Generate label entropy values based on the color mapping
# Higher sparsity (less negative) tends to have higher entropy (redder colors)
# Lower sparsity (more negative) tends to have lower entropy (bluer colors)
entropy_base = 0.8 + 0.4 * (log10_sparsity + 4) / 4.5  # Scale to roughly 0.8 to 1.2
entropy_noise = np.random.normal(0, 0.15, n_points)
label_entropy = entropy_base + entropy_noise

# Clip entropy values to reasonable range (0 to ~3 based on colorbar)
label_entropy = np.clip(label_entropy, 0, 2.8)

# Add some additional structure to match the visible patterns
# Create two main clusters as visible in the plot
cluster_1_mask = (log10_sparsity < -2) & (log10_sae_mean_acts < 0)
cluster_2_mask = (log10_sparsity > -2) & (log10_sae_mean_acts > -0.5)

# Adjust entropy for clusters
label_entropy[cluster_1_mask] = np.clip(label_entropy[cluster_1_mask] * 0.7, 0, 1.5)  # Lower entropy for blue cluster
label_entropy[cluster_2_mask] = np.clip(label_entropy[cluster_2_mask] * 1.4 + 0.5, 1.0, 2.8)  # Higher entropy for red cluster

# Create the final dataset
data = pd.DataFrame({
    'log10_sparsity': log10_sparsity,
    'log10_sae_mean_acts': log10_sae_mean_acts,
    'label_entropy': label_entropy
})

# Sort by sparsity for better visualization
data = data.sort_values('log10_sparsity').reset_index(drop=True)

# Display first few rows and basic statistics
print("Generated dataset shape:", data.shape)
print("\nFirst 10 rows:")
print(data.head(10))

print("\nDataset statistics:")
print(data.describe())

# Save to CSV
data.to_csv('sae_analysis_data.csv', index=False)
print("\nData saved to 'sae_analysis_data.csv'")

# Show the ranges to verify they match the plot
print(f"\nData ranges:")
print(f"Log10 Sparsity: {data['log10_sparsity'].min():.2f} to {data['log10_sparsity'].max():.2f}")
print(f"Log10 SAE Mean Acts: {data['log10_sae_mean_acts'].min():.2f} to {data['log10_sae_mean_acts'].max():.2f}")
print(f"Label Entropy: {data['label_entropy'].min():.2f} to {data['label_entropy'].max():.2f}")



# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import requests
# from torch.utils.data import DataLoader 
# from datasets import Dataset, Features, Value  #  Image
# from datasets import Image as dataset_Image 
# import json
# from tqdm import tqdm, trange

# # def get_image_batches(iterable_dataset):
# #   # device = self.cfg.device
# #   batch_of_conversations = []
# #   with torch.no_grad():  # self.cfg.store_size
# #     for _ in trange(2, desc = "Filling activation store with images"):
# #       try:
# #         batch_of_conversations.append(next(iterable_dataset)[label_key])
# #       except StopIteration:
# #         iterable_dataset = iter(dataset.shuffle())
# #         batch_of_conversations.append(next(iterable_dataset)[label_key])
# #   return batch_of_images
      
      
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
# model.to("cuda:0")

# # # prepare image and text prompt, using the appropriate prompt template
# # # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# # # image = Image.open(requests.get(url, stream=True).raw)

# # # image = Image.open("dataset/SFHQ/SFHQ_pt1_00000115.jpg")

# # data_path = "./dataset/select.json"
# # try:
# #     with open(data_path, "r") as f:
# #         data_json = json.load(f)
# # except:
# #     with open(data_path, "r") as f:
# #         data_json = [json.loads(line) for line in f.readlines()]

# # dataset_dict = {
# #     "image": [item["image_path"] for item in data_json],
# #     "label": [item["label"] for item in data_json]
# # }

# # features = Features({
# #     "image": dataset_Image(), 
# #     "label": Value("string")
# # })

# # hf_dataset = Dataset.from_dict(dataset_dict, features=features)

# # dataset = hf_dataset
# # image_key = 'image'
# # label_key = 'label'
# # iterable_dataset = iter(dataset)

# # device = ""
# from datasets import load_dataset
# dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
        
# image_key = 'image'
# label_key = 'label'

# labels = dataset.features[label_key].names
# dataset = dataset.shuffle(seed=42)
# iterable_dataset = iter(dataset)

        

# key1 = "What is shown in this image?"     # "What is t he full name of the person in the image?"
# key2 = "What is shown in this image?" 
# batch_of_images = []
# with torch.no_grad():
#   for _ in trange(2, desc = "Filling activation store with images"):
#     try:
#       batch_of_images.append(next(iterable_dataset)[image_key])
#     except StopIteration:
#       iterable_dataset = iter(dataset.shuffle())
#       batch_of_images.append(next(iterable_dataset)[image_key])

# # def conversation_form(key):
# #   conversation = [
# #     {
# #       "role": "user",
# #       "content": [
# #           {"type": "text", "text": key},
# #           {"type": "image"},
# #         ],
# #     },
# #   ]
# #   return conversation

# # batch_of_conversations = []
# # with torch.no_grad():  # self.cfg.store_size
# #   for _ in trange(2, desc = "Filling activation store with images"):
# #     try:
# #       batch_of_conversations.append(conversation_form(next(iterable_dataset)[label_key]))
# #     except StopIteration:
# #       iterable_dataset = iter(dataset.shuffle())
# #       batch_of_conversations.append(conversation_form(next(iterable_dataset)[label_key]))


# # # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# # # Each value in "content" has to be a list of dicts with types ("text", "image") 

# conversation1 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": key1},
#           {"type": "image"},
#         ],
#     },
# ]
# conversation2 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": key2},
#           {"type": "image"},
#         ],
#     },
# ]

# prompt1 = processor.apply_chat_template(conversation1, add_generation_prompt=True)
# prompt2 = processor.apply_chat_template(conversation2, add_generation_prompt=True)


# # batch_of_prompts = []
# # for ele in batch_of_conversations:
# #   batch_of_prompts.append(processor.apply_chat_template(ele, add_generation_prompt=True))

# batch_of_prompts = [prompt1, prompt2]  

# inputs = processor(images=batch_of_images, text=batch_of_prompts, padding=True, return_tensors="pt").to("cuda:0")

# qingli = 3
# # # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)

# print(processor.decode(output[0], skip_special_tokens=True))
# print(processor.decode(output[1], skip_special_tokens=True))



  


####load dataset
# data_path = "./dataset/select.json"
# try:
#     with open(data_path, "r") as f:
#         data_json = json.load(f)
# except:
#     with open(data_path, "r") as f:
#         data_json = [json.loads(line) for line in f.readlines()]

# dataset_dict = {
#     "image": [item["image_path"] for item in data_json],
#     "label": [item["label"] for item in data_json]
# }

# features = Features({
#     "image": dataset_Image(), 
#     "label": Value("string")
# })

# hf_dataset = Dataset.from_dict(dataset_dict, features=features)

# dataset = hf_dataset
# image_key = 'image'
# label_key = 'label'
# iterable_dataset = iter(dataset)

# batch_of_images = []
# with torch.no_grad():
#   for _ in trange(2, desc = "Filling activation store with images"):
#     try:
#       batch_of_images.append(next(iterable_dataset)[image_key])
#     except StopIteration:
#       iterable_dataset = iter(dataset.shuffle())
#       batch_of_images.append(next(iterable_dataset)[image_key])


# key1 = "What is the full name of the person in the image?" 
# key2 = "What is shown in this image?" 
# conversation1 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": key1},
#           {"type": "image"},
#         ],
#     },
# ]
# conversation2 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": key2},
#           {"type": "image"},
#         ],
#     },
# ]

# prompt1 = model.processor.apply_chat_template(conversation1, add_generation_prompt=True)
# prompt2 = model.processor.apply_chat_template(conversation2, add_generation_prompt=True)

# batch_of_prompts = [prompt1, prompt2]  

# inputs = model.processor(images=batch_of_images, text=batch_of_prompts, padding=True, return_tensors="pt").to("cuda:0")

# generate the original response
# output = model.model.generate(**inputs, max_new_tokens=100)
# print(model.processor.decode(output[0], skip_special_tokens=True))
# print(model.processor.decode(output[1], skip_special_tokens=True))





# output, cache_dict = model.model.run_with_hooks(sae_hooks, return_type='output', **model_inputs).item()
# zero_ablation_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, zero_ablation, return_module_output=False)]
# zero_ablation_loss = model.model.run_with_hooks(zero_ablation_hooks, return_type='loss', **model_inputs).item()

# output = model(return_type='output', **model_inputs)
# logits = output.logits
# next_str = logits_to_next_str(logits, model = model.model)

# answers = {}
# max_tokens = 10
# completion = ""
# gen = 0
# while gen < max_tokens:
#   logits = 
#   next_str = logits_to_next_str(logits)
#   completion += next_str
#   out += next_str
#   gen+=1

# 尝试进行decode
# logits = output.logits 
# predicted_ids = torch.argmax(logits, dim=-1)
# decoded_text = model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
# print(decoded_text)
# print("qingli ")

# sae_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, sae_hook, return_module_output=False)]
# reconstruction_loss = model.model.run_with_hooks(sae_hooks, return_type='loss', **model_inputs).item()

# zero_ablation_hooks = [Hook(sparse_autoencoder.cfg.block_layer, sparse_autoencoder.cfg.module_name, zero_ablation, return_module_output=False)]
# zero_ablation_loss = model.model.run_with_hooks(zero_ablation_hooks, return_type='loss', **model_inputs).item()