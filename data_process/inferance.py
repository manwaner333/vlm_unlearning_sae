import os
import sys
import argparse
import json
import torch
import random
import math
import gc
from PIL import Image
from tqdm import tqdm
from rouge_score import rouge_scorer
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor,
    MllamaForConditionalGeneration, 
)
from huggingface_hub import hf_hub_download
from transformers import ( 
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration
)
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from datasets import load_dataset

# random.seed(233)

def main(args):
    file = args.eval_file
    split = args.split
    loss_type = args.loss_type
    dataset = load_dataset(args.eval_file, "Retain_Set")['train']
    print(len(dataset))
  
    model_path, processor = args.model_path, None
    if "llava" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        if args.checkpoint_path is not None:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
    
    elif "llama-3.2" in model_path:
        model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_path)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        if args.checkpoint_path is not None:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'
    
    if args.loss_type in ['ga', 'gd', 'kl', 'po', 'icd']:
        config = LoraConfig(
            r=128, 
            lora_alpha=256, 
            target_modules=target_modules, 
            lora_dropout=0.05,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
        model.merge_and_unload() 
    
    # dist.init_process_group(backend="nccl")
    # Get the local GPU rank
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)  # Assign process to correct GPU

    # num_gpus = torch.cuda.device_count()
    # device = torch.device("cuda" if num_gpus > 0 else "cpu")
    # if num_gpus > 1:
    #     model = nn.DataParallel(model.half())
    # model = model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.half().to(device)
    model.eval()

    
    # with open(file, "r") as f:
    #     person_data = [json.loads(line) for line in f.readlines()][0:400]
        # person_data = [line for line in person_data if line['unique_id'] in data_split[split]]  # 我将其注释掉的
    
    data = []
    for line in dataset:
        image = line['image']
        person_id = line['ID']
        image_path = line['Directory']
        biography = line['biography']
        question = line['question']
        answer = line['answer']
        generation_task = line['Generation_Task']
        data.append(
            {"image": image,
            "person_id": person_id,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            })

        for qa in generation_task:
            if qa['Type'] == 'Image_Textual':
                data.append(
                    {"image": image,
                    "person_id": person_id,
                    "image_path": image_path,
                    "question": qa['Question'],
                    "answer": qa['Ground_Truth']
                    })
    
    # random.shuffle(data)
    # print(
    #     f"Full dataset length (only include fictitious examples): {len(data)}."
    # )
    eval_data = data
    print(
        f"Subset length of the full dataset for evaluation: {len(eval_data)}."
    )

    count = 0
    nlls = []
    os.makedirs("./outputs", exist_ok=True)
    with open(f"./outputs/{args.model_name}_{split}_results.json", "w") as f:
        rougeL_list = []
        for line in tqdm(eval_data):
            image = line['image']
            # image_path = line['image_path']
            # image = Image.open(image_path)
            person_id = line['person_id']
            image_path = line['image_path']
            question, answer = line['question'], line['answer']
            if "llava-phi" in model_path:
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
                prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n" ### LLaVA-Phi
                text_input = tokenizer(prompt, return_tensors='pt')
                text_input = {k: v.to(model.device) for k, v in text_input.items()}
                inputs = {**text_input, "pixel_values": image_tensor}
                output = model.generate(**inputs, max_new_tokens=40)
                prediction = tokenizer.decode(output[0])
                prediction = prediction[prediction.find("<|assistant|>") + len("<|assistant|>"): ].strip(" ")
            elif "llava" in model_path:
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n {question} ASSISTANT: "
                text_input = tokenizer(prompt, return_tensors='pt')
                text_input = {k: v.to(model.device) for k, v in text_input.items()}
                inputs = {**text_input, "pixel_values": image_tensor}
                output = model.generate(**inputs, max_new_tokens=128)
                prediction = tokenizer.decode(output[0])
                prediction = prediction[prediction.index("ASSISTANT:"): ]
            elif "llama-3.2" in model_path:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text":question}
                    ]}
                ]
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)

                for k,v in inputs.items():
                    print(k,v.shape)

                sys.exit(0)

                output = model.generate(**inputs, max_new_tokens=128)
                prediction = processor.decode(output[0])
                prediction = prediction[prediction.index("assistant<|end_header_id|>")+ len("assistant<|end_header_id|>"):].strip("\n").strip(" ")


            outputs = {
                    "person_id": person_id,
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "prediction": prediction[:prediction.find(".") + 1].strip("")
                }

            pred, gt = outputs['prediction'], outputs['answer']
            pred = pred[:pred.find(".")].strip("")
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(gt, pred)
            rougeL_list.append(rouge_scores['rougeL'].precision)

            print(outputs)
            f.write(f"{json.dumps(outputs)}\n") 
            
            count += 1
            if count > 20:
                break 

        print(
            f"Avg RougeL scores: {sum(rougeL_list) / len(rougeL_list)}"
        )




if __name__ == "__main__":
    # eval_file = "outputs/exp1_ga_retain95_results.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        help="the path to the eval file")
    
    parser.add_argument("--split",
                        default=None,
                        type=str,
                        help="")

    parser.add_argument("--loss_type",
                        choices=["ga", "kl", "po", "icd", "retain", "full"],
                        default="ga",
                        type=str,
                        help="unlearning method")

    parser.add_argument("--model_path",
                        choices=None,
                        default="gray311/vlm_unlearned_ft_llava_v1.6_vicuna_7b",
                        type=str,
                        help="model path")

    parser.add_argument("--model_name",
                        choices=None,
                        default="llava-phi",
                        type=str,
                        help="model name")
    
    parser.add_argument("--checkpoint_path",
                        choices=None,
                        default="",
                        type=str,
                        help="lora weights of unlearning methods")

    args = parser.parse_args()
    main(args)

"""
python inference.py \
    --eval_file ./dataset/full.json \
    --loss_type full \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini_retain

python inference.py \
    --eval_file ./dataset/full.json \
    --loss_type full \
    --model_path ./models/final_ft_6_epochs_lr0.0002_llava-phi_full


python inference.py \
    --eval_file ./dataset/full.json \
    --loss_type ga \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --model_name llava-phi \
    --checkpoint_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/ga_0.0003_forget5_5/checkpoint.pt

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type po \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --model_name llava-phi \
    --checkpoint_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/idk_0.0003_forget5_5/checkpoint.pt

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type kl \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --model_name llava-phi \
    --checkpoint_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/icd_0.0003_forget5_5/checkpoint.pt




### llama-3.2-vision

python inference.py \
    --eval_file ./dataset/full.json \
    --split retain5 \
    --loss_type full \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_2_epochs_lr1e-05_llama-3.2-vision_retain

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type full \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full


"""