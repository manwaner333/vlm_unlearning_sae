import os
import json
import logging
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import re
import numpy as np
from resource import openai_key
import csv



def load_json_lines(file_path):
    data = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: 
                    try:
                        entry = json.loads(line) 
                        label = entry["label"]
                        instruction = entry["instruction"]
                        output_texts = entry["output_texts"]
                        data[label] = (instruction, output_texts) 
                    except json.JSONDecodeError:
                        print(f"Skip unparseable lines: {line}")
    return data

def merge_json_files(adj_sae, no_adj_sae, original_model, output_file):
    adj_sae_file = load_json_lines(adj_sae)
    no_adj_sae_file = load_json_lines(no_adj_sae)
    original_model_file = load_json_lines(original_model)

    common_labels = set(adj_sae_file.keys()) & set(no_adj_sae_file.keys()) & set(original_model_file.keys())
    sorted_common_labels = sorted(common_labels)
    
    merged_data = []
    for label in sorted_common_labels:
        merged_entry = {
            "label": label,
            "original": original_model_file[label][1],
            "no_adjust": no_adj_sae_file[label][1],
            "adjust": adj_sae_file[label][1]
        }
        merged_data.append(merged_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(" ", 1)
            
            if len(parts) == 2:
                key, value = parts
                if key.isdigit():
                    data_dict[int(key)] = value.strip()
            else:
                print(f"Skip unparseable lines:{line}")

    return data_dict


sys_prompt = """ You are a helpful and impartial assistant.  You will receive a text description and a target object. Your task is to evaluate whether the text description refers to the target object.

Please use the following scale to rate your evaluation:
- Rating: [[2]]: The text description does not contain the target object.
- Rating: [[1]]: The text description contains the target object.

Provide your rating strictly in this format: "Rating: [[rating]]", where the rating inside the double brackets must be either 1 or 2.
"""


def call_llm_api(messages, max_tokens, temperature, top_p, n, stop, engine="gpt-4o"):
    client = OpenAI(api_key=openai_key)
    result = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        seed=0,
    )

    return result


def gpt_wrapper(qa, sys_prompt=None, max_tokens=512):
    if sys_prompt is None:
        messages = []
    else:
        messages = [{
            "role": "system",
            "content": sys_prompt
        }]
    messages.append(
        {
            "role": "user",
            "content": qa
        }
    )
    try:
        result = call_llm_api(messages, max_tokens, temperature=0.0, top_p=1.0, n=1, stop=["\n\n"])
    except Exception as e:
        print(e)
        return None, str(e)
    raw = result.choices[0].message.content
    score = re.findall(r': \[\[(\d)\]\]', raw)
    if len(score) != 1:
        print(f"Error: {raw}")
        return None, raw
    return int(score[0]), raw


# def evaluate_gpt():
#     dataset = [{"Original": "Nobel prize in literature to be announced http:\/\/t.co\/qxlEqdl3",
#                "Add_context": "Stay tuned for the exciting announcement of the Nobel Prize in Literature, which will take place shortly. You can follow the event live here: http:\/\/t.co\/qxlEqdl3."},
#                {"Original": "\u201c@marvicleonen: Is it true that UP won UAAP basketball?\u201d -- Next year, Dean. Sure na 'yan!",
#                 "Add_context": "Aren't we all excited about the performance of UP in the UAAP basketball finals? I heard they clinched the title this year! What do you think, Dean? It's a sure win!"}
#                ]
#     result = []
#     for item in tqdm(dataset):
#         original_text = item['Original']
#         text_by_AI = item['Add_context']
#         # hallucination avoidance
#         qa = f"Original Text: {original_text}\nAI-generated Text: {text_by_AI}"
#         input_prompt = f"{sys_prompt}\n\nNow classify the following response:\n{qa}"
#         pred, raw = gpt_wrapper(input_prompt, sys_prompt=None)
#         result.append(pred)
#     return result


if __name__ == '__main__':
    
    ### 1 合并生成的三种文件
    # directory = "dashboard_2621440"
    # adj_sae = f"{directory}/adj_sae_outputs.json"
    # no_adj_sae = f"{directory}/no_adj_sae_outputs.json"
    # original_model = f"{directory}/original_model_outputs.json"
    # output_file = f"{directory}/merged_file.json"

    # merge_json_files(adj_sae, no_adj_sae, original_model, output_file)
    
    
    ### 2. 用openai评估生成的内容
    file_path = "dashboard_2621440/objects_class.txt" 
    data_dict = txt_to_dict(file_path)
    res_file = "dashboard_2621440/merged_file.json"
    new_res_file = "dashboard_2621440/merged_file_add_evaluation.json"
    a = []
    correct_num = 0
    with open(res_file, "r", encoding="utf-8") as f:
        for line in f:
            print("qingli!!")
            line = json.loads(line)
            label = line['label']
            object = data_dict[label]
            original_text = line["original"]
            no_adjust = line['no_adjust']
            adjust = line['adjust']
            qa = f"Text: {original_text}\n Object: {object}"
            # qa = f"Text: {adjust}\n Object: {object}"
            input_prompt = f"{sys_prompt}\n\nNow classify the following response:\n{qa}"
            pred, raw = gpt_wrapper(input_prompt, sys_prompt=None)
            if pred == 2:
                correct_num += 1
            res = {"label": label, "original": original_text, "no_adjust": no_adjust, "adjust": adjust, "object": object,"evaluation": pred}  # "instruction": instruction,
            a.append(res)
    
    correct_ratio = correct_num / 200
    print(f"Correct Ratio: {correct_ratio}")

                
    with open(new_res_file, "a", encoding="utf-8") as f_new:
        for entry in a:
            f_new.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    ### 2. 统计评估的内容
    # correct_num = 0
    # total_num = 0
    # with open(new_res_file, "r", encoding="utf-8") as f_new:
    #     for line in f_new:
    #         line = json.loads(line)
    #         total_num += 1
    #         if line['evaluation'] == 2:
    #             correct_num += 1
                
    # correct_ratio = correct_num / total_num
    # print(f"Correct Ratio: {correct_ratio}")