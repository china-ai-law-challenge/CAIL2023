from transformers import AutoModel
import torch
import json
import os
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from peft import PeftModel
import argparse
import re

def readjson(input_path):
    with open(input_path, 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
    return data

def writejson(data, output_path):
    with open(output_path, 'w', encoding="utf-8") as file:
       data = json.dumps(data, ensure_ascii=False)
       file.write(data)

def check_sentence_range_format(string):
    pattern = r"^第\d+句到第\d+句$"
    is_valid = bool(re.match(pattern, string))
    # print(string)
    # print(is_valid)
    return is_valid

def extract_first_number(string):
    pattern = r"\d+"
    match = re.findall(pattern, string)
    if match:
        first_number = int(match[0])
        return first_number
    else:
        return None

def generate(model,tokenizer,text):
    with torch.no_grad():
        input_text = text
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = model.generate(
            input_ids=input_ids,
            min_length=20,
            max_length=2048,
            do_sample=False,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
        # answer = output.split(input_text)[-1]
    return output.strip()
    

if __name__ == "__main__":

    input_dir = "segjson2/case"
    output_dir = "segresult3/inter"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="model_checkpoints/checkpoint-1600")
    argparser.add_argument("--interactive", default=True)
    args = argparser.parse_args()

    model = ChatGLMForConditionalGeneration.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.eval()
    model.half().cuda()
    
    filepathlist = os.listdir(input_dir)
    for file_name in filepathlist:
        file_path = os.path.join(input_dir, file_name)
        case_info = readjson(file_path)
        for seg in case_info["segments"]:
            prompt = {"instruction": "请找出文本中的待证事实",
                        "input": seg["case_info"],
                        }
            text = prompt["instruction"] + prompt["input"]
            pred = generate(model,tokenizer, text)

            pred_res = pred.split()[-1]
            if pred_res[-1] != "句":
                pred_split = re.split(";|；", pred_res[:-1])
            else:
                pred_split = re.split(";|；", pred_res)
            # print(pred_split)
            inter_result = []
            for p in pred_split:
                if check_sentence_range_format(p):
                    print(p)
                    res = extract_first_number(p)
                    print(res)
                    inter_result.append(res)
            seg["inter_result"] = inter_result
            # print(seg["inter_result"])
        writejson(case_info, os.path.join(output_dir, "inter_seg_case_" + str(case_info["id"]) + ".json"))


        