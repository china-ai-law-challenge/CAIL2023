from transformers import AutoModel
import torch
import os
import re
import json
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from peft import PeftModel
import argparse

def readjson(input_path):
    with open(input_path, 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
    return data

def writejson(data, output_path):
    with open(output_path, 'w', encoding="utf-8") as file:
       data = json.dumps(data, ensure_ascii=False)
       file.write(data)

def sorted_alphanum(arr):
    """
    对带有数字的字符串进行排序
    :param arr: 待排序字符串列表
    :return: 排序后的字符串列表
    """
    def convert_text(text):
        """
        提取字符串中的数字并进行转换，如果没有数字则返回原字符串
        :param text: 待处理字符串
        :return: 返回元组(a, b)，其中a为不包含数字的字符串，b为数字字符串的转换结果
        """
        return tuple(int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text))

    return sorted(arr, key=convert_text)

def generate(model,tokenizer,text):
    with torch.no_grad():
        input_text = text
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = model.generate(
            input_ids=input_ids,
            min_length=20,
            max_length=512,
            do_sample=False,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
        # answer = output.split(input_text)[-1]
    return output.strip()
    

if __name__ == "__main__":
    case_dir = "rawdata1/test.json"
    input_dir = "segresult3/ev"
    output_dir = "finalresult4"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="LexiLaw_finetune")
    argparser.add_argument("--interactive", default=True)
    args = argparser.parse_args()
    
    model = ChatGLMForConditionalGeneration.from_pretrained(args.base_model, trust_remote_code=True,  local_files_only=True)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.base_model, trust_remote_code=True,  local_files_only=True)
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.eval()
    model.half().cuda()

    cases = readjson(case_dir)
    filepathlist = os.listdir(input_dir)
    filepathlist = sorted_alphanum(filepathlist)
    pred_result = []
    for case, file_name in zip(cases, filepathlist): 
        file_path = os.path.join(input_dir, file_name)
        inter_ev_info = readjson(file_path)
        # print(inter_ev_info)
        assert case["id"] == inter_ev_info["id"]
        case_split = case["Case_info"].split("。")
        case_len = len(case_split)
        exist_list = []
        input = "目前该案件所掌握的事实如下。"
        for inter in inter_ev_info["Inter_result"]:
            input += case_split[inter]+"。"
                    
        prompt = {"instruction": "请根据下述的案件信息,总结犯罪事实。",
                    "input": input,
                }
        text = prompt["instruction"] + prompt["input"]
        # print(text)
        pred = generate(model,tokenizer, text)
        pred_res = pred.split()[-1]
        # print(pred_res)
        inter_ev_info["Final_result"] = pred_res
        
        for evs in inter_ev_info["Evidence_link"]:
            for ev in inter_ev_info["Evidence_link"][evs]:
                if ev[0] > case_len or ev[1] > case_len:
                    inter_ev_info["Evidence_link"][evs].remove(ev)
        print(inter_ev_info)
        pred_result.append(inter_ev_info)
        # print(pred_result)

    writejson(pred_result, os.path.join(output_dir, "pred.json"))
