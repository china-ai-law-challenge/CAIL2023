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

def extract_number(string):
    pattern = r"\d+"
    match = re.findall(pattern, string)
    if match:
        if len(match) > 1:
            first_number = int(match[0])
            second_number = int(match[1])
        elif len(match) == 1:
            first_number = int(match[0])
            second_number = int(match[0])
        
        return first_number, second_number
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

    input_dir = "segresult3/inter"
    output_dir = "segresult3/ev"
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
    case_list = []
    # TODO: 合一块
    for file_name in filepathlist:
        file_path = os.path.join(input_dir, file_name)
        case_info = readjson(file_path)
        case_i = {}
        case_i["id"] = case_info["id"]
        inter_list = []
        ev_dict = {}
        for inter_seg in case_info["segments"]:
            for seg in case_info["segments"]:
                # seg["Evidence_link"] = {}
                for inter_i in inter_seg["inter_result"]:
                    print(inter_i)
                    print(inter_seg["end"])
                    if inter_i > (inter_seg["end"] - inter_seg["start"]): continue
    
                    inter = inter_seg["case_info"].split("。")[inter_i] + "。"
                    prompt = {"instruction": "请在文本中找出下述待证事实对应的相关证据。"+ inter,
                                "input": seg["case_info"],}
                    text = prompt["instruction"] + prompt["input"]
                    pred = generate(model, tokenizer, text)

                    pred_res = pred.split()[-1]
                    print(pred_res)
                    if pred_res[-1] != "句":
                        pred_split = re.split(";|；", pred_res[:-1])
                    else:
                        pred_split = re.split(";|；", pred_res)
                    # print(pred_split)
                    ev_result = []
                    for p in pred_split:
                        if check_sentence_range_format(p):
                            # print(p)
                            start, end = extract_number(p)
                            ev_result.append([start, end])

                    if ev_result != []:
                        if inter_i+inter_seg["start"] not in inter_list:
                            inter_list.append(inter_i+inter_seg["start"])
                        for sid in range(len(ev_result)):
                            ev_result[sid][0] = ev_result[sid][0] + seg["start"]
                            ev_result[sid][1] = ev_result[sid][1] + seg["start"]
                        if str(inter_i+inter_seg["start"]) in ev_dict:
                            ev_dict[str(inter_i+inter_seg["start"])].extend(ev_result)
                            print("case {}, inter {}, add new ev list {}".format(case_info["id"], 
                                                    str(inter_i+inter_seg["start"]),
                                                    ev_dict[str(inter_i+inter_seg["start"])]))
                        else:
                            ev_dict[str(inter_i+inter_seg["start"])] = ev_result
                            print("case {}, inter {}, add new ev list {}".format(case_info["id"], 
                                                    str(inter_i+inter_seg["start"]),
                                                    ev_dict[str(inter_i+inter_seg["start"])]))

        case_i["Inter_result"]= inter_list
        case_i["Evidence_link"]= ev_dict
        # print(seg["inter_result"])
        writejson(case_i, os.path.join(output_dir, "ev_case_" + str(case_info["id"]) + ".json"))


        