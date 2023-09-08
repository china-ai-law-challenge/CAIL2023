import json
import os
max_words = 1500

def readjson(input_path):
    with open(input_path, 'r', encoding="utf-8") as file:
        str = file.read()
        data = json.loads(str)
    return data

def writejson(dataset, output_path):
    with open(output_path, 'w', encoding="utf-8") as file:
        for data in dataset:
            data = json.dumps(data, ensure_ascii=False)
            file.write(data)
            file.write('\n')

def labeldata(dataset):
    inter_id = "inter"
    inter_id += "_e" + str(dataset['id'])
    child_dataset = []
    for seg in dataset['Inter_segments']:
        seg_id = inter_id
        seg_id += ("_"+ str(seg["seg"]) +
                    "_s"+ str(seg["start"]) + 
                    "_e"+str(seg["end"]))
        answer = []
        # print(dataset['id'])
        # print(seg["start"])
        # print(seg['inter_result'])
        for sid in seg['inter_result']:
            # print(sid-seg["start"])
            answer.append("第"+ str(sid-seg["start"]) + "句到第"+ str(sid-seg["start"]) + "句")

        if answer != []:
            answer = '；'.join(answer)+ "。"
        else:
            answer = ""
        prompt = {"instruction": "请找出文本中的待证事实",
                   "input": seg["case_info"],
                   "answer": answer}

        child_dataset.append(prompt)

    return child_dataset

def main():
    input_dir = "segjson2/interseg"
    output_path = "labeldata3/train_inter.json"
    input_dir_list = os.listdir(input_dir)
    dataset = []
    for filename in input_dir_list:
        file_path = os.path.join(input_dir, filename)
        data = readjson(file_path)
        event_inter = labeldata(data)
        # print(event_inter)
        dataset.extend(event_inter)
    
    writejson(dataset, output_path)


if __name__ == '__main__':
    main()