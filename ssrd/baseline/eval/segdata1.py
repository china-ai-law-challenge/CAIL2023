import json
import os
max_words = 1500

def readjson(input_path):
    with open(input_path, 'r', encoding="utf-8") as file:
        str = file.read()
        dataset = json.loads(str)
    return dataset

def writejson(data, output_path):
    with open(output_path, 'w', encoding="utf-8") as file:
       data = json.dumps(data, ensure_ascii=False)
       file.write(data)

def seg_data(data):
    item = ""
    case_split = {}
    case_split['id'] = data['id']
    segments = []
    start = 0
    end = 0
    seg_cnt = 0
    for sent in data["Case_info"].split("。"):
        if len(item) > max_words:
            segments.append({
                "seg": seg_cnt, 
                "start": start,
                "end": end,
                "case_info": item,}
            )
            seg_cnt+=1
            start = end+1
            item = ""
        
        item += sent
        item += "。"
        end+=1
    
    if item != "":
        segments.append({
            "seg": seg_cnt, 
            "start": start,
            "end": end,
            "case_info": item,}
        )

    case_split["segments"] = segments

    return case_split
    
def splitjson(dataset, output_dir):
    for data in dataset:
        case_split  = seg_data(data)
        case_path = os.path.join(output_dir, "case", "case_seg_"+str(data["id"])+".json")
        writejson(case_split, case_path)

def main():
    input_path = "rawdata1/test.json"
    output_dir = "segjson2"
    dataset = readjson(input_path)
    splitjson(dataset, output_dir)
    # print(dataset[0])


if __name__ == '__main__':
    main()