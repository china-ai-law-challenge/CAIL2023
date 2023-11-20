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

def handle_info(data, start, end):
    sub_interresult = []
    sub_evidence = {}
    truth_info = []
    
    for residx in range(start, end+1):
        if residx in data["Inter_result"]:
            sub_interresult.append(residx)
    
    for truth in data["Inter_result"]:
        for ev in data["Evidence_link"][str(truth)]:
            # print(data["Evidence_link"][str(truth)])
            if ev[0] >= start and ev[1] <= end:
                # print(ev)
                if str(truth) not in sub_evidence:
                    sub_evidence[str(truth)] = []
                sub_evidence[str(truth)].append(ev)
                # print(sub_evidence)

    for t_id in sub_evidence:
        truth_info.append({
            "t_id": t_id, 
            "info":  data["Case_info"].split("。")[int(t_id)] + "。",
        })
    return sub_interresult, sub_evidence, truth_info

    
def seg_data(data):
    inter_split = {}
    inter_split['id'] = data['id']
    ev_split = {}
    ev_split['id'] = data['id']
    inter_segments = []
    evidence_segments = []
    start = 0
    end = 0
    seg_cnt = 0
    item = ""

    for sent in data["Case_info"].split("。"):
        if len(item) > max_words:
            sub_interresult, sub_evidence, truth_info = handle_info(data, start, end)
            inter_segments.append({
                "seg": seg_cnt, 
                "start": start,
                "end": end,
                "case_info": item,
                "inter_result": sub_interresult,}
            )

            evidence_segments.append({
                "seg": seg_cnt, 
                "start": start,
                "end": end,
                "case_info": item,
                "truth_info": truth_info,
                "ev_result": sub_evidence,}
            )
            seg_cnt+=1
            start = end+1
            item = ""
        
        item += sent
        item += "。"
        end+=1
    
    if item != "":
        sub_interresult, sub_evidence, truth_info = handle_info(data, start, end)
        inter_segments.append({
            "seg": seg_cnt, 
            "start": start,
            "end": end,
            "case_info": item,
            "inter_result": sub_interresult,}
        )

        evidence_segments.append({
            "seg": seg_cnt, 
            "start": start,
            "end": end,
            "case_info": item,
            "truth_info": truth_info,
            "ev_result": sub_evidence,}
        )

    inter_split["Inter_segments"] = inter_segments
    ev_split["Ev_segments"] = evidence_segments
    print(inter_split)
    print(ev_split)

    return inter_split, ev_split 
    
def splitjson(dataset, output_dir):
    for data in dataset:
        inter_split, ev_split  = seg_data(data)
        inter_path = os.path.join(output_dir, "evseg", "inter_seg_"+str(data["id"])+".json")
        ev_path =  os.path.join(output_dir, "interseg", "ev_seg_"+str(data["id"])+".json")
        writejson(inter_split, inter_path)
        writejson(ev_split, ev_path)

def main():
    input_path = "rawdata1/train.json"
    output_dir = "segjson2"
    dataset = readjson(input_path)
    splitjson(dataset, output_dir)
    # print(dataset[0])


if __name__ == '__main__':
    main()