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
    ev_id = "evidence"
    ev_id += "_e" + str(dataset['id'])
    child_dataset = []
    for seg in dataset['Ev_segments']:
        seg_id = ev_id
        seg_id += ("_"+ str(seg["seg"]) +
                    "_s"+ str(seg["start"]) + 
                    "_e"+str(seg["end"]))

        for truth in seg['truth_info']:
            # print(sid-seg["start"]) 
            answer = []
            # print(dataset['id'])
            # print(seg["start"])
            # print(seg['inter_result'])
            for ev in seg['ev_result'][str(truth['t_id'])]:
                answer.append("第"+ str(ev[0]-seg["start"]) + "句到第" + str(ev[1]-seg["start"]) + "句")

            if answer != []:
                answer = '；'.join(answer)+ "。"
            else:
                answer = ""
            prompt = {"instruction": "请在文本中找出下述待证事实对应的相关证据。{},".format(truth["info"]),
                        "input": seg["case_info"],
                        "answer": answer}

            child_dataset.append(prompt)

    return child_dataset

def main():
    input_dir = "segjson2/evseg"
    output_path = "labeldata3/train_ev.json"
    input_dir_list = os.listdir(input_dir)
    dataset = []
    for filename in input_dir_list:
        file_path = os.path.join(input_dir, filename)
        data = readjson(file_path)
        event_ev = labeldata(data)
        # print(event_ev)
        dataset.extend(event_ev)
    
    writejson(dataset, output_path)


if __name__ == '__main__':
    main()