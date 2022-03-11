import os
import json
import jieba



def get_files(dir_path):
    file_list = []
    for root, dir, files in os.walk(dir_path):
        for file in files:
            file_list.append(root + '/' + file)
    return file_list


if __name__ == '__main__':
    # os.chdir("./wiki_zh")
    # jieba.analyse.set_stop_words("stop_words.txt")
    file_list = get_files("./wiki_zh")
    total_text = ""
    separated_sens = ''
    wiki_data_file = open("src/data/input.txt", "w+")
    for file in file_list:
        print("dealing with:" + file)
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                json_obj = json.loads(line)
                total_text += json_obj['text'].replace('\n', '')
                seg_list = jieba.cut(total_text, cut_all=False)
                separated_sens += " ".join(seg_list) + "\n"
                print(1)
        wiki_data_file.write(separated_sens)
        wiki_data_file.flush()
        total_text = ''
        separated_sens = ''
    wiki_data_file.close()
