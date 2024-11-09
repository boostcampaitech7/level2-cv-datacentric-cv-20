import os
import json

special_chars = [
    '.', ':', '-', ';', '*', '(', ')', '/', '%', '#', '!', '[', ']', '@', '+', '=', '&', '_', '>', '<', '$', '"', '\\', '{', '}'
]

def is_special_char(c):
    return c in special_chars

def is_special_char_majority(transcription):
    if not transcription:
        return False
    special_count = sum(1 for c in transcription if is_special_char(c))
    return (special_count / len(transcription)) >= 0.5

def should_exclude_transcription(transcription):
    return not transcription or is_special_char_majority(transcription)

def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img_name, img_data in data["images"].items():
        words = img_data.get("words", {})
        words_filtered = {}

        for word_id, word_data in words.items():
            transcription = word_data.get("transcription", "")
            if not should_exclude_transcription(transcription):
                words_filtered[word_id] = word_data
        
        if not words_filtered:
            img_data.pop("words", None) 
        else:
            img_data["words"] = words_filtered

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data_folder = "../data/"
for lang_folder in os.listdir(data_folder):
    json_path = os.path.join(data_folder, lang_folder, "ufo", "train.json")
    output_path = os.path.join(data_folder, lang_folder, "ufo", "remove_sc_train.json")
    
    if os.path.exists(json_path):
        process_json(json_path, output_path)
