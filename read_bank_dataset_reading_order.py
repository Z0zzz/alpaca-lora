# coding=utf-8
import json
import copy
import torch
from torch.utils.data import Dataset
import transformers
import os
import random

class ReadBankDataset(Dataset):
    def __init__(self, features):
        self.make_dataset(features)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        sample = {}
        sample["input_ids"] = self.all_input_ids[idx]
        sample["labels"] = self.all_labels[idx]
        sample["attention_mask"] = self.all_attention_mask[idx]

        return sample


    def make_dataset(self, features):
        # Convert to Tensors and build dataset
        #self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # self.all_bboxes = torch.tensor([f.bboxes if torch.min(f.bboxes[:,3]-f.bboxes[:,1]) >= 0 for f in features], dtype=torch.long)
        self.size = 0
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for f in features:
            bboxes = torch.tensor(f.bboxes)
            if torch.min(bboxes[:,3]-bboxes[:,1]) >= 0 and torch.min(bboxes[:,2]-bboxes[:,0]) >= 0:
                # print("padding attn mask: ", torch.sum(torch.tensor(f.attention_mask) == 0))
                # print("input ids padding: ", torch.sum(torch.tensor(f.input_ids) == 32000))
                all_labels.append(f.label)
                all_input_ids.append(f.input_ids)
                all_attention_mask.append(f.attention_mask)
                self.size += 1
        self.all_input_ids = torch.tensor(all_input_ids)
        self.all_labels = torch.tensor(all_labels)
        self.all_attention_mask = torch.tensor(all_attention_mask)
    


class DocFeature(object):
    def __init__(self, input_ids, bboxes, attention_mask, token_type_ids, label):
        assert (
            0 <= all(bboxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            bboxes
        )
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.bboxes = bboxes
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def get_word_formatted(element):
    return f"{element[0]} <{element[1]}, {element[2]}, {element[3]}, {element[4]}>"

def convert_examples_to_features(
    layout_data,
    qa_data,
    tokenizer,
    max_length=2048,
    label_list=None,
    pad_on_left=False,
    pad_token="[PAD]",
    pad_token_id=32000,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    for_inference = False
):
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    features = []
    for idx in range(len(layout_data)):
        if idx % 100 == 0:
            print(f"processing {idx} / {len(layout_data)}")
        layout_tokens = []
        qa_tokens = []
        layout_words = []
        qa_words = []

        bboxes = []
        words = []
    
        # process layout data
        layout_len_limit = 50
        layout_len_count = 0
        words_layout_info = []
        for line in layout_data[idx]:
            _, token, y1, x1, y2, x2, w, h = line.split("\t")
            layout_len_count += 1 
            bbox = [float(x1) / float(w),float(y1) / float(h),float(x2) / float(w),float(y2) / float(h)]
            bbox = [int(x * 1000) for x in bbox]
            words_layout_info.append([token] + bbox)
            bboxes.append(bbox)
            if layout_len_count > layout_len_limit:
                break
        
        label_words = " ".join([word[0] for word in words_layout_info])
        words_layout_info_split1 = words_layout_info[:len(words_layout_info)//2]
        words_layout_info_split2 = words_layout_info[len(words_layout_info)//2:]
        random.shuffle(words_layout_info_split1)
        words_layout_info_split2.sort(key=lambda x: x[0])
        words_layout_info_split = words_layout_info_split1 + words_layout_info_split2
        layout_info = map(get_word_formatted, words_layout_info)
        layout_info = ',\n'.join(layout_info)
        prompt = f"Given the following sequence of information composed of a word and its corresponding bounding box of the format '[word] <x1, y1, x2, y2>': \n{layout_info}\nRecover the following text of the correct reading order:"
        inputs = tokenizer(prompt)
        prompt_len = len(inputs.input_ids)
        if prompt_len > 1400:
            print(prompt_len, "skipping")
            continue
        if not for_inference:
            label = tokenizer(label_words, padding='max_length', max_length=max_length - prompt_len, truncation=True)
            total_inputs = inputs.input_ids + label.input_ids[1:]   # not including <\s> token
            total_attn_mask = inputs.attention_mask + label.attention_mask[1:]
            label = total_inputs[:]
            print(tokenizer.decode(total_inputs))
            for i in range(prompt_len):
                label[i] = -100

            features.append(
            DocFeature(
                input_ids = total_inputs,
                attention_mask = total_attn_mask,
                token_type_ids=None,
                bboxes = bboxes,
                label=label,
            )
        )
        else:    
            features.append({"input_ids": inputs.input_ids, "attention_mask":inputs.attention_mask})
    return features


def get_doc_from_name(file_names):
    data_dir = "/data/mengke/readingBank/zilongwang/raw-outputs/DocRO-data-1/zip-docs-output-1"
    entries = os.listdir(data_dir)
    layout_datas = []
    qa_datas = []
    for file in file_names:
        for entry in entries:
            if file in entry:
                break
        layout_file = open(os.path.join(data_dir,entry))
        qa_file = open(f"./Dataset/readBank_qa/generated_{file}")
    
        layout_data = []
        next(layout_file)
        for entry in layout_file:
            layout_data.append(entry)

        qa_data = []
        next(qa_file)
        for entry in qa_file:
            qa_data.append(entry)

        layout_datas.append(layout_data)
        qa_datas.append(qa_data)

    return layout_datas, qa_datas


def get_all_qa():
    data_dir = "/home/mengke/DocLLM/Dataset/readBank_qa"
    entries = os.listdir(data_dir)
    task_names = []
    for entry in entries:
        task_names.append(entry.split("_")[1])
    return task_names

def get_examples():
    data_dir = "/data/mengke/readingBank/zilongwang/raw-outputs/DocRO-data-1/zip-docs-output-1"
    qa_data_dir = "/home/mengke/DocLLM/Dataset/readBank_qa"
    entries = os.listdir(data_dir)
    qa_entries = get_all_qa()

    task_names = []
    for entry in entries:
        task_names.append(entry)
    
    layout_datas = []
    qa_datas = []
    for idx, entry in enumerate(task_names):
        if entry not in qa_entries:
            continue

        layout_file = open(os.path.join(data_dir,entry))
        qa_file = open(os.path.join(qa_data_dir, f"generated_{entry}"))
        
        layout_data = []
        next(layout_file)
        for entry in layout_file:
            layout_data.append(entry)

        qa_data = []
        next(qa_file)
        for entry in qa_file:
            if "\n" == entry:
                continue
            qa_data.append(entry.strip() + "\n")
            
        layout_datas.append(layout_data)
        qa_datas.append(qa_data)

    return layout_datas, qa_datas

def get_examples_layout_only():
    data_dir = "/data/mengke/readingBank/zilongwang/raw-outputs/DocRO-data-1/zip-docs-output-1"
    entries = os.listdir(data_dir)
    task_names = []
    for entry in entries:
        task_names.append(entry)
    
    layout_datas = []
    qa_datas = []
    
    for idx, entry in enumerate(task_names):
        layout_file = open(os.path.join(data_dir,entry))

        layout_data = []
        next(layout_file)
        for entry in layout_file:
            layout_data.append(entry)

        qa_data = []

        layout_datas.append(layout_data)
        qa_datas.append(qa_data)
    
    return layout_datas, qa_datas


def get_dataset(model_path='/data/mengke', num_examples = 3000, training_split=0.9, for_inference=False):
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)

    layout_datas, qa_datas = get_examples_layout_only()

    training_sample = int(training_split * num_examples)

    layout_data = layout_datas[:training_sample]
    qa_data = qa_datas[:training_sample]
    features = convert_examples_to_features(layout_data, qa_data, tokenizer, for_inference=for_inference)
    if not for_inference:
        train_dataset = ReadBankDataset(features)
    else:
        train_dataset = features
    
    layout_data = layout_datas[training_sample:num_examples]
    qa_data = qa_datas[training_sample:num_examples]
    
    features = convert_examples_to_features(layout_data, qa_data, tokenizer, for_inference=for_inference)
    if not for_inference:    
        eval_dataset = ReadBankDataset(features)
    else:
        eval_dataset = features

    return train_dataset, eval_dataset

def get_huggingface_dataset(num_examples = 8000):
    layout_data, _ = get_examples_layout_only()
    layout_data = layout_data[:min(num_examples, len(layout_data))]
    tokenizer = transformers.LlamaTokenizer.from_pretrained("/data/mengke")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data = []
    features = []
    for idx in range(len(layout_data)):
        dp = {}
        if idx % 100 == 0:
            print(f"processing {idx} / {len(layout_data)}")
        layout_tokens = []
        qa_tokens = []
        layout_words = []
        qa_words = []

        bboxes = []
        words = []
    
        # process layout data
        layout_len_limit = 50
        layout_len_count = 0
        words_layout_info = []
        for line in layout_data[idx]:
            _, token, y1, x1, y2, x2, w, h = line.split("\t")
            layout_len_count += 1 
            bbox = [float(x1) / float(w),float(y1) / float(h),float(x2) / float(w),float(y2) / float(h)]
            bbox = [int(x * 1000) for x in bbox]
            words_layout_info.append([token] + bbox)
            bboxes.append(bbox)
            if layout_len_count > layout_len_limit:
                break
        
        label_words = " ".join([word[0] for word in words_layout_info])
        # words_layout_info_split1 = words_layout_info[:len(words_layout_info)//2]
        # words_layout_info_split2 = words_layout_info[len(words_layout_info)//2:]
        random.shuffle(words_layout_info)
        # words_layout_info_split2.sort(key=lambda x: x[0])
        # words_layout_info_split = words_layout_info_split1 + words_layout_info_split2
        layout_info = map(get_word_formatted, words_layout_info)
        layout_info = ',\n'.join(layout_info)
        prompt = f"Given the following sequence of information composed of a word and its corresponding bounding box of the format '[word] <x1, y1, x2, y2>': \n{layout_info}\nRecover the following text of the correct reading order:"
        '''
        label_words = " ".join([word[0] for word in words_layout_info])
        inputs = [word[0] for word in words_layout_info]
        random.shuffle(inputs)
        inputs = " ".join(inputs)
        prompt = f"Given the following words of a text in a random order, recover its original order."
        '''
        dp["instruction"] = prompt
        dp["input"] = None
        dp["output"] = label_words
        # print(dp)
        data.append(dp)
    import json
    with open("reading_bank_ro_instruction_dataset.json", "w") as f:
        json.dump(data, f)

    from datasets import Dataset, DatasetDict
    import pandas as pd
    # Load your data from the JSON file
    with open("reading_bank_ro_instruction_dataset.json", "r") as f:
        data = json.load(f)
    
    
    # Convert your data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a Dataset object
    dataset = Dataset.from_pandas(df)
    # Register your dataset with a name, e.g., "my_dataset"
    datasets_dict = DatasetDict({"train": dataset})
    return datasets_dict
    # from datasets import save_to_disk
    # save_to_disk(datasets_dict, dataset_config_name="my_dataset")

def get_huggingface_dataset2(num_examples = 3000):
    import json,random
    from datasets import Dataset, DatasetDict
    import pandas as pd
    new_data = []
    # Load your data from the JSON file
    with open("alpaca_data.json", "r") as f:
        data = json.load(f)
    for dp in data:
        instruction = dp["instruction"]
        input = dp["input"]
        output = dp["output"]
        instruction = "Rearrange the words to form meaningful sentences."
        input = output.split(" ")
        random.shuffle(input)
        input = " ".join(input)

        dp["instruction"] = instruction
        dp["input"] = input
        new_data.append(dp)
    with open("reading_bank_ro_instruction_dataset.json", "w") as f:
        json.dump(new_data, f)

    from datasets import Dataset, DatasetDict
    import pandas as pd
    # Load your data from the JSON file
    with open("reading_bank_ro_instruction_dataset.json", "r") as f:
        data = json.load(f)
    # Convert your data to a pandas DataFrame
    df = pd.DataFrame(data)
    # Convert the DataFrame to a Dataset object
    dataset = Dataset.from_pandas(df)
    # Register your dataset with a name, e.g., "my_dataset"
    datasets_dict = DatasetDict({"train": dataset})
    return datasets_dict



# get_dataset()