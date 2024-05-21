"""
python shrink_data.py --dataset_path "preprocessed_data/c4_realnewslike_t5-base_256/train" --print_len
python shrink_data.py --dataset_path "preprocessed_data/c4_realnewslike_t5-base_256" --print_len
python shrink_data.py --dataset_path "preprocessed_data/c4_realnewslike_t5-base_256/train" --start 20000000
python shrink_data.py --dataset_path "preprocessed_data/c4_realnewslike_t5-base_256/train" --end -2000000
"""

from datasets import load_from_disk
import os
from transformers import (
    AutoTokenizer,
)
import argparse
from datasets import Dataset
import json
import shutil


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path of local data")
    parser.add_argument("--print_len", action='store_true')
    parser.add_argument("--start", type=int, default=0, help="Start index of the data to be shrinked")
    parser.add_argument("--end", type=int, default=0, help="End index of the data to be shrinked")
    args = parser.parse_args(args)
    return args


def shrink_seq_len(dataset, new_seq_length):
    tokenizer_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir="tokenizer" + tokenizer_name
    )

    # Define a function to adjust sequence length
    def adjust_sequence(example):
        example['source'] = example['source'][:256]  # adjust to the column you have the sources in
        return example

    new_dataset = dataset.map(adjust_sequence, batched=True)
    return new_dataset


def shrink_data(data_path, new_data_path, start, end, args):
    """
    :return: new data size
    """
    dataset = load_from_disk(data_path)
    print("Loaded data set path:", data_path)
    data_size = len(dataset)
    if args.print_len:
        print(f"Total data size: {data_size}\n")
        return 0
    if end <= 0:
        end += data_size
    reduced_dataset = dataset.select(range(start, end))
    reduced_dataset.save_to_disk(new_data_path)
    print(f"New data saved to {new_data_path}")
    return end - start


def main(args):
    print(args)
    data_path = args.dataset_path
    data_path = os.path.join(os.path.dirname(data_path), os.path.basename(data_path))
    new_data_path = data_path + "_reduced"
    dataset = load_from_disk(data_path)
    if isinstance(dataset, Dataset):
        print("The file is a dataset")
        shrink_data(data_path, new_data_path, args.start, args.end, args)
    elif 'train' in dataset and 'validation' in dataset:
        print("-" * 40)
        print("The file is a dataset dict.")
        train_data_path = os.path.join(data_path, "train")
        validation_data_path = os.path.join(data_path, "validation")
        if not args.print_len:
            os.makedirs(new_data_path, exist_ok=True)
        new_train_size = shrink_data(train_data_path, os.path.join(new_data_path, "train"), args.start, args.end, args)
        shrink_data(validation_data_path, os.path.join(new_data_path, "validation"), 0, 0, args)
        with open(os.path.join(data_path, "args.json")) as f:
            old_args = json.load(f)
        if "start" in old_args:
            old_args["start"] = old_args["start"] + args.start
        else:
            old_args["start"] = args.start
        old_args["end"] = old_args["start"] + new_train_size

        if not args.print_len:
            with open(os.path.join(new_data_path, "args.json"), 'w') as f:
                json.dump(old_args, f, indent=4)
            shutil.copy(os.path.join(data_path, "dataset_dict.json"), os.path.join(new_data_path, "dataset_dict.json"))
    else:
        raise ValueError("Unknown file type")


# print first 100 samples of the dataset
# print(dataset[:100])

if __name__ == '__main__':
    args = parse_args()
    main(args)
