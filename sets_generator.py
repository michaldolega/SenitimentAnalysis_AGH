import main
import math
import os
import random

DATA_DIR = "data"
MY_DATA_DIR = "my_data"
SENTENCES_DATASET_FILE_DICT = "dictionary.txt"
SENTENCES_LABELS = "sentiment_labels.txt"

phrases_dict, labels_dict = main.generate_phrases_and_labels_dicts(SENTENCES_DATASET_FILE_DICT, SENTENCES_LABELS)

test_set_size = math.floor(len(phrases_dict) * 0.2)

all_indexes = set(phrases_dict.keys())
test_indexes = set(random.sample(all_indexes, test_set_size))
train_indexes = all_indexes.difference(test_indexes)


with open(os.path.join(MY_DATA_DIR, "my_test_phrases.txt"), "w+", encoding="UTF-8") as file:
    for index in test_indexes:
        input_tuple = (str(phrases_dict[index]), str(index))
        file.write("|".join(input_tuple) + "\n")

with open(os.path.join(MY_DATA_DIR, "my_test_labels.txt"), "w+", encoding="UTF-8") as file:
    for index in test_indexes:
        input_tuple = (str(index), str(labels_dict[index]))
        file.write("|".join(input_tuple) + "\n")

with open(os.path.join(MY_DATA_DIR, "my_train_phrases.txt"), "w+", encoding="UTF-8") as file:
    for index in train_indexes:
        input_tuple = (str(phrases_dict[index]), str(index))
        file.write("|".join(input_tuple) + "\n")

with open(os.path.join(MY_DATA_DIR, "my_train_labels.txt"), "w+", encoding="UTF-8") as file:
    for index in train_indexes:
        input_tuple = (str(index), str(labels_dict[index]))
        file.write("|".join(input_tuple) + "\n")


