import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import GRU, Dense, Bidirectional, Dropout, Masking
from functools import lru_cache
from frozendict import frozendict
from random import shuffle
from keras.models import load_model

#   Directories
DATA_DIR = "data"
MY_DATA_DIR = "my_data"
MODEL_DIR = "model"

#   Source of all phrases in dataset
ALL_SENTENCES_FILE_DICT = "dictionary.txt"

#   Source of sentences chosen for model learning (80% of whole dataset)
MY_SENTENCES_TRAIN = "my_train_phrases.txt"
MY_LABELS_TRAIN = "my_train_labels.txt"

#   Source of sentences chosen for model validation (20% of whole dataset)
MY_SENTENCES_TEST = "my_test_phrases.txt"
MY_LABELS_TEST = "my_test_labels.txt"


def get_words_set(filename):
    """Reads content of file, removes index of sentence and returns set of words present in provided file"""
    result = set()
    with open(os.path.join(DATA_DIR, filename), "r", encoding="UTF-8") as source_file:
        for line in source_file:
            for word in line.split("|")[0].split():
                result.add(word.lower())
    return result


def get_words_dict(words_set):
    """Generates dictionary of words and indexes. Indexes are used to determine
     indexes of non-zero element in word-vevtor"""
    return frozendict({word: i for i, word in enumerate(sorted(words_set))})


@lru_cache(maxsize=1000)
def get_word_vector(word, words_dictionary):
    """Generates vectors, which are indexes for words without any order relation between them
    :param word: Word for which user wants generate index vector
    :param words_dictionary: Dictionary where keys are words represented as strings and values are indexes
    :return: Vector representing word in context of used dataset
    """
    vector = np.zeros(len(words_dictionary))
    if word in words_dictionary.keys():
        vector[words_dictionary[word]] = 1
    return vector


def phrase_to_tensor(phrase, words_dictionary):
    """Converts phrase in form of string into tensors shape accepted by model"""
    words_vec_array = [get_word_vector(word.lower(), words_dictionary) for word in phrase.split()]
    result_matrix = np.column_stack(words_vec_array)
    rows, cols = result_matrix.shape
    padded_matrix = np.pad(result_matrix, ((0, 0), (0, 50 - cols)), constant_values=(None, 0), mode="constant")
    transposed_matrix = np.transpose(padded_matrix)
    reshaped = np.reshape(transposed_matrix, (1,) + transposed_matrix.shape)
    return reshaped


def get_dataset_split(split_source):
    with open(os.path.join(MY_DATA_DIR, split_source), "r", encoding="UTF-8") as split_file:
        split_dict = {}
        for split_line in split_file:
            index, splitset_label = split_line.split(",")
            split_dict[int(index)] = int(splitset_label)


def generate_phrases_and_labels_dicts(phrases_source, labels_source):
    """Generates dictionaries used for mapping phrases to labels"""
    with open(os.path.join(MY_DATA_DIR, phrases_source), "r", encoding="UTF-8") as phrases_file:
        with open(os.path.join(MY_DATA_DIR, labels_source), "r", encoding="UTF-8") as labels_file:
            labels_dict = {}
            phrases_dict = {}
            for label_line in labels_file:
                index, label = label_line.split("|")
                labels_dict[int(index)] = float(label)
            for phrase_line in phrases_file:
                phrase, index = phrase_line.split("|")
                if len(phrase.split()) <= 50:
                    phrases_dict[int(index)] = phrase
            return phrases_dict, labels_dict


def generate_model_input(phrases_dict, labels_dict, words_dictionary, batch_size):
    """Generate model input as required by keras in form of batches with given size"""
    while True:
        phrases_batch = []
        labels_batch = []
        for key, phrase in phrases_dict.items():
            phrases_batch.append(phrase_to_tensor(phrase, words_dictionary))
            labels_batch.append(np.reshape(np.array(labels_dict[key]), (1, 1)))
            if len(phrases_batch) >= batch_size:
                input_item = (np.concatenate(phrases_batch), np.concatenate(labels_batch))
                phrases_batch.clear()
                labels_batch.clear()
                yield input_item
            else:
                continue
        input_item = (np.concatenate(phrases_batch), np.concatenate(labels_batch))
        phrases_batch.clear()
        labels_batch.clear()
        yield input_item


def generate_validation_input(phrases_dict, labels_dict, words_dictionary, batch_size):
    """Generate validation input as required by keras in form of batches with given size.
    Only difference to generate_model_input() function is shuffling, which helps to avoid
    overfitting model"""
    while True:
        phrases_batch = []
        labels_batch = []
        data = list(phrases_dict.items())
        shuffle(data)
        for key, phrase in data:
            phrases_batch.append(phrase_to_tensor(phrase, words_dictionary))
            labels_batch.append(np.reshape(np.array(labels_dict[key]), (1, 1)))
            if len(phrases_batch) >= batch_size:
                input_item = (np.concatenate(phrases_batch), np.concatenate(labels_batch))
                phrases_batch.clear()
                labels_batch.clear()
                yield input_item
            else:
                continue
        input_item = (np.concatenate(phrases_batch), np.concatenate(labels_batch))
        phrases_batch.clear()
        labels_batch.clear()
        yield input_item


def generate_model():
    """Function which can be used to generate model and save it into "model" directory
    with saving progress after each epoch"""
    words_set = get_words_set(ALL_SENTENCES_FILE_DICT)
    words_dict = get_words_dict(words_set)

    train_phrases_dictionary, train_labels_dictionary = generate_phrases_and_labels_dicts(MY_SENTENCES_TRAIN,
                                                                                          MY_LABELS_TRAIN)
    test_phrases_dictionary, test_labels_dictionary = generate_phrases_and_labels_dicts(MY_SENTENCES_TEST,
                                                                                        MY_LABELS_TEST)
    vector_size = len(words_dict.items())

    if not os.path.exists(os.path.join(MODEL_DIR, "temp")):
        os.makedirs(os.path.join(MODEL_DIR, "temp"))

    if not os.path.exists(os.path.join(MODEL_DIR)):
        os.makedirs(MODEL_DIR)

    checkpointer = ModelCheckpoint(os.path.join(MODEL_DIR, "temp", "my_temp_model.h5"), monitor='val_loss', verbose=0,
                                   save_best_only=False,
                                   save_weights_only=False, mode='auto', period=1)

    model = Sequential()
    model.add(Masking(input_shape=(None, vector_size)))
    model.add(Dense(50, activation="tanh"))
    model.add(Bidirectional(GRU(units=300, return_sequences=True)))
    model.add(Dropout(rate=0.5))
    model.add(Bidirectional(GRU(units=300, return_sequences=True)))
    model.add(Dropout(rate=0.5))
    model.add(Bidirectional(GRU(units=300)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(50, activation="tanh"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    model.summary(line_length=79)

    model.fit_generator(
        generator=generate_model_input(train_phrases_dictionary, train_labels_dictionary, words_dict, 300),
        validation_data=generate_validation_input(test_phrases_dictionary, test_labels_dictionary, words_dict, 1),
        steps_per_epoch=640, validation_steps=470, epochs=30, callbacks=[checkpointer])

    model.save(os.path.join(MODEL_DIR, "my_model.h5"))


def generate_printable_estimation(phrase, prediction):
    """Format output for user with estimation"""
    return '\nEstimation is in range <0;1>. The closer to 1 estimation is, the sentiment is more favorable. ' \
           'In general sentiment above 0.5 should be considered as favorable:\n' \
           + f'\n[PHRASE]: {phrase}\n[ESTIMATION]: {round(float(prediction[0][0]), 2)}\n\n'


if __name__ == "__main__":

    loaded_model = load_model(os.path.join(MODEL_DIR, "my_model.h5"))
    words_set = get_words_set(ALL_SENTENCES_FILE_DICT)
    words_dict = get_words_dict(words_set)

    while True:
        try:
            phrase = input(
                "Please provide selected movie review fragment to receive sentiment estimation.\n"
                "To quit please type :quit \n\n:> ")
            if phrase == ":quit":
                quit()
            phrase_matrix = phrase_to_tensor(phrase, words_dict)
            prediction = loaded_model.predict(phrase_matrix)

            print(generate_printable_estimation(phrase, prediction))

        except ValueError:
            print("Exception occurred during processing. Probably empty input was provided. Try again.\n")
            continue
