from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import keras
import numpy as np
import random
import sys
import json
import os
import argparse


def build_index(chars:list):
    # index corresponding to a given char
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # char corresponding to an index
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char


def model_train(text:str, seed_len:int, stride:int=3, batch_size:int=128, epochs:int=60):
    """
    Builds and trains keras model to predict a next char based on the given string.

    :param epochs:
    :param batch_size:
    :param text: text to be used for model training
    :param seed_len: length of a chars sequence to use as model input
    :param stride:
    :return:
    """
    model_props = {}

    # split the whole text into chars
    chars = sorted(list(set(text)))
    model_props["chars"] = chars
    model_props["seed_len"] = seed_len

    char_indices, indices_char = build_index(chars=chars)

    sentences = []
    next_chars = []
    for i in range(0, len(text) - seed_len, stride):
        sentences.append(text[i: i + seed_len])
        next_chars.append(text[i + seed_len])

    x = np.zeros((len(sentences), seed_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    # the idea is to generate a next char given the first 'seed_len' chars
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = Sequential()
    model.add(LSTM(128, input_shape=(seed_len, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    return model, model_props


def model_apply(model:Sequential, seed:str, maxlen:int, chars:list, seed_len:int):
    """
    Return original text(seed) + sequence of generated chars.
    :param model: trained keras model build to predict the next char based on the given sequence
    :param seed: beginning of a phrase to be processed by the model
    :param maxlen: number of chars to be generated
    :param chars: all chars known by the model
    :param seed_len: required lengh of the seed.
    :return:
    """

    generated = ''
    if len(seed) < seed_len:
        sentence = (seed_len - len(seed)) * " " + seed
    elif len(seed) > seed_len:
        sentence = seed[-seed_len:]
    else:
        sentence = seed
    generated += sentence

    char_indices, indices_char = build_index(chars=chars)

    for i in range(maxlen):
        # prepare input features
        x_pred = np.zeros((1, seed_len, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        # use previous prediction (character) to predict the next one
        sentence = sentence[1:] + next_char
        generated += next_char
    return generated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode",
                    required=True,
                    choices=["train", "use"],
                    dest="model_mode",
                    help="script working mode: train|use")

    ap.add_argument("--name",
                    required=True,
                    type=str,
                    dest="model_name",
                    help="model name. used to save the trained model or to load an existing one")

    ap.add_argument("--seed-len",
                    required=False,
                    type=int,
                    default=10,
                    dest="seed_len",
                    help="length of initial seed - length of the string to start with")

    ap.add_argument("--seed",
                    required=False,
                    type=str,
                    dest="seed",
                    help="text to start with when model is in use")

    ap.add_argument("--text-file",
                    required=False,
                    type=argparse.FileType('r'),
                    dest="text_file_handle",
                    help="name of the text file to be used for training")

    ap.add_argument("--maxlen",
                    required=False,
                    type=int,
                    dest="maxlen",
                    help="number of chars to be generated")

    ap.add_argument("--epochs",
                    required=False,
                    type=int,
                    dest="epochs",
                    help="number of epochs for model training")

    args = ap.parse_args()

    if args.model_mode == "train":
        model, model_props = model_train(text=args.text_file_handle.read(),
                                         seed_len=args.seed_len,
                                         epochs=args.epochs)

        if not os.path.isdir(args.model_name):
            os.makedirs(args.model_name)

        with open(os.path.join(args.model_name, f"model_props.json"), "w") as f:
            json.dump(model_props, f)

        model.save(os.path.join(args.model_name, f"model.h5"))

    if args.model_mode == "use":
        with open(os.path.join(args.model_name, f"model_props.json"), "r") as f:
            model_props = json.load(f)

        model = keras.models.load_model(os.path.join(args.model_name, f"model.h5"))

        generated = model_apply(model=model, seed=args.seed, maxlen=args.maxlen, **model_props)
        print(generated)




