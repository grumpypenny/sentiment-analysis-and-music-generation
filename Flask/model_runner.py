import os
import sys

import torch
import torch.nn as nn

from char_model_runner import SentimentGRU
from char_model_runner import convert_to_stoi

from gen_runner import Generator
from gen_runner import sample_sequence


def generate_ABC(inp_string):

    S_140_KEY= {0: "negative",
                1: "positive"}

    SENTIMENT_MODEL_NAME = "test-10.pth"

    # Load Sentiment Model
    saved_dictionary = torch.load(f"../Models/"+SENTIMENT_MODEL_NAME)
    vocab = saved_dictionary['vocab']
    model_gru = SentimentGRU(len(vocab), 100, 2, bi=True, layers=1)
    model_gru.load_state_dict(saved_dictionary['model_state_dict'])
    model_gru.eval()
    print("Loaded model from ../Models/"+SENTIMENT_MODEL_NAME)
    print(f"Vocab Size: {len(vocab)}")

    # Process input and feed into sentiment model
    softmax = nn.Softmax(dim=0)

    inp_string = inp_string.lower()
    stoi = convert_to_stoi(vocab, inp_string)
    x_tensor = torch.tensor([stoi])
    raw_pred = model_gru(x_tensor)
    pred = raw_pred.max(1, keepdim=True)[1]
    pred_idx = int(pred[0][0])

    # TODO: Get maximum model instead of hard coded one
    if S_140_KEY[pred_idx] == "positive":
        saved_dictionary = torch.load("../Models/happy-10.pth")
    else:
        saved_dictionary = torch.load("../Models/sad-8.pth")

    # Load Music Model
    vocab = saved_dictionary['vocab']
    gen = Generator(vocab.vocab_size, 64)
    gen.load_state_dict(saved_dictionary['model_state_dict'])
    gen.eval()

    # Get ABC string
    return sample_sequence(gen, vocab, 500, 0.6, output_file=False, print_out=False)