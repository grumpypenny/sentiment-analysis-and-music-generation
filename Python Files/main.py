import sys
import torch
import torch.nn as nn
from char_model import SentimentGRU
from char_model import convert_to_stoi
from char_model import get_emotion

from gen import Generator
from gen import sample_sequence
from gen import Vocabulary

def interactive(model, vocab):
    # Interactive Mode
    inp_string = ""

    softmax = nn.Softmax(dim=0)

    while True:
        inp_string = input("Enter message: ")

        if not inp_string:
            break # exit

        inp_string = inp_string.lower()

        stoi = convert_to_stoi(vocab, inp_string)
        x_tensor = torch.tensor([stoi])

        raw_pred = model_gru(x_tensor)
        pred = raw_pred.max(1, keepdim=True)[1]
        pred_idx = int(pred[0][0])

        if get_emotion(pred_idx) == "positive":
            happy()
        else:
            sad()

def happy():
    saved_dictionary = torch.load("../ABC-generation/saved_models/happy_load-10.pth")
    generate_music(saved_dictionary)

def sad():
    saved_dictionary = torch.load("../ABC-generation/saved_models/sad-8.pth")
    generate_music(saved_dictionary)
    

def generate_music(saved_dictionary):
    vocab = saved_dictionary['vocab']
    gen = Generator(vocab.vocab_size, 64)
    gen.load_state_dict(saved_dictionary['model_state_dict'])
    gen.eval()
    sample_sequence(gen, vocab, 500, 0.6, output_file=False)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: py main.py [sentiment analysis model name]")
        exit(1)
    
    saved_dictionary = torch.load(f"../Models/{sys.argv[1]}.pth")
    vocab = saved_dictionary['vocab']
    model_gru = SentimentGRU(len(vocab), 100, 2, bi=True, layers=1)
    model_gru.load_state_dict(saved_dictionary['model_state_dict'])
    model_gru.eval()
    print("Loaded model from ../Models/", sys.argv[1] + ".pth")
    print(f"Vocab Size: {len(vocab)}")
    interactive(model_gru, vocab)