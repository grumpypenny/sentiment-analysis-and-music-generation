import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import csv
import matplotlib.pyplot as plt

import sys

class Generator(nn.Module):
    """
    NETWORK: GRU -> fully connected
    """

    def __init__(self, vocab_size, hidden_size, n_layers=1, drop=0):
        """
        Create a network. 
        NOTE: if n_layers is 1 then drop MUST be 0
        """
        super(Generator, self).__init__()

        # matrix for making the one hot vectors
        self.ident = torch.eye(vocab_size)

        # GRU layer
        self.rnn = nn.GRU(vocab_size, hidden_size, n_layers, batch_first=True, dropout=drop).cuda()

        # fully connected layer 
        # outputs distribution over the next token given rnn output
        self.fc = nn.Linear(hidden_size, vocab_size).cuda()

    def forward(self, inp, hidden=None):
        # turn character into one-hot
        inp = self.ident[inp]
        inp = inp.cuda()

        # forward pass, get next output and hidden state
        output, hidden = self.rnn(inp, hidden)
        
        # predict distribution over next token
        output = self.fc(output).cuda()

        return output, hidden

class Vocabulary():
    """
    Stores the vocabulary of the data

    vocab_stoi: dictionary that maps strings to ints
    vocab_itos: dictionary that maps ints to strings
    vocab_size: the number of different items in the vocabulary

    """
    
    def __init__(self, data, text_field):
        # self.vocab_stoi = {}
        # self.vocab_itos = {}
        # self.vocab_size = 0

        text_field.build_vocab(data)
        self.vocab_stoi = text_field.vocab.stoi # so we don't have to rewrite sample_sequence
        self.vocab_itos = text_field.vocab.itos # so we don't have to rewrite sample_sequence
        self.vocab_size = len(text_field.vocab.itos)

        # print("vocab size: ", self.vocab_size)

def sample_sequence(model, vocab, max_len=100, temperature=0.5):
    """
    Generate a sequence from <model>
    <vocab> is a Vocabulary object that matches the dataset 
        <model> is trained on
    """
    generated_sequence = ""
   
    inp = torch.Tensor([vocab.vocab_stoi["<BOS>"]]).long()
    hidden = None
    for p in range(max_len):
        output, hidden = model(inp.unsqueeze(0), hidden)

        # increase output to avoid 0 prob 
        output += 0.0001
        # Sample from the network as a multinomial distribution
        # print("output =",output)
        output_dist = output.data.view(-1).div(temperature).exp()
        # print("dist =", output_dist)
        top_i = int(torch.multinomial(output_dist, 1)[0])
        # Add predicted character to string and use as next input
        predicted_char = vocab.vocab_itos[top_i]
        
        if predicted_char == "<EOS>":
            break
        generated_sequence += predicted_char       
        inp = torch.Tensor([top_i]).long()
    return generated_sequence

def get_data():
    """
    Get the cleaned abc's and return:
    the abc's in a tabular dataset and the textfield
    """
    text_field = torchtext.data.Field(sequential=True,      # text sequence
                                    tokenize=lambda x: x, # because are building a character-RNN
                                    include_lengths=True, # to track the length of sequences, for batching
                                    batch_first=True,
                                    use_vocab=True,       # to turn each character into an integer index
                                    init_token="<BOS>",   # BOS token
                                    eos_token="<EOS>")    # EOS token

    fields = [('text', text_field)]
    abc = torchtext.data.TabularDataset("data.csv", "csv", fields)

    return abc, text_field

def train(model, data, vocab, batch_size=8, num_epochs=1, lr=0.001, print_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    iteration = 0
    
    data_iter = torchtext.data.BucketIterator(data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.text),
                                              shuffle=True,
                                              sort_within_batch=True)

    loss_data = []
    avg_loss = 0

    for e in range(num_epochs):
        # get training set
        for (tweet, lengths), label in data_iter:
            target = tweet[:, 1:] # the tweet from index 1 to end
            inp = tweet[:, :-1] # the entire tweet
            # cleanup
            optimizer.zero_grad()
            # forward pass
            output, _ = model(inp.cuda())
            loss = criterion(output.reshape(-1, vocab.vocab_size).cuda(), target.reshape(-1).cuda())
            # backward pass
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            iteration += 1 # increment iteration count
            if iteration % print_every == 0 and len(sys.argv) == 1:
                print("[Iter %d] Loss %f" % (iteration, float(avg_loss/print_every)))
                loss_data.append(avg_loss/print_every)
                # print("    " + sample_sequence(model, 140, 0.8))
                avg_loss = 0
                
        if len(sys.argv) == 1:
            print("[Finished %d Epochs]" % (e + 1))



    if len(sys.argv) == 1:
        plt.title("Loss Training Curve")
        plt.plot(loss_data, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show() 

if __name__ == "__main__":
    
    abc, text_field = get_data()
    v = Vocabulary(abc, text_field)
    vocab_stoi = v.vocab_stoi
    vocab_itos = v.vocab_itos
    vocab_size = v.vocab_size

    model = Generator(vocab_size, 64)

    train(model, abc, v,  batch_size=32, num_epochs=50, lr=0.005, print_every=50)

    print(sample_sequence(model, v, max_len=500, temperature=0.4))