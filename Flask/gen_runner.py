import torch
import torch.nn as nn
import torch.optim as optim

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

def sample_sequence(model, vocab, max_len=100, temperature=0.5, output_file=True, print_out=False):
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

    if print_out:
        print(generated_sequence)

    if output_file:
        with open(f'{sys.argv[3]}.abc', 'w') as writer:
            writer.write(generated_sequence)
        return None
    else:
        return generated_sequence