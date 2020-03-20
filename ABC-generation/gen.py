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

    with open(f'{sys.argv[3]}.abc', 'w') as writer:
        writer.write(generated_sequence)

    print(generated_sequence)
    
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
    abc = torchtext.data.TabularDataset("happy.csv", "csv", fields)

    return abc, text_field

def train_model(data, vocab, batch_size=8, num_epochs=1, lr=0.001, print_every=100, check_point_interval=1):
    
    model = Generator(v.vocab_size, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    iteration = 0
    loss_data = []
    avg_loss = 0
    curr_epochs = 0

    if sys.argv[1] != "-n":
        saved_dictionary = torch.load(f"saved_models/{sys.argv[1]}.pth")
        model.load_state_dict(saved_dictionary['model_state_dict'])
        optimizer.load_state_dict(saved_dictionary['optimizer_state_dict'])
        curr_epochs = saved_dictionary['epoch']
        loss = saved_dictionary['loss']
        loss_data = [loss.item()]
        avg_loss = loss.item()
        iteration = saved_dictionary['iteration']
        model.train()
        print(f"Resuming Training at Epoch-{curr_epochs}")
    
    data_iter = torchtext.data.BucketIterator(data,
                                              batch_size=batch_size,
                                              sort_key=lambda x: len(x.text),
                                              shuffle=True,
                                              sort_within_batch=True)

    for epoch in range(curr_epochs, curr_epochs + num_epochs):
        # get training set
        for (sequence, lengths), label in data_iter:
            target = sequence[:, 1:] # the tweet from index 1 to end
            inp = sequence[:, :-1] # the entire tweet
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
            if iteration % print_every == 0:
                print("[Iter %d] Loss %f" % (iteration, float(avg_loss/print_every)))
                loss_data.append(avg_loss/print_every)
                # print("    " + sample_sequence(model, 140, 0.8))
                avg_loss = 0
                
        print("[Finished %d Epochs]" % (epoch + 1))

        if (epoch + 1) % check_point_interval == 0:
            # Save model
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab' : vocab
            }, f"saved_models/{sys.argv[2]}-{epoch+1}.pth")
            print(f"Reached Checkpoint [{epoch+1}]: Saved model to saved_models/{sys.argv[2]}-{epoch+1}.pth")


    plt.title("Loss Training Curve")
    plt.plot(loss_data, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show() 

    return model


def interactive(model, vocab):
    while True:
        temp = 0
        while True:
            temp = input("enter temperature or enter 'exit' to quit\n")

            if temp == "exit":
                return

            try:
                float(temp)
                break
            except ValueError:        
                print("please enter a valid float")


        sample_sequence(model, vocab, 500, float(temp))
        print("\n\n\n")


if __name__ == "__main__":
    
    interactive_mode = False
    if len(sys.argv) != 4:
        print("Usage: py gen.py load_name save_name abc_file_name")
        print("put '-i' for save mode to load in interactive mode")
        print("Let load_name = -n if training new model from scratch")
        exit(0)

    if sys.argv[2] == "-i":
        interactive_mode = True

    if not interactive_mode:
        abc, text_field = get_data()
        v = Vocabulary(abc, text_field)
        vocab_stoi = v.vocab_stoi
        vocab_itos = v.vocab_itos

        model = train_model(abc, v,  batch_size=32, num_epochs=10, lr=0.005, print_every=50, check_point_interval=1)

        sample_sequence(model, v, max_len=500, temperature=0.4)
    else:
        saved_dictionary = torch.load(f"saved_models/{sys.argv[1]}.pth")
        vocab = saved_dictionary['vocab']
        gen = Generator(vocab.vocab_size, 64)
        gen.load_state_dict(saved_dictionary['model_state_dict'])
        gen.eval()
        print("Loaded model from ../saved_models/", sys.argv[1] + ".pth")
        interactive(gen, vocab)