import torch
import torchtext
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

"""
Source code:
https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556
"""
BATCH_SIZE = 128
NUM_CLASSES = 5
EPOCHS = 10
LR = 0.004
HIDDEN_SIZE = 100


CF_KEY = {"sad" : 0,
          "happy" : 1,
          "anger" : 2,
          "relief" : 3}

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, hidden_size, 2)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.p2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)
        
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)
        
        p = torch.tanh(p)
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)
        output = output.view(conv_seq_len * batch_size, self.hidden_size) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = torch.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden

def train_rnn_network(model, train, valid, num_epochs=5, learning_rate=1e-5, checkpoint=None):

    print(f"Training for {num_epochs} epochs with lr={learning_rate}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for msg, labels in train:
            optimizer.zero_grad()
            pred = model(msg[0].cuda())
            loss = criterion(pred.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train))
        valid_acc.append(get_accuracy(model, valid))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))

        if checkpoint and (epoch + 1) % checkpoint == 0:
            model.save_model(model.name + f"-{epoch + 1}")
    
    # Plot curves
    plt.title("Loss Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Accuracy Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

def get_accuracy(model, data):
    """ Compute the accuracy of the `model` across a dataset `data`
    
    Example usage:
    
    >>> model = MyRNN() # to be defined
    >>> get_accuracy(model, valid) # the variable `valid` is from above
    """
    correct, total = 0 , 0
    for msg, labels in data:
        output = model(msg[0].cuda())
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.cuda().view_as(pred.cuda())).sum().item()
        total += labels.shape[0]
    return correct / total

def convert_to_stoi(vocab, string):

    stoi = []
    for s in string:
        # If character is in vocab, append its index
        # else, append the index of <unk>
        if s in vocab:
            stoi.append(vocab[s])
        else:
            stoi.append(vocab['<unk>'])
    
    return stoi

# set up datafield for messages
text_field = torchtext.data.Field(sequential=True,    # text sequence
                                tokenize=lambda x: x, # because are building a character-RNN
                                include_lengths=True, # to track the length of sequences, for batching
                                batch_first=True,
                                use_vocab=True)       # to turn each character into an integer index

# set up datafield for labels
label_field = torchtext.data.Field(sequential=False, # not a sequence
                                use_vocab=False,     # don't need to track vocabulary
                                is_target=True,      
                                batch_first=True,
                                preprocessing=lambda x: CF_KEY[x]) # convert text to 0 and 1
                                # preprocessing=lambda x: EMO_TO_CLASS[x])

fields = [('label', label_field), ('tweet', text_field)]
dataset = torchtext.data.TabularDataset("../Data/only4.csv", # name of the file
                                        "csv",               # fields are separated by a tab
                                        fields)

# 0.6, 0.2, 0.2 split, respectively
train, valid, test = dataset.split([0.6, 0.2, 0.2])

# # Use only for crowd flower
sad = []
happy = []
anger = []
relief = []

for item in train.examples:
    label = item.label
    if label == 0:
        sad.append(item)
    elif label == 1:
        happy.append(item)
    elif label == 2:
        anger.append(item)
    elif label == 3:
        relief.append(item)

train.examples = train.examples + anger * 10
train.examples = train.examples + relief * 9

print(f"Training Dataset: {len(train)}; Validation Dataset:{len(valid)}; Testing Dataset:{len(test)}")
print(f"sad: {len(sad)}; happy: {len(happy)}; anger: {len(anger)}; relief: {len(relief)}")

# build a vocabulary of every character in dataset
text_field.build_vocab(dataset)    
vocab = text_field.vocab.stoi

train_iter = torchtext.data.BucketIterator(train,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,                  # Shuffle the training dataset
                                        sort_key=lambda x: len(x.tweet), # to minimize padding
                                        sort_within_batch=True,        # sort within each batch
                                        repeat=False)                  # repeat the iterator for many epochs

test_iter = torchtext.data.BucketIterator(test,
                                        batch_size=BATCH_SIZE,
                                        sort_key=lambda x: len(x.tweet), # to minimize padding
                                        sort_within_batch=True,        # sort within each batch
                                        repeat=False)                  # repeat the iterator for many epochs

valid_iter = torchtext.data.BucketIterator(valid,
                                        batch_size=BATCH_SIZE,
                                        sort_key=lambda x: len(x.tweet), # to minimize padding
                                        sort_within_batch=True,        # sort within each batch
                                        repeat=False)                  # repeat the iterator for many epochs

rnn = RNN(len(text_field.vocab.stoi), HIDDEN_SIZE, NUM_CLASSES, 2)
train_rnn_network(rnn, train_iter, valid_iter, num_epochs=EPOCHS, learning_rate=LR)

print("Test Accuracy:", get_accuracy(rnn, test_iter))