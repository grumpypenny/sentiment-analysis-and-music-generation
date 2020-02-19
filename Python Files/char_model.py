# -*- coding: utf-8 -*-

import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import sys
import csv

EMO_TO_CLASS = {"excited" : 0,
                "anger" : 1, 
                "worry" : 2,
                "sad" : 3,
                "relief" : 4,
                "love" : 5,
                "happy" : 6,
                "neutral" : 7}

# S_140_KEY= {"negative": 0,
#             "neutral": 1,
#             "positive": 2}


S_140_KEY= {"negative": 0,
            "positive": 1}


class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bi=False):
        super(SentimentGRU, self).__init__()

        self.name = ""

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bi).cuda()
        
        factor = int(2 + (int(bi) * 2))
        self.fcX = nn.Linear(hidden_size*factor, 200).cuda()
        self.fc1 = nn.Linear(200, 70).cuda()
        self.fc2 = nn.Linear(70, 10).cuda()
        self.fc3 = nn.Linear(10, num_classes).cuda()

        # self.fcX = nn.Linear(hidden_size*2, num_classes).cuda()
    
    def forward(self, x):
        # Convert x to one hot
        ident = torch.eye(self.input_size)
        x = ident[x].cuda()

        # Set an initial hidden state
        h0 = torch.zeros(1, x.shape[0], self.hidden_size).cuda()
       
        # Forward propagate the GRU 
        out, _ = self.rnn(x)

        # Get the max and mean vector and concatenate them        
        out = torch.cat([torch.max(out, dim=1)[0], 
                        torch.mean(out, dim=1)], dim=1).cuda() 
       
        out = self.fcX(out).cuda()
        out = self.fc1(F.relu(out).cuda()).cuda()
        out = self.fc2(F.relu(out).cuda()).cuda()
        out = self.fc3(F.relu(out).cuda()).cuda()
        return out

    def save_model(self, name):
        if name:
            torch.save(self.state_dict(), "../Models/"+name+".pth")
            print(f"Saved model to ../Models/{name}.pth")

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

def get_emotion(class_num):
    
    for pair in S_140_KEY.items():
    # for pair in EMO_TO_CLASS.items():
        if class_num == pair[1]:
            return pair[0]

def get_class_num(emotion):
    return S_140_KEY[emotion]
    # return EMO_TO_CLASS[emotion]

if __name__ == "__main__":

    BATCH_SIZE = 64
    NUM_CLASSES = 2
    # NUM_CLASSES = 8
    
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
                                    preprocessing=lambda x: S_140_KEY[x]) # convert text to 0 and 1
                                    # preprocessing=lambda x: EMO_TO_CLASS[x])

    fields = [('label', label_field), ('tweet', text_field)]
    dataset = torchtext.data.TabularDataset("../Data/s140_2000tweets.csv", # name of the file
                                            "csv",               # fields are separated by a tab
                                            fields)

    # 0.6, 0.2, 0.2 split, respectively
    train, valid, test = dataset.split([0.6, 0.2, 0.2])

    # # Use only for crowd flower
    # excited = []
    # angry = []
    # relief = []
    # love = []
    # happy = []

    # for item in train.examples:
    #     label = item.label
    #     if label == 0:
    #         excited.append(item)
    #     elif label == 1:
    #         angry.append(item)
    #     elif label == 4:
    #         relief.append(item)
    #     elif label == 5:
    #         love.append(item)
    #     elif label == 6:
    #         happy.append(item)
        
    # # duplicate each spam message 6 more times
    # train.examples = train.examples + excited * 2
    # train.examples = train.examples + angry * 6
    # train.examples = train.examples + relief * 6
    # train.examples = train.examples + love * 2
    # train.examples = train.examples + happy * 2

    print(f"Training Dataset: {len(train)}; Validation Dataset:{len(valid)}; Testing Dataset:{len(test)}")

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


    model_gru = SentimentGRU(len(text_field.vocab.stoi), 100, NUM_CLASSES, True)

    if 4 <= len(sys.argv) <= 5 and sys.argv[1] == "-c":
        epochs = int(sys.argv[2])
        lrate = float(sys.argv[3])
        csv_name = ""

        if len(sys.argv) >= 5:
            csv_name = sys.argv[4]
        
        train_rnn_network(model_gru, train_iter, valid_iter, num_epochs=epochs, learning_rate=lrate)
        print("Test Accuracy:", get_accuracy(model_gru, test_iter))
       
        confusion_matrix = []
        for i in range(NUM_CLASSES):
            a = [0]
            confusion_matrix.append(a*NUM_CLASSES)

        for tweet, label in test_iter:
            # tweet = tweet.unsqueeze(0)
            output = model_gru(tweet[0].cuda())
            pred = output.max(1, keepdim=True)[1]
            
            for i in range(pred.shape[0]):
                pred_idx = int(pred[i][0])
                true_idx = int(label[i])

                confusion_matrix[true_idx][pred_idx] += 1
        
        print("Confusion Matrix:")
        for row in confusion_matrix:
            print(row)

        if csv_name:
            with open(f"../../Sentiment Analysis Model Report/{csv_name}.csv", "w+", encoding='utf-8', newline='') as new_data:

                writer = csv.writer(new_data)
                # header = [""] + list(range(NUM_CLASSES))
                header = [""]

                for i in range(NUM_CLASSES):
                    header += [get_emotion(i)]
                
                writer.writerow(header)

                i = 0
                for row in confusion_matrix:
                    # indexed_row = [i] + row
                    indexed_row = [get_emotion(i)] + row
                    writer.writerow(indexed_row)
                    i += 1

            print("")
            print(f"Saved to: ../../Sentiment Analysis Model Report/{csv_name}.csv")

    elif len(sys.argv) >= 4 and sys.argv[1] == "-t":
        
        epochs = int(sys.argv[2])
        lrate = float(sys.argv[3])
        model_name = ""
        checkpoint_interval = None

        if len(sys.argv) >= 5:
            model_name = sys.argv[4]
        if len(sys.argv) == 6:
            checkpoint_interval = int(sys.argv[5])

        model_gru.name = model_name
        train_rnn_network(model_gru, train_iter, valid_iter, num_epochs=epochs, learning_rate=lrate, checkpoint=checkpoint_interval)
        print("Test Accuracy:", get_accuracy(model_gru, test_iter))

        if not checkpoint_interval:
            model_gru.save_model(model_name)

    else:
        print("Bad Usage")
                
