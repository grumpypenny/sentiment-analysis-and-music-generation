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
import random

EMO_TO_CLASS = {"excited" : 0,
                "anger" : 1, 
                "worry" : 2,
                "sad" : 3,
                "relief" : 4,
                "love" : 5,
                "happy" : 6,
                "neutral" : 7}

S_140_KEY= {"negative": 0,
            "neutral": 1,
            "positive": 2}

class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bi=False):
        super(SentimentGRU, self).__init__()
        self.name = ""
        self.emb = nn.Embedding.from_pretrained(glove.vectors).cuda()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bi).cuda()
        
        factor = int(2 + (int(bi) * 2))
        self.fcX = nn.Linear(hidden_size*factor, 200).cuda()
        self.fc1 = nn.Linear(200, num_classes).cuda()

        # self.fcX = nn.Linear(hidden_size*2, num_classes).cuda()
    
    def forward(self, x):
        # Convert x to embedding
        # print(x.shape)
        x = self.emb(x).cuda()
        # print(x.shape)
        # exit(0)

        # Set an initial hidden state
        h0 = torch.zeros(1, x.shape[0], self.hidden_size).cuda()
       
        # Forward propagate the GRU 
        out, _ = self.rnn(x)

        # Get the max and mean vector and concatenate them        
        out = torch.cat([torch.max(out, dim=1)[0], 
                        torch.mean(out, dim=1)], dim=1).cuda() 

        out = self.fcX(out).cuda()
        out = self.fc1(F.relu(out).cuda()).cuda()
        
        return out

    def save_model(self, name):
        if name:
            torch.save(self.state_dict(), "../Models/"+name+".pth")
            print(f"Saved model to ../Models/{name}.pth")

def get_emotion(class_num):
    
    for pair in S_140_KEY.items():
    # for pair in EMO_TO_CLASS.items():
        if class_num == pair[1]:
            return pair[0]

def get_class_num(emotion):
    return S_140_KEY[emotion]
    # return EMO_TO_CLASS[emotion]

def train_rnn_network(model, train, train_batch, valid, num_epochs=5, learning_rate=1e-5, checkpoint=None):

    print(f"Training for {num_epochs} epochs with lr={learning_rate}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for batch in train_batch:
            tweets, labels = batch
            optimizer.zero_grad()
            pred = model(tweets.cuda())
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
    for p in data:
        tweet, label = p
        tweet = tweet.unsqueeze(0)
        output = model(tweet.cuda())
        pred = output.max(1, keepdim=True)[1]
        correct += int(pred.cuda() == label.cuda())
        total += 1
    return correct / total

def split_tweet(tweet):
    # separate punctuations
    tweet = tweet.replace(".", " . ") \
                 .replace(",", " , ") \
                 .replace(";", " ; ") \
                 .replace("?", " ? ")

    return tweet.split()

def get_tweet_vectors(glove, f, glove_size, split, shuffle=False):

    if sum(split) != 1:
        print("Split does not sum to 100%")
        exit(0)

    reader = csv.reader(f)
    
    train, valid, test = [], [], []

    all_data = []

    for i, line in enumerate(reader):
        tweet = line[1]
        idxs = [glove.stoi[w]        # lookup the index of word
                for w in split_tweet(tweet)
                if w in glove.stoi] # keep words that has an embedding

        if not idxs: # ignore tweets without any word with an embedding
            continue

        idxs = torch.tensor(idxs) # convert list to pytorch tensor
        label = torch.tensor(get_class_num(line[0])).long()

        all_data.append((idxs, label))

    if shuffle:
        random.shuffle(all_data)

    n = len(all_data)
    train_n = int(split[0]*n)
    valid_n = int(split[1]*n)
    test_n = int(split[2]*n)

    train = all_data[:train_n]
    valid = all_data[train_n:train_n+valid_n]
    test = all_data[train_n+valid_n:train_n+valid_n+test_n]

    return train, valid, test

def find_max_length(dataset):
    m = -1
    for data in dataset:
        if len(data[0]) > m:
            m = len(data[0])
    return m

def make_batches(dataset, batch_size, vector_size):

    batches = []

    batch = []
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if batch and len(batch) % batch_size == 0:
            # Batch max length
            max_length = find_max_length(batch)

            tensor_batch = [[0 for i in range(max_length)] for j in range(batch_size)]
            labels = []

            for i in range(batch_size):
                for j in range(len(batch[i][0])):
                    tensor_batch[i][j] = int(batch[i][0][j])
                labels.append(batch[i][1])

            batches.append((torch.tensor(tensor_batch), torch.tensor(labels)))
            batch = []

    return batches

if __name__ == "__main__":
    
    BATCH_SIZE = 128
    GLOVE_SIZE = 100
    NUM_CLASSES = 3
    # NUM_CLASSES = 8

    glove = torchtext.vocab.GloVe(name="6B", dim=GLOVE_SIZE, max_vectors=10000)
    f = open("../Data/s140_500tweets.csv", "rt")

    # 0.6, 0.2, 0.2 split, respectively
    train, valid, test = get_tweet_vectors(glove, f, GLOVE_SIZE, [0.6, 0.2, 0.2], shuffle=True)
    
    # Use only for crowd flower
    # excited = []
    # angry = []
    # relief = []
    # love = []
    # happy = []

    # for item in train:
    #     label = int(item[1])
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
    # train = train + excited * 2
    # train = train + angry * 6
    # train = train + relief * 6
    # train = train + love * 2
    # train = train + happy * 2

    print(f"Training Dataset: {len(train)}; Validation Dataset:{len(valid)}; Testing Dataset:{len(test)}")
    train_batches = make_batches(train, BATCH_SIZE, GLOVE_SIZE)
    model_gru = SentimentGRU(GLOVE_SIZE, 100, NUM_CLASSES, bi=True)

    # Confusion Matrix Generation Mode
    if 4 <= len(sys.argv) <= 5 and sys.argv[1] == "-c":
        epochs = int(sys.argv[2])
        lrate = float(sys.argv[3])
        csv_name = ""

        if len(sys.argv) >= 5:
            csv_name = sys.argv[4]
        
        train_rnn_network(model_gru, train, train_batches, valid, num_epochs=epochs, learning_rate=lrate)
        print("Test Accuracy:", get_accuracy(model_gru, test))
       
        confusion_matrix = []
        for i in range(NUM_CLASSES):
            a = [0]
            confusion_matrix.append(a*NUM_CLASSES)

        for data in test:
            tweet, label = data
            tweet = tweet.unsqueeze(0)
            output = model_gru(tweet.cuda())
            pred = output.max(1, keepdim=True)[1]

            pred_idx = int(pred)            
            true_idx = int(label)

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

    # Training Mode
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
        train_rnn_network(model_gru, train, train_batches, valid, num_epochs=epochs, learning_rate=lrate, checkpoint=checkpoint_interval)
        print("Test Accuracy:", get_accuracy(model_gru, test))
        
        if not checkpoint_interval:
            model_gru.save_model(model_name)

    else:
        print("Bad Usage")
                
