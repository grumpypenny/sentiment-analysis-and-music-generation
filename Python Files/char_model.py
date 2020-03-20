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

CF_KEY = {"sad" : 0,
          "happy" : 1,
        #   "neutral" : 2,
          "anger" : 2,
          "relief" : 3}

S_140_KEY= {"negative": 0,
            "positive": 1}


class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bi=False, layers=1):
        super(SentimentGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bi, num_layers=layers).cuda()
        
        factor = int(2 + (int(bi) * 2))
        # self.fcX = nn.Linear(hidden_size*factor, 200).cuda()
        # self.fc1 = nn.Linear(200, 70).cuda()
        # self.fc2 = nn.Linear(70, 10).cuda()
        # self.fc3 = nn.Linear(10, num_classes).cuda()

        self.fc = nn.Linear(hidden_size*factor, num_classes).cuda()
    
    def forward(self, x):
        # Convert x to one hot
        ident = torch.eye(self.input_size)
        x = ident[x].cuda()

        # Set an initial hidden state
        h0 = torch.zeros(2, x.shape[0], self.hidden_size).cuda()
       
        # Forward propagate the GRU 
        out, _ = self.rnn(x, h0)

        # Get the max and mean vector and concatenate them        
        out = torch.cat([torch.max(out, dim=1)[0], 
                        torch.mean(out, dim=1)], dim=1).cuda() 
       
        # out = self.fcX(out).cuda()
        # out = self.fc1(F.relu(out).cuda()).cuda()
        # out = self.fc2(F.relu(out).cuda()).cuda()
        # out = self.fc3(F.relu(out).cuda()).cuda()

        out = self.fc(out).cuda()

        return out

def train_rnn_network(input_size, hidden_size, num_classes, train, valid, vocab, bid=False, num_layers=1, num_epochs=5, learning_rate=1e-5, checkpoint=1):

    model = SentimentGRU(input_size, hidden_size, num_classes, bi=bid, layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses, train_acc, valid_acc = [], [], []
    curr_epochs = 0
    epochs = []

    if sys.argv[1] != "-n":
        saved_dictionary = torch.load(f"../Models/{sys.argv[1]}.pth")
        model.load_state_dict(saved_dictionary['model_state_dict'])
        optimizer.load_state_dict(saved_dictionary['optimizer_state_dict'])
        curr_epochs = saved_dictionary['epoch']
        losses = saved_dictionary['losses']
        train_acc = saved_dictionary['train_acc']
        valid_acc = saved_dictionary['valid_acc']
        epochs = saved_dictionary['epoch_list']
        model.train()
        print(f"Resuming Training at Epoch-{curr_epochs}")

    for epoch in range(curr_epochs, curr_epochs + num_epochs):
        for msg, labels in train:
            optimizer.zero_grad()
            pred = model(msg[0].cuda())
            loss = criterion(pred.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train))
        valid_acc.append(get_accuracy(model, valid))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))

        if (epoch + 1) % checkpoint == 0:
            # Save model
            torch.save({
            'epoch': epoch+1,
            'epoch_list': epochs,
            'losses': losses,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab' : vocab
            }, f"../Models/{sys.argv[2]}-{epoch+1}.pth")
            print(f"Reached Checkpoint [{epoch+1}]: Saved model to ../Models/{sys.argv[2]}-{epoch+1}.pth")
            generate_confusion_matrix(model, num_classes)
    
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

    return model

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

def generate_confusion_matrix(model, num_classes):

    confusion_matrix = []
    for i in range(num_classes):
        a = [0]
        confusion_matrix.append(a*num_classes)

    for tweet, label in test_iter:
        # tweet = tweet.unsqueeze(0)
        output = model(tweet[0].cuda())
        pred = output.max(1, keepdim=True)[1]
        
        for i in range(pred.shape[0]):
            pred_idx = int(pred[i][0])
            true_idx = int(label[i])

            confusion_matrix[true_idx][pred_idx] += 1
    
    print("Confusion Matrix:")
    for row in confusion_matrix:
        print(row)

    # Use for saving to csv:
    # if csv_name:
    #     with open(f"../Confusion Matrices/{csv_name}.csv", "w+", encoding='utf-8', newline='') as new_data:

    #         writer = csv.writer(new_data)
    #         # header = [""] + list(range(num_classes))
    #         header = [""]

    #         for i in range(num_classes):
    #             header += [get_emotion(i)]
            
    #         writer.writerow(header)

    #         i = 0
    #         for row in confusion_matrix:
    #             # indexed_row = [i] + row
    #             indexed_row = [get_emotion(i)] + row
    #             writer.writerow(indexed_row)
    #             i += 1

    #     print("")
    #     print(f"Saved to: ../../Sentiment Analysis Model Report/{csv_name}.csv")

def interactive(model_gru, vocab):
    # Interactive Mode
    inp_string = ""

    softmax = nn.Softmax(dim=0)

    while True:
        inp_string = input("Enter message: ")

        if not inp_string:
            break # exit

        stoi = convert_to_stoi(vocab, inp_string)
        x_tensor = torch.tensor([stoi])

        raw_pred = model_gru(x_tensor)
        pred = raw_pred.max(1, keepdim=True)[1]
        pred_idx = int(pred[0][0])

        print(f"{get_emotion(pred_idx)}; Confidence: {softmax(raw_pred[0])[pred_idx]}\n")


if __name__ == "__main__":

    interactive_mode = False

    if len(sys.argv) != 4:
        print("Usage: py char_model.py [load_name] [save_name] [data_set_name]")
        print("or")
        print("py char_model [load_name] -i [data_set_name] for interactive mode")
        print("Let load_name = -n if training new model from scratch")
        exit(0)
    
    if sys.argv[2] == "-i":
        print("entering interactive mode")
        interactive_mode = True

    BATCH_SIZE = 1024
    NUM_CLASSES = 2
    EPOCHS = 5
    LR = 0.004
    HIDDEN_SIZE = 100
    DATA_SET_NAME = sys.argv[3]
    # NUM_CLASSES = 8

    if not interactive_mode:

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
        train = torchtext.data.TabularDataset(f"../Data/{DATA_SET_NAME}-train.csv", # name of the file
                                                "csv",               # fields are separated by a tab
                                                fields)

        valid = torchtext.data.TabularDataset(f"../Data/{DATA_SET_NAME}-validation.csv", # name of the file
                                                "csv",               # fields are separated by a tab
                                                fields)

        test = torchtext.data.TabularDataset(f"../Data/{DATA_SET_NAME}-test.csv", # name of the file
                                                "csv",               # fields are separated by a tab
                                                fields)


        sad = []
        happy = []

        for item in train.examples:
            label = item.label
            if label == 0:
                sad.append(item)
            elif label == 1:
                happy.append(item)

        print(f"sad = {len(sad)}, happy = {len(happy)}")

        print(f"Training Dataset: {len(train)}; Validation Dataset:{len(valid)}; Testing Dataset:{len(test)}")

        text_field.build_vocab(train)    
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

        model_gru = train_rnn_network(len(text_field.vocab.stoi), HIDDEN_SIZE, NUM_CLASSES, train_iter, 
                                  valid_iter, vocab, bid=True, num_epochs=EPOCHS, learning_rate=LR, checkpoint=5)
        print("Test Accuracy:", get_accuracy(model_gru, test_iter))
    
    else:
        saved_dictionary = torch.load(f"../Models/{sys.argv[1]}.pth")
        vocab = saved_dictionary['vocab']
        model_gru = SentimentGRU(len(vocab), HIDDEN_SIZE, NUM_CLASSES, bi=True, layers=1)
        model_gru.load_state_dict(saved_dictionary['model_state_dict'])
        model_gru.eval()
        print("Loaded model from ../Models/", sys.argv[2] + ".pth")
        interactive(model_gru, vocab)

