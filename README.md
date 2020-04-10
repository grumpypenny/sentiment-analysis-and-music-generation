# Sentiment Analysis with Music Generation
## About ##
The goal of this project is to create unique music based on text given by a user. Given a sad story it will generate sad music. Likewise given a happy tale we get joyful music. This can be used to content creators to generate music based on video and movie scripts and can also be used while reading a novel to generate music in the background based on the read text.

This project was created by two University of Toronto Students, Ajitesh Misra and Sharven Prasad Dhanasekar 

## Installation and Usage ##
#### Repository Structure ####

    ├── ABC-Generation              # Where Music generation models are trained (not required to use application)
    ├── Data                        # The datasets for sentiment analysis training
    ├── Models                      # The models used by the application
    ├── Music Generation Curves     # The trainings curves of the music generation models
    ├── Readme Resources            # The resources used by the README file
    ├── Sample Music                # Some sample music generated as placeholders
    ├── Sentiment Curves            # The training curves of the sentiment analysis model
    │   ├── Confusion Matrices      # Generated confusion matrices while training
    │   ├── Crowd Flower            # The training curves on the crowd-flower sentiment dataset
    │   ├── S140                    # The training curves on the sentiment140 dataset
    ├── Source Files                # The source Python files for the models and data processing
    │   ├── abcmidi                 # The library used to convert from ABC to MIDI
    │   ├── tempABC                 # The generated ABC file from the given input will be placed here
    ├── .gitignore
    ├── LICENSE
    └── README.md          
          
1. Clone the repository using the `Clone or Download` button
2. Run `main.py` in the directory `Source Files`
- You will be prompted to input text that will be used to generate music \
*Use the following instructions as we resolve a bug with the abc2midi conversion library:*
- After inputting text, it will output a file called `song.abc` in the directory as `Source Files/tempABC`
- Copy/Paste the contents of this file to [online abc to midi converter](https://www.mandolintab.net/abcconverter.php) and download the `.midi` file generated
- This `.midi` file can then be opened in an audio editing software such as [Audacity](https://www.audacityteam.org/) to view and play the generated musical notes

## Model Architecture ##

We used three seperate models that are linked together:
- Sentiment Analysis model predicts the sentiment in the given text
- Two identical Music Generation models, one is trained on happy music and the other on sad music 

If the predicted sentiment is sad, we sample from the sad music generation model. Otherwise, we sample from the happy music genration model.

### Sentiment Analysis Model ###

This model was trained on the [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset. This dataset contains 1.6 million tweets labelled as postive or negative. We chose to use only postive and negative as our clases to keep the problem simple.

**Architecture:**

We used a bi-directional GRU Recurent Neural Net model. Out of all the GRU activiation hidden vectors, the maximum vector as well as the mean vector is then concatenated and fed into a Fully Connected MLP for classification. This MLP network only has one layer and outputs a the 2D vector that corresponds to the sentiment.

**How we classify the sentiment:**

We first build a vocab dictionary of characters that are found in the training dataset. This dictionary assigns an index to each character which will be used to encode that character into a one-hot vector. 

The inputted text's characters is converted into a series of one-hot vectors using the previously mentioned vocab dictionary and this series of one-hot vectors is fed into the GRU.

Since we are batching the dataset, all text sequences in a batch are padded with a "padding character" so that the input length is the same for all examples in a batch. If a character in the example is not found in the dictionary, it is encoded as an "unknown character".

After feeding in the batch into the GRU, we concatenate the max activation and mean activation of GRU and feed it into a MLP to get the final sentiment prediction. This prediction is then backpropagated on using Adam as the optimizer and Cross Entropy Loss as the loss function. 

**Final Performance**
Hyperparameter Selection:
- Training Batch Size = 1024
- Epochs = 10
- Learning Rate = 0.004
- GRU Hidden Layer Size = 100

Final Test Accuracy ~**83%**

<img src="Readme%20Resources/SentimentLoss.png">
<img src="Readme%20Resources/SentimentAcc.png">
<img src="Readme%20Resources/SentimentConfusion.png">

### Music Generation Model ###

To generate music we used [ABCnotation](https://abcnotation.com/learn). This format gives a versitile way to storing music as raw text. The AI generates music in this format, which can then be changed into an audio format like midi. 

The reason why we picked this format is because it allows us to to easily generate music with weaker machines. 

**Architecture:**

The model uses one hots to encode different characters into vectors which are much easier to work with. 

Our model has two layers, a GRU and a fully connected. The GRU we used is a single layer. 
The GRU is used to make sure the character we generate on this pass depends on the last character. That way as we generate character a pattern forms. 

**How we generate music:**

The AI is trained on a two datasets, one for happy songs and one for sad songs. We seperated songs into happy and sad based on the key. 
For example a song in A major would be happy while a song in A minor would be sad. Both models are the same, the only difference is the data its trained on. 

Once the sentiment of the text is found, a sample is sequenced from the correct model. We do not want the same characters from the model each time, so what we do is generate a multinomial distribution of all the possible characters we could. The way we generate text also takes in a integer called temperature. The higher the temperature, the more uniform the character distribution is. What this does is it ensures we do not always get the same characters when we run the model. The higher the temperature makes the text more random. We sample one character at a time, only stopping once we reach the character limit. 

**Final Performance**
Hyperparameter Selection:
- Training Batch Size = 32
- Epochs = 10
- Learning Rate = 0.005
- GRU Hidden Layer Size = 64

*TODO Ajitesh: Add training curves for music generation*

## Licensing ##

[GPLv3 Licensing Information](https://github.com/grumpypenny/sentiment-analysis-and-music-generation/blob/master/LICENSE)

## References ##

- Much of our code is based on these [course notes](https://www.cs.toronto.edu/~lczhang/360/) by Lisa Zhang
- The ABC files we used for training can be found [Here](https://abcnotation.com/tunes)
