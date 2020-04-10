# Sentiment Analysis with Music Generation
## About ##
The goal of this project is to create unique music based on text given by a user. Given a sad story it will generate sad music. Likewise given a happy tale we get joyful music. This can be used to content creators to generate music based on video and movie scripts and can also be used while reading a novel to generate music in the background based on the read text.

This project was created by two University of Toronto Students, Ajitesh Misra and Sharven Prasad Dhanasekar 

## Installation and Usage ##
#### Repository Structure ####
*TODO Sharven: Come back here after the repo is cleaned*

    ├── readme_resources        # The resources loaded in the README (e.g. the game screenshot)
    ├── src                     # The source code for the game
    │   ├── dataset             # The Sudoku boards in text file format that will be loaded by the game
    |   |   ├── debug           # The Sudoku boards in text file format that can be used for debugging
    ├── test                    # The testing scripts that can be used to test parts of the project
    ├── LICENSE
    └── README.md          
          
1. Clone the repository using the `Clone or Download` button
2. Run `main.py` in the directory `Python Files`
- You will be prompted to input text that will be used to generate music \
*Use the following instructions as we resolve a bug with the abc2midi conversion library:*
- After inputting text, it will output a file called `out.abc` in the same directory as `main.py`
- Copy/Paste the contents of this text file to [online abc to midi converter](https://www.mandolintab.net/abcconverter.php) and download the `.midi` file generated
- This `.midi` file can then be opened in an audio editing software such as [Audacity](https://www.audacityteam.org/) to view and play the generated musical notes

## Model Architecture ##

#### Sentiment Analysis Model ####

This model was trained on the [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset. This dataset contains 1.6 million tweets labelled as postive or negative. We chose to use only postive and negative as our clases to keep the problem simple.

##### Architecture: #####

We used a bi-directional GRU Recurent Neural Net model. Out of all the GRU activiation hidden vectors, the maximum vector as well as the mean vector is then concatenated and fed into a Fully Connected MLP for classification. This MLP network only has one layer and outputs a the 2D vector that corresponds to the sentiment.

##### How we classify the sentiment: #####

We first build a vocab dictionary of characters that are found in the training dataset. This dictionary assigns an index to each character which will be used to encode that character into a one-hot vector. 

The inputted text's characters is converted into a series of one-hot vectors using the previously mentioned vocab dictionary and this series of one-hot vectors is fed into the GRU.

Since we are batching the dataset, all text sequences in a batch are padded with a "padding character" so that the input length is the same for all examples in a batch. If a character in the example is not found in the dictionary, it is encoded as an "unknown character".

After feeding in the batch into the GRU, we concatenate the max activation and mean activation of GRU and feed it into a MLP to get the final sentiment prediction. This prediction is then backpropagated on using Adam as the optimizer and Cross Entropy Loss as the loss function. 

##### Final Performance #####
Hyperparameter Selection:
- Training Batch Size = 1024
- Epochs = 5
- Learning Rate = 0.004
- GRU Hidden Size = 100

Final Test Accuracy ~**83%**

| Training Curve  | Confusion Matrix |
| ------------- | ------------- |
| ![Sentiment Analysis Training Curve](/Train-Validation-Graph/allData30Epochs.JPG)  | *TODO Sharven: Add confusion matrix*  |

#### Music Generation Model ####

To generate music we used [ABCnotation](https://abcnotation.com/learn). This format gives a versitile way to storing music as raw text. 

The AI generates music in this format, which can then be changed into an audio format like midi. 

##### Architecture: #####

The model uses one hots to encode different characters into vectors which are much easier to work with. 

Our model has two layers, a GRU and a fully connected. The GRU we used is a single layer. 
The GRU is used to make sure the character we generate on this pass depends on the last character. That way as we generate character a pattern forms. 

##### How we generate music: #####

The AI is trained on a two datasets, one for happy songs and one for sad songs. We seperated songs into happy and sad based on the key. 
For example a song in A major would be happy while a song in A minor would be sad. Both models are the same, the only difference is the data its trained on. 

Once the sentiment of the text is found, a sample is sequenced from the correct model. We do not want the same characters from the model each time, so what we do is generate a multinomial distribution of all the possible characters we could. The way we generate text also takes in a integer called temperature. The higher the temperature, the more uniform the character distribution is. What this does is it ensures we do not always get the same characters when we run the model. The higher the temperature makes the text more random. We sample one character at a time, only stopping once we reach the character limit. 

## Licensing ##

*TODO: Pick licence and paste blurb*

## Bibliography ##

*TODO: Put all papers and libraries used here*
