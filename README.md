# Sentiment Analysis with Music Generation
The goal of this project is to create unique music based on text given by a user. Given a sad story it will generate sad music. Likewise given a happy tale we get joyful music. This can be used to content creators to generate music based on video and movie scripts and can also be used while reading a novel to generate music in the background based on the read text.

This project was created by two University of Toronto Students, Ajitesh Misra and Sharven Prasad Dhanasekar 

## Table of Contents ##
* [Installation and Usage](#installation-and-usage)
* [Model Architecture](#model-architecture)
* [Converting ABC to Audio](#conversion-to-audio)
* [Licensing](#licensing)
* [References](#references)
* [Deployment](#deployment)

## Installation and Usage ##
#### Repository Structure ####

    ├── Data                        # The datasets used by the models
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
    |   ├── outputMIDI              # The generated MIDI file will be placed here
    ├── .gitignore
    ├── LICENSE
    ├── requirements.txt
    └── README.md          
          
1. Clone the repository using the `Clone or Download` button
2. Run `pip install -r requirements.txt` to make sure all packages are installed
3. Run `main.py` in the directory `Source Files`
- Command Line Arguments: 
    - the name of the sentiment analysis model in the `Models` folder (without the ".pth")
    - `test-10` is the best model included in the repo. It is what we recommend to use as the argument. 
4. You will be prompted to input text that will be used to generate music 
5. After inputting text, it will output a file called `song.abc` in the directory as `Source Files/tempABC`
    - If you are on Linux you can run `a2m.py` to convert the abc file to MIDI directly. The new file will be `Source Files/outputMIDI/out.midi`
    - If you are on Windows, the above will not work, instead use [this website](https://www.mandolintab.net/abcconverter.php) to convert the ABC to music. 

__NOTE:__

This code requires a GPU to run. 

## Model Architecture ##

We used three separate models that are linked together:
- Sentiment Analysis model predicts the sentiment in the given text
- Two identical Music Generation models, one is trained on happy music and the other on sad music 

If the predicted sentiment is sad, we sample from the sad music generation model. Otherwise, we sample from the happy music generation model.

### Saving and Loading Models ###

All the code saves and loads the models in such a way that they can resume training and provide predictions. All models are saved under the `Models` folder.

The convention we used is we save the model as `<model_name>-<number of epochs>`. For example, if we train a model named `example` for 20 epochs, `example-20.pth` will be the filename its saved as.

### Sentiment Analysis Model ###

This model was trained on the [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset. This dataset contains 1.6 million tweets labelled as positive or negative. We chose to use only positive and negative as our classes to keep the problem simple.

**Architecture:**

We used a bi-directional GRU Recurrent Neural Net model. Out of all the GRU activation hidden vectors, the maximum vector as well as the mean vector is then concatenated and fed into a Fully Connected MLP for classification. This MLP network only has one layer and outputs a the 2D vector that corresponds to the sentiment.

**How we classify the sentiment:**

We first build a vocab dictionary of characters that are found in the training dataset. This dictionary assigns an index to each character which will be used to encode that character into a one-hot vector. 

The inputted text's characters are converted into a series of one-hot vectors using the previously mentioned vocab dictionary and this series of one-hot vectors is fed into the GRU.

Since we are batching the dataset, all text sequences in a batch are padded with a "padding character" so that the input length is the same for all examples in a batch. If a character in the example is not found in the dictionary, it is encoded as an "unknown character".

After feeding in the batch into the GRU, we concatenate the max activation and mean activation of GRU and feed it into a MLP to get the final sentiment prediction. This prediction is then backpropagated on using Adam as the optimizer and Cross Entropy Loss as the loss function. 

**Final Performance**

Hyperparameter Selection:
- Training Batch Size = 1024
- Epochs = 10
- Learning Rate = 0.004
- GRU Hidden Layer Size = 100

Final Test Accuracy ~**83%**

__Training curves:__

<img src="Readme%20Resources/SentimentLoss.png">
<img src="Readme%20Resources/SentimentAcc.png">
<img src="Readme%20Resources/SentimentConfusion.png">

**Train a new Sentiment Analysis model**

1. Download the [sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140) and place it in the `Data` folder
2. Run `cleaner.py` in `Source Files`
- Command Line Arguments:
    - The load name of the dataset (in this case it is the sentiment140 dataset without the ".csv")
    - The save name that the datasets should be saved as
    - The amount of data that should be cleaned and processed (set it as -1 if you wish to process the full dataset)
    - The training split percentage (a decimal point number)
    - The validation split percentage (a decimal point number)
    - The test split percentage (a decimal point number)
    
*Note the split values need to sum to 1 for it to work properly*
- After running `cleaner.py`, you will see three files generated in the `Data` folder with the given save name followed by `-train.csv`, `-test.csv`, and `-validation.csv`
3. Run `char_model.py` in `Source Files`
- Command Line Arguments:
    - The name of the model you wish to continue training  (set it as -n if you wish to train from scratch)
    - The name that the model should be saved as
    - The name of the dataset used for training (in this case it would be the save name provided to `cleaner.py`)
- After training, it will display the training curve

Note that the hyperparameters of the model can be modified at the bottom of `char_model.py`.

### Music Generation Model ###

To generate music we used [ABCnotation](https://abcnotation.com/learn). This format gives a versatile way to storing music as raw text. The AI generates music in this format, which can then be changed into an audio format like midi. 

The reason why we picked this format is because it allows us to easily generate music with weaker machines. 

**Architecture:**

The model uses one hots to encode different characters into vectors which are much easier to work with. 

Our model has two layers, a GRU and a fully connected. The GRU we used is a single layer. 
The GRU is used to make sure the character we generate on this pass depends on the last character. That way as we generate character a pattern forms. 

**How we generate music:**

The AI is trained on two datasets, one for happy songs and one for sad songs. We separated songs into happy and sad based on the mode.
There are 7 common modes in music, in order from darkest to brightest:
* Locrian
* Phrygian
* Minor
* Dorian
* Mixolydian
* Major
* Lydian

We defined 'happy' as any song written in Major or Lydian mode, all else being called sad. It is important to note that Mixolydian is not always sad in practice but for our purposes we decided to group it in with sad as it is rarely as bright as Major or Lydian. 

For further information we used [this video](https://www.youtube.com/watch?v=jNY_ZCUBmcA)

Both models are the same, the only difference is the data its trained on. 

Once the sentiment of the text is found, a sample is sequenced from the correct model. We do not want the same characters from the model each time, so what we do is generate a multinomial distribution of all the possible characters that the model can put, then sample from the most likely characters. The way we generate text also takes in a integer called temperature. The higher the temperature, the more uniform the character distribution is. What this does is it ensures we do not always get the same characters when we run the model. The higher the temperature, the more random the text. We chose a temperature value of 0.6. We sample one character at a time, only stopping once we reach the character limit or we generate the end of string character. 

One the ABC has been generated we store it in a .abc file. We then use the `abcmidi` package, additional reading [here](http://abc.sourceforge.net/abcMIDI/original/). This package reads the .abc file and creates a MIDI of it. MIDI is a more useful format as it allows music playback. 

**Final Performance**

Hyperparameter Selection:
- Training Batch Size = 32
- Epochs = 200 for happy model and 30 for sad model
- Learning Rate = 0.005
- GRU Hidden Layer Size = 64

__Training curves:__

We trained both models for 10 epochs. Every 50 iterations we took the average loss and appended it to the graph. The reason happy goes on for much longer is that we have more happy songs in our dataset than sad songs.

The first graph is happy, the second one is sad. 

![Happy](Readme%20Resources/MusicHappy.png)
![Sad](Readme%20Resources/MusicSad.png)


**Train a new Music Generation model**

1. Run `gen.py` in `Source Files` with the following command line arguments:
    1. The name of an existing model you want to continue training or `-n` to train a new model
    2. The name you want the new model to be saved under or `-i` to load up the model interactively
    3. The filename you want the text output to be saved under
    4. the emotion you want to train. `happy` and `sad` are the only two options for now. 
2. The data is already in the repo, under `/ABC-generation/happy.csv` and `/ABC-generation/sad.csv`. Both have already been cleaned.

___NOTE:___

When training a sad model an unknown CUDA error may occur. If it happens then you can resume training the model from the last successful epoch. 

**Getting the data**

Here is how we got the data for generating the music:

1. First we downloaded abc files from [this collection](https://abcnotation.com/tunes), we recommend the *A cleaned version of the Nottingham dataset for machine learning research* as a start as it is well formatted and small enough to train simple models on. 
The GitHub link can be found [here](https://github.com/jukedeck/nottingham-dataset)

2. Place all the abc files inside one folder

3. Run `Data_Finder.py` giving it the path to the folder of abc files and the name of a csv to store the cleaned versions of them in.

4. To separate the data into happy and sad run `sorter.py` to create two new csv containing songs of only that emotion named `happy.csv` and `sad.csv` respectively. 

## Conversion to Audio ##

When running `main.py` it stores the output to `tempABC/song.abc`. To turn this into a MIDI file (an audio format that can be listened to) run `a2m.py`. This stores the output MIDI file under `outputMIDI/out.midi`. 

__NOTE:__

- This process will overwrite any existing song in `outputMIDI`. 
- `a2m.py` uses `abcmidi`, which only works on Linux systems, on windows it will crash. 
- If on windows then you can copy the abc file into [this website](https://www.mandolintab.net/abcconverter.php) to convert it.

## Deployment ##

This code has been deployed to a website which can be found [here](https://sentimuse.azurewebsites.net/)

We used Microsoft Azure and the python package `flask` to deploy the webapp. The files used are not in the repository however. They were made by converting `main.py` into something flask could use to run the website. 

For front end we only used `CSS` and `html` with no `javascript`. 

For source control we used `Kudu` and `GitHub` to maintain and update the website. 

## Licensing ##

[GPLv3 Licensing Information](https://github.com/grumpypenny/sentiment-analysis-and-music-generation/blob/master/LICENSE)

## References ##

- Much of our code is based on these [course notes](https://www.cs.toronto.edu/~lczhang/360/) by Lisa Zhang, 2019
    - full URL: https://www.cs.toronto.edu/~lczhang/360/
- The [abcMIDI package](http://abc.sourceforge.net/abcMIDI/original/) is used convert the ABC file into a MIDI file, written by James Allwright, supported by Seymour Shlien, 2006
    - full URL: http://abc.sourceforge.net/abcMIDI/original/
- The ABC files we used for training can be found [Here](https://abcnotation.com/tunes)
    - full URL: https://abcnotation.com/tunes
- We learned about musical modes from [this youtube video](https://www.youtube.com/watch?v=jNY_ZCUBmcA) by David Bennett Piano, 2019
    - full URL: https://www.youtube.com/watch?v=jNY_ZCUBmcA
