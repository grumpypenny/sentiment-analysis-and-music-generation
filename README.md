# Sentiment Analysis with Music Generation
## About ##
The goal of this project is to create unique music based on text given by a user. Given a sad story it will generate sad music. Likewise given a happy tale we get joyful music.

This project was created by two University of Toronto Students, Ajitesh Misra and Sharven Prasad Dhanasekar 

## Installation and Usage ##
#### Repository Structure ####
Come back here after the repo is cleaned.

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

#### Music Generation Model ####

To generate music we used [ABCnotation.](https://abcnotation.com/learn). This format gives a versitile way to storing music as raw text. 

The AI generates music in this format, which can then be changed into an audio format like midi. 

__model architecture:__

The model uses one hots to encode different characters into vectors which are much easier to work with. 

Our model has two layers, a GRU and a fully connected. The GRU we used is a single layer. 
The GRU is used to make sure the character we generate on this pass depends on the last character. That way as we generate character a pattern forms. 

__Here is how we make music:__

The AI is trained on a two datasets, one for happy songs and one for sad songs. We seperated songs into happy and sad based on the key. 
For example a song in A major would be happy while a song in A minor would be sad. Both models are the same, the only difference is the data its trained on. 

Once the sentiment of the text is found, a sample is sequenced from the correct model. We do not want the same characters from the model each time, so what we do is generate a multinomial distribution of all the possible characters we could. The way we generate text also takes in a integer called temperature. The higher the temperature, the more uniform the character distribution is. What this does is it ensures we do not always get the same characters when we run the model. The higher the temperature makes the text more random. We sample one character at a time, only stopping once we reach the character limit. 

## Licensing ##

## Bibliography ##
Put all papers and libraries used here
