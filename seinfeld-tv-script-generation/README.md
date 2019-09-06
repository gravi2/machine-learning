## Project Overview

As part of this project, I built a Recurrent Neural Net (RNN) model that uses LSTM(Long Short Term Memory) & Embedding, to generate a new TV script for the famous Seinfeld TV show. The model accept as input, a seed/prime word as an input and then generates the script of a given length (e.g 400 words). 


[Sample output](generated_script_1.txt)

As part of the this project, I was able to achieve following tasks:

1. Explore and analyze the given Seinfield scripts. Understand the dataset stats, like the number of unique words, lines, average number of words in each line. This understanding helped me preprocess data to be given as input, come up with model architecture and finally tweak the hyper parameters.

2. Explore, understand and implement various aspects of LSTM cells (Batching, sequencing, hidden state)

3. Used Embeddings (embedding layer), to avoid computational overhead that results due to one-hot encoding of input.  

4. Handle the preprocessing of input data and batching using TensorDataset and DataLoader.

5. Implement a fully functional Recurrent Neural Net (RNN) Model in PyTorch, to achieve a model loss of 3.47 in just 15 Epochs.


## RNN Model 
My RNN model had following architecture and steps. 
1. Prepocess the Seinfeld data scripts and preprocess them as input words. 
2. Use an embedding layer as the first layer to pass the input and create lookup tables. This is to avoid  one-hot encoding and the related/wasted computation. 
3. 2 layers of LSTM stacked over each other, to process the sequence of input words and carry the hidden state from one hidden layer to the other
4. A fully connected linear layer that produces the scores of predicted next word.
5. I experiments with various sequence_lenght, hidden dimmensions, learning rate to achieve the desidered model loss. 

## Project Files

1. The project folder contains:
    1. dlnd_tv_script_generation.ipynb file, which is the Jupyter Notebook file for the project. It contains all the code and output from the project.
    2. dlnd_tv_script_generation.html file, which is the Jupyter Notebook executed and saved output.
    3. helper.py, problem_unittest.py, which are helper files used for this project.
    4. preprocess.p, is the saved lookup table.
    5. generated_script_1.txt, is a sample output from model.  

2. The dataset used for project can be found in [Seinfeld_Scripts.txt](data/Seinfeld_Scripts.txt).

3. Make sure you already have a working python environment, with all the usual packages (Numpy, pytorch, jupyter).

6. Open the dlnd_tv_script_generation.ipynb notebook and follow the instructions.
	