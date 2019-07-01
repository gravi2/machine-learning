# Project Overview
In this project, I have a set of bike rental data. And using Neural Network, I try to predict the daily bike rental ridership.  

# Dataset
The dataset includes the details of the year, month, day and hour data. The data also includes data on bikes rented on holidays, weekends and working days. 
The dataset is available on CSV files and is present in the [Bike-Sharing-Dataset](./Bike-Sharing-Dataset/) folder.

# Neural Network (Regression)
My NeuralNetwork is defined in [my_answers.py](./my_answers.py). The model has 3 layers ( 1 input, 1 hidden and 1 output layer). 

The input layer nodes are configurable and match the number of features in our dataset.
The hidden layer nodes are also configurable and tested with 2 nodes.
The output layer has 1 node and outputs the bike rental prediction for the given data.  

 The Neural network:
1. Defines random weights for the input and hidden layers. 
2. Defines forward pass and back propogation (Gradient descent) implementations.
3. Trains the model using the traininig data from dataset.
4. Validates the model by using part of the unseen dataset.

# Accuracy
1. My model seems to be predicting 85-86% based on the validation loss.
2. It seems, when it fails, the model is predicting the 'cnt' higher than it should.
3. May be we need more data. I say that because, increasing the iterations is not helping beyond 3000/4000. Increasing the hidden layers beyond 10 is also not helping in the accuracy.