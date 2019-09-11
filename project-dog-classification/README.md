[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

As part of this project, I built a Convolutional Neural Net (CNN) model that estimates the breed of a given dog image. If supplied an image of a human, the model will identify the resembling dog breed.  

![Sample Output][image1]

As part of the this project, I was able to achieve following tasks:

1. Write a function that uses OpenCV to detect humans in a image and use to detect Humans vs Dogs in the samples images of the project.

2. Write a Convolutional Neural Net (CNN) Model from scratch to achieve a accuracy of 12%. in 40 Epochs

3. Load large sets of Human and Dog datasets, using ImageFolder dataloader.

4. I also visualized the outputs from each convolutional layer in the scratch model.  

5. Using Transfer learning, I was able to create a model that was able to achieve a accuracy of 80% in just 10 epochs


## CNN Model 
As part of Transfer learning model, I used the ResNet-152 model. To use the transfer learning, following steps were done. 
1. Load the pretrained resnet-152 model. 
2. Freeze all the convolutions layers. 
3. Replace the last fully connected layer with a custom Linear layer. This Layer has a output features of 133 to match the expected Dog breeds classification. 
4. I experiments with various learning rates (0.01, 0.03, 0.003) and momentum settings for the SGD optimizer. 


## Project Files

1. The repository contains dog_app.ipynb file, which is the Jupyter Notebook file for the project. It contains all the code and output from the project. 

2. The dataset used for project can be downloaded at the links below:
	1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.

	2. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

5. Make sure you have already installed the necessary Python packages according to the README in the program repository.

6. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```
