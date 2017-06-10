README


INTRO

	Implements a general form of a feed forward artificial neural network with dropout on 
	the output layer and normalization of the input.
	
	The network is built as a computation graph within tensorflow.
  
	Two functions required to deploy network:
		get_Samples
		artificial_Net

	
FUNCTIONALITY

	get_Samples
		
		Purpose:
			
			Takes input data and transforms it into something more friendly to the network.
			Normalizes continuous data to have a mean of 0 and standard deviation of 1,
			maps categorical variables to integers.
			
		Definition:
		
			get_Samples(data='Balance_Data.csv', categorical=True)
			
		Inputs:
			
			
		Includes a general data 
		preprocessor step which cleans and decodes input data into a form usable by the network 
		by mapping categorical variables to integers and normalizing continuous data to have a 
		mean of 0 and a standard deviation of 1.
	
		get_Samples takes the name of the datafile to process and the content type of the file 
		(categorical samples or non-categorical?).

The output of this function is inputX and inputY:
	inputX:
	
		a numpy array that has each sample as a row
	
	inputY:
		
		a numpy array, one-hot encoding of which category that sample belongs to.


artificial_Net

	Creates a fully connected artificial network with dropout on the layer before 
	predictions to reduce overfitting.

	At the very least, requires input data and input labels (inputX and inputY).
	It splits the input data into testing and training data on it's own, selecting 30% of 
	the data to be testing data.

	Allows the structure of the net to be defined and quickly redefined to focus on spending 
	time testing ideas instead of taking time to modify the code. This feature is controlled 
	by a variable called initVector which allows you to specify the shape of the net 
	as follows:

	Suppose we have data with 35 input features and 10 possible output classes.

	If supplied with:
	initVector=[0,200,400,200,0]

	artificial_Net will first reset the vector so the input and output are of correct size:

	initVector = [35, 200, 400, 200, 10]

	then create a 5 layer net with layer sizes from left to right in the vector:
	35 layer input layer
	200 node hidden layer
	400 node hidden layer
	200 node hidden layer
	10 node output layer


	A secondary feature that I was interested in trying out was using a non linear function 
	of the biases and weights to see what effect that would have on the training.  The second
	function I implemented was two sets of weights: U, W, a set of biases B, input X:

	f(X) = U*X^2 + W*X + B

	I found that it did very little for most of my chosen datasets and, of course, 
	increased training time due to the extra set of parameters that needed to train.
	Since there is already a non-linearity being applied by the activation function there is
	really no need for a more complicated function at each node.

	This secondary objective function can be used by supplying

	objFunc = "QUAD"

	to artificial_Net

	note:
		Alpha (the learning rate), batch size, dropout rate and number of training epochs
		can be adjusted as well.


	Supply nSamples=someInt to adjust how many samples from the training data are used.  
	(artificially make the problem more challenging by removing some of the data)

	Supply minCost=someFloat to tell the optimizer to stop once it finds the training cost to
	be less than minCost (try to avoid overfitting)

	Supply alpha=someFloat to modify the learning rate

	Supply training_epochs=someInt to modify the number of training epochs

	Supply keepPercent=someInt to set (1-(percent_of_neurons_randomly_ignored)), in other
	words, to set the complement of the number of neurons that are randomly ignored.

	Supply silent=True to stop training print outs

	Supply overlap=True to overlap the training and testing data.  Not generally a good idea,
	but was interesting to play with.

	objFunc can either be an empty string or be "QUAD" to get the quadratic objective 
	function at each neuron

	batchSize controls the size of the batch fed to the network's training algorithm at each
	epoch.

	Supply batching=False to turn off batches and just use the entire data set to train on.
	Not recommended as the computed gradient vector is an average of all the gradients caused
	by the training data, therefore it will ignore small, rare and/or possibly very significant 
	features in place of general features that work across most data.  





DEMO

	I tried four different datasets, three were retrieved from http://archive.ics.uci.edu/ml/
	Specifically:
		Balance Data
		Mushroom Data
		Ionosphere Data

	as well as the MNIST handwritten digit dataset

	In the main function of the program are four demos



FURTHER STUDY

	I want to expand this code to feature hyperparameter optimization as well as add 
	tensorboard support and model saver support that I added to the convolutional network 
	code.

	I still have a lot of models to try to code, I'm rather interested in 
	generative adversarial models, want to try and use one to "dream" music, trained off a
	few of my favorite artists.
