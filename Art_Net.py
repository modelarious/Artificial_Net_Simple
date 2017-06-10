import tensorflow as tf
import pandas as pd
import numpy as np
import random
import sys

#purpose of this project:
#design a simple reusable api for me to be able to deploy vanilla versions of
#variety of different types of networks
# I suppose once I'm done here, it's going to be rather similar to 
# the way Keras or scikitlearn have their api set up


#to add:
# CNN(vanilla and RESnet (helps against vanishing gradient))

# RNN(vanilla, LSTM and GRU), 

# DBN (need an RBM model, then stack that), (helps against the vanishing
# gradient problem)), pretrain the RBM layers to reconstruct input as best
# as possible by extracting features, each layer learns entire input

# autoencoder
#
#hyperparameter optimization
#ex:
#http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# seems like the main method is to just brute force a few values for each 
# variable and track the training... not viable for networks that take long
# to train.
# Wikipedia mentions gradient-based optimization referring to this paper
#http://home.thep.lu.se/~mattias/publications/papers/Design_IEEE96.pdf
# Bayesian optimization could also be done through Bayespot
# could also try MOE as it runs in CUDA 
def get_Samples(data='Balance_Data.csv', categorical=True):
    '''
    read in a csv file
    must have a column named 'class'
    must have the same number of features for each training example
    any holes in data must have value NaN
    
    if categorical is False, every column apart from the class 
    column is assumed to contain continuous data and is normalized
    
    if categorical is True, it is assumed that the unique elements of each 
    column are unique states that that column can take on, and maps an 
    integer to each state for that column
    '''
    
    try:
        dataframe = pd.read_csv(data)
    except pd.io.common.EmptyDataError:
        sys.exit("Must not provide get_Samples with a blank file")
    except OSError:
        sys.exit("File doesn't exist")
    
    # drop any column that contains no data
    dataframe = dataframe.dropna(axis=1, how='all') 
    
    # drop any row that has nan in it
    dataframe = dataframe.dropna()  
    
    # drop any column that consists entirely of the same value
    nunique = dataframe.apply(pd.Series.nunique)
    dropColumns = nunique[nunique == 1].index
    dataframe = dataframe.drop(dropColumns, axis=1)    
    
    # if we have managed to delete all the data with the last few steps
    # or an empty data file was specified
    if len(dataframe) <= 0:
        sys.exit("No data except column names!")
    
    # get column names          
    columns = dataframe.columns.values.tolist()
    
    # if "Class" isn't a column title throw an error
    if "Class" not in columns:
        sys.exit("Class not a column in the data")
    
    # no longer need this in our list of columns, as we'll use it to specify
    # all the columns that are inputs to our neural net
    columns.remove("Class")
    
    # If this leaves us with no more columns, throw an error
    if len(columns) == 0:
        sys.exit("Need more than just class column")
    
    # create labels for our data
    inputY = dataframe.loc[:, ['Class']].as_matrix()
    
    # no need to iterate over it anymore
    dataframe.drop(['Class'], axis=1)
    
    #get the number of output classes
    unique_classes = np.unique(inputY)
    num_unique_classes = len(unique_classes)
    num_samples = len(inputY)
    
    
    # these next few lines take the class that is requested by each label,
    # and marks that point in the row as 1 and the rest as 0
    # for example:
    # we are told that sample 4 has label 3 out of 4 possible labels, 
    # so we create the label [0,0,1,0]
    # and store it in the array at the proper place
    dict_map = {}
    for i in range(num_unique_classes):
        dict_map[unique_classes[i]] = i
    
    #num_samples long, num_unique_classes wide 
    Y = np.zeros([num_samples, num_unique_classes])  
    
    
    for i in range(num_samples):
        curr_class = inputY[i]
        curr_class_encode = dict_map[curr_class[0]]
        Y[i][curr_class_encode] = 1.0
    
    # maps unique column states to integers
    # if a column can take on values 'a' and 'b', then 'a' is 0, 'b' is 1
    # then replace all 'a's in the column with 0 and 'b's with 1
    if categorical == True:
        for col in dataframe.select_dtypes(exclude=['int', 'float']).columns:
            dataframe[col] = pd.Categorical(
                dataframe[col], categories=dataframe[col].unique()).codes    
    
    
    # create our input data
    X = dataframe.loc[:, columns].as_matrix()
    print("\nNumber of samples:", num_samples)
    print("Number of categories:", num_unique_classes)
    print("Number of features per sample:", len(columns))
    
    
    
    #normalize input data
    means = np.mean(X, axis=0)
    stddevs = np.std(X, axis=0)
    X[:,0] = (X[:,0] - means[0]) / stddevs[0]
    X[:,1] = (X[:,1] - means[1]) / stddevs[1]        
    
    
    return X, Y


def BATCH(to_batch_x, to_batch_y, batchSize=0):
    
    length = to_batch_x.shape[0]
    # if batchSize is unspecified, or batchSize is greater than the number
    # of samples available
    if batchSize <= 0 or batchSize > length:
        batchSize = int(length/10)
        if batchSize == 0:
            batchSize += 1
    batch_indices = np.random.choice(length, batchSize, replace=False)
    return to_batch_x[batch_indices], to_batch_y[batch_indices] 


def MNIST_Data_Get():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    inputX = np.concatenate((mnist.test.images, mnist.train.images, mnist.validation.images), axis=0)
    inputY = np.concatenate((mnist.test.labels, mnist.train.labels, mnist.validation.labels), axis=0)
    
    return inputX, inputY
    
    
def generate_Random_Samples(num, IN_X, IN_Y, overlap=False):
    total_samples = IN_X.shape[0]
    testingNum = int(total_samples*0.3)
    
    #fix divide by 0 errors
    if testingNum == 0:
        testingNum += 1
    
    # if overlap is true, there may be some overlap between 
    # training and testing data
    # not typically a good idea, but fun to play with
    if overlap == False:
        
        if num + testingNum > total_samples:

            num = total_samples - testingNum
            
            string = "Too many samples requested for generate_Random_Samples. "\
            "\nMaximum selectable is " + str(total_samples) + " with overlap=True or "\
             + str(num) + " with it off."
            
            print(string)
            
        ind = np.random.choice(
            range(total_samples), num + testingNum, replace=False)
        training_indices = ind[:num]
        testing_indices = ind[num:]
        
    else:
        if num > total_samples:
            
            num = total_samples
            
            string = "Too many samples requested for generate_Random_Samples. "\
            "\nMaximum selectable is " + str(total_samples) + " with overlap=True or "\
              + str(num) + " with it off."
            
            print(string)
        
        if total_samples == num:
            training_indices = np.arange(total_samples)
        else:
            training_indices = np.random.choice(
                range(total_samples), num, replace=False)
        
        testing_indices = np.random.choice(
            range(total_samples), testingNum, replace=False)
    
    trainX = IN_X[training_indices]
    trainY = IN_Y[training_indices]
    testX = IN_X[testing_indices]
    testY = IN_Y[testing_indices]
    
    print("Number of training samples", num)
    print("Number of testing samples", testingNum)
    
    return trainX, trainY, testX, testY


def formatted_print(i, fcShape):
    if i % 10 == 1 and i % 100 not in range(10,20):
        print("flow through", str(i) + "st layer", fcShape[i], \
              "to", fcShape[i+1])
    elif i % 10 == 2 and i % 100 not in range(10,20):
        print("flow through", str(i) + "nd layer", fcShape[i], \
              "to", fcShape[i+1])
    elif i % 10 == 3 and i % 100 not in range(10,20):
        print("flow through", str(i) + "rd layer", fcShape[i], \
              "to", fcShape[i+1])
    else: 
        print("flow through", str(i) + "th layer", fcShape[i], \
              "to", fcShape[i+1])   
        
        
def instantiate_biases(fcShape, silent):
    biases = {}
    for i in range(fcShape.size - 1):
        biases[str(i)] = tf.Variable(tf.zeros([fcShape[i+1]]))
        if silent == False:
            formatted_print(i, fcShape)
        
    return biases
    
    
def instantiate_weights(fcShape):
    weights = {}
    for i in range(fcShape.size - 1):
        weights[str(i)] = tf.Variable(tf.random_normal(
            [fcShape[i], fcShape[i+1]], 
            stddev=1.0/tf.sqrt(2.0)))   
    return weights
    
    
def one_Hot(IN):
    '''
    turns prediction probabilities into labels
    
    takes an array of n samples by m features
    uses a zeroed out array and makes the most prominent column in each
    row (most prominent feature in each sample) a 1.0
    
    If input is 
    [[0.1, 0.9],
     [0.4, 0.7]]
     
    resulting output would be 
    [[0.0, 1.0],
     [0.0, 1.0]]
     
    input:
    [[0.1, 0.5, 0.4],
     [0.6, 0.3, 0.1]]
      
    output:
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0]]
     
    '''
    # initialize array
    OUT = np.zeros(IN.shape)
    
    # gather argmax of each row
    argmaxes = np.argmax(IN, axis=1)
    
    # set the argmax position in the OUT array as 1.0
    OUT[np.arange(argmaxes.size), argmaxes] = 1.0
    
    return OUT

def Quad_obj(fcShape, output_layer, weights, biases):
    
    # additional variables for the quadratic function
    second_degree_weights = instantiate_weights(fcShape) 
    
    # Version with nx^2 + mx + b
    for i in range(fcShape.size - 1):
       
        if i == (fcShape.size - 2):
            #last layer
            
            #configurable dropout
            keep_prob = tf.placeholder(tf.float32)
            output_layer = tf.nn.dropout(output_layer, keep_prob)
            
            first_term = tf.matmul(
                tf.square(output_layer),
                second_degree_weights[str(i)])
            
            second_term = tf.matmul(output_layer, weights[str(i)])
            
            #need to store logits at output layer
            LOGITS = tf.add(tf.add(first_term, second_term), 
                            biases[str(i)]) 
            OUTPUT = tf.nn.softmax(LOGITS) 
            
        else:
            #flow through i^th layer
            first_term = tf.matmul(
                tf.square(output_layer),
                second_degree_weights[str(i)])
            
            second_term = tf.matmul(output_layer, weights[str(i)])

            output_layer = tf.add(tf.add(first_term, second_term), 
                                  biases[str(i)]) 
            output_layer = tf.nn.softmax(output_layer) 
    
    return OUTPUT, LOGITS, keep_prob

def linear_obj(fcShape, output_layer, weights, biases):
    for i in range(fcShape.size - 1):
        
        if i == (fcShape.size - 2):
            #last layer
            keep_prob = tf.placeholder(tf.float32)
            output_layer = tf.nn.dropout(output_layer, keep_prob)
            
            #need to store logits at output layer
            LOGITS = tf.add(tf.matmul(output_layer, weights[str(i)]), 
                            biases[str(i)])
            OUTPUT = tf.nn.softmax(LOGITS) 
         
        else:
            #flow through i^th layer
            output_layer = tf.add(tf.matmul(output_layer, weights[str(i)]), 
                                  biases[str(i)])
            output_layer = tf.nn.softmax(output_layer) 
    return OUTPUT, LOGITS, keep_prob   

def add_art_layers(IN, fcShape, objFunc, silent):    

    weights = instantiate_weights(fcShape)
    biases = instantiate_biases(fcShape, silent)
    
    #flow through the layers
    if objFunc == "QUAD":
        OUTPUT, LOGITS, keepProb = Quad_obj(fcShape, IN, weights, biases)
           
    else:
        #Version with just mx + b
        OUTPUT, LOGITS, keepProb = linear_obj(fcShape, IN, weights, biases)
        
    return OUTPUT, LOGITS, keepProb   

def create_Simple_Artificial_Network(layerSizes, silent=False, objFunc=''):
    #we get as input:
    #[2, 20, 20, 30, 2]
    #means 2 input layers, 2 output layers and 3 hidden layers, sizes 20, 20, 30, 
    #if it's shorter than size 2, then we don't have a proper initialization vector    
    #softmax activation
    
    # get the input and output layer sizes
    input_layer_size = layerSizes[0]
    output_layer_size = layerSizes[-1]
    
    #input and output placeholder, feed data to IN, feed labels to LABEL_IN
    IN = tf.placeholder(tf.float32, [None, input_layer_size])
    LABEL_IN = tf.placeholder(tf.float32, [None, output_layer_size]) 
    
    
    OUTPUT, LOGITS, keepProb = add_art_layers(IN, layerSizes, objFunc, silent)
    return IN, LABEL_IN, OUTPUT, LOGITS, keepProb
    

def check_Tests(A, B):
    #could be achieved with argmaxes as well
    if list in [type(A), type(B)]:
        return "Instead of numpy array, received list"
    
    if A.shape != B.shape:
        return "Shapes of two inputs didn't match"
    
    percent_correct = np.mean(np.equal(A, B)) 
    return percent_correct * 100


def network_Architecture(vector):
    length = len(vector)
    if length == 2:
        print("Input layer with", vector[0], "inputs.")
        print("Output layer with", vector[1], "outputs.")
    else:
        print("Input layer with", vector[0], "inputs.")
        for i in range(1, length-1):
            print("Hidden layer with", vector[i], "nodes")
        print("Output layer with", vector[-1], "outputs.")        


def error_Check(inputX, inputY, nSamples, initVector, minCost, alpha, 
                training_epochs, strictness, silent, overlap, objFunc,
                keepPercent, batchSize, batching):
    
    acceptedObjFuncs = ["", "QUAD"]
    
    if inputX.shape[0] != inputY.shape[0]:
        print("Must have the same number of labels and training samples")
        return -1
    
    if type(nSamples) != int:
        print("nSamples must be an integer")
        return -1
    
    if len(initVector) <= 1:
        print("init vector is too short, must be at least 2 layers")
        return -1
    
    if type(minCost) not in [int, float]:
        print("minCost must be float or integer")
        return -1
    
    if type(alpha) not in [int, float]:
        print("alpha must be float or integer")
        return -1
    
    if type(training_epochs) != int:
        print("training_epochs must be an integer")
        return -1
    
    if type(strictness) not in [int, float]:
        print("strictness must be float or integer")
        return -1
    
    if type(silent) != bool:
        print("silent must be either True or False")
        return -1
    
    if type(overlap) != bool:
        print("silent must be either True or False")
        return -1
    
    if strictness > 100:
        print("Not possible to receive greater than 100% on tests")
        return -1
    
    if type(keepPercent) not in [float, int]:
        print("keepPercent must be an int or float")
        return -1
        
    if keepPercent > 1 or keepPercent < 0:
        print("keepPercent must be between 0 and 1")
        return -1
        
    if objFunc not in acceptedObjFuncs:
        print("objFunc can only take on values: ", end='')
        for func in acceptedObjFuncs[:-1]:
            if func == '':
                print("Empty string", end=', ')
            else:
                print(func, end=", ")
        
        print(acceptedObjFuncs[-1] + '.')
        return -1   
    
    if type(batchSize) != int:
        print("batchSize must be int")
        return -1
    
    if batching not in [True, False]:
        print("batching must be boolean")
        return -1
    
    #passed all tests
    return 0

def set_Display_Step(step):
    rv = int(step/20)
    if rv == 0:
        rv += 1
    return rv



def artificial_Net(inputX, inputY, nSamples=0, initVector=np.array([0,0]), 
                   minCost=1e-6, alpha=0.02, training_epochs=10000, strictness=90, 
                   keepPercent=0.5, silent=False, overlap=False, objFunc='',
                   batchSize=50, batching=True):
    '''
    inputX is a 2d numpy array
    each row is a single sample's inputs
    
    inputY is a 2d numpy array
    each row is a single sample's labels
    if there are two categories, inputY should have rows that are either [0,1] or [1,0]
    
    nSamples is the number of samples to be used for training the net
    
    
    
    initVector is the shape of the Artificial Net
    Init Vector examples
    imagine 22 features and 2 categories:

    initVector = np.array([22, 15, 7 , 16, 2])
    Input layer with 22 inputs.
    Hidden layer with 15 nodes
    Hidden layer with 7 nodes
    Hidden layer with 16 nodes
    Output layer with 2 outputs.

    the input layer and output layer will correct themselves to fit the data they
    are working with:
    initVector = np.array([99,24,3400])
    Input layer with 22 inputs.
    Hidden layer with 24 nodes
    Output layer with 2 outputs.

    So it's safe to leave them as 0 and let artificial_Net set them:
    initVector = np.array([0, 15, 7 , 15, 0]) ->
    [22, 15, 7, 15, 2]
    

    minCost is the cost at which the network stops training
    
    alpha (also known as learning rate or step size) is the scaling factor for 
    the amount that we shift our estimates each step
    
    minCost is the acceptable minimum cost, when training, if the cost reaches
    this value, it will assume it's done, can be set to 0 to remove feature
    
    training_epochs is the maximum episodes of training we are going 
    to perform, it could be less due to the minCost variable.
    
    strictness is the test score the network must reach to pass
    
    keepPercent is the keep rate for dropout on the last layer
    
    silent=True prints out significantly less information
    
    overlap dictates if there will be overlap between training and testing data
    
    objFunc can be set to "QUAD" to use the 
    following quadratic function on each neuron
    softmax( U*(X**2) + W*X + B )
    
    batchSize is the size of each batch fed for an epoch
    '''
    #ensure correct data container
    inputX = np.array(inputX)
    inputY = np.array(inputY)
    initVector = np.array(initVector)
    
    rc = error_Check(inputX, inputY, nSamples, initVector, minCost, alpha, 
                     training_epochs, strictness, silent, overlap, objFunc,
                     keepPercent, batchSize, batching)
    if rc == -1:
        #error occured
        return -1
    
    total = inputX.shape[0]
    if nSamples == 0:
        
        if overlap == True:
            nSamples = total
            
        else:
            nSamples = int(0.7*inputX.shape[0])
        
            if (int(0.7*total) + int(0.3*total)) < total:
                # if the sum of 70% of the data and 30% of the data is 1 less 
                # than the actual total (floating point error), adjust by 1
                nSamples += 1
        
    display_step = set_Display_Step(training_epochs)

    trainX, trainY, testX, testY = generate_Random_Samples(
        nSamples, inputX, inputY, overlap)
    
    if silent == True:
        print("Executing...")    
    
    # if the batch size is equal to the total number 
    # of samples, turn batching off
    if batching==True and batchSize==trainX.shape[0]:
        batching = False
        

    n_features = inputX.shape[1]
    n_categories = inputY.shape[1]
    
    # set the size of the input and output layer of the network
    initVector[0] = n_features
    initVector[-1] = n_categories
    
    #build the network
    IN, LABEL_IN, OUTPUT, LOGITS, keep_rate = create_Simple_Artificial_Network(
        initVector, silent, objFunc)
    
    #calculate cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=LOGITS, labels=LABEL_IN))
    
    #minimize cost
    optimizer = tf.train.AdamOptimizer(alpha).minimize(cost)
    
    #initialize all variables and run the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #optimize
    for i in range(training_epochs):
        
        # if batching is being used
        if batching==True:
            
            # collect a random sample from the training data
            sampleX, sampleY = BATCH(trainX, trainY, batchSize)
            
            # optimize
            sess.run(optimizer, 
                     feed_dict = {IN: sampleX, 
                                  LABEL_IN:sampleY, 
                                  keep_rate:keepPercent})
            
        else:
            # train on entire train data
            sess.run(optimizer,
                     feed_dict = {IN: trainX, 
                                  LABEL_IN:trainY, 
                                  keep_rate:keepPercent})
        # log training
        if i % display_step == 0:
            
            # check what it thinks when you give it the input data
            if batching == True:
                cc = sess.run(cost, 
                              feed_dict = {IN: sampleX, 
                                           LABEL_IN:sampleY, 
                                           keep_rate:1.0})
            else:
                cc = sess.run(cost,
                              feed_dict = {IN: trainX,
                                           LABEL_IN:trainY,
                                           keep_rate:1.0})
            
            
            if silent == False:
                print("Training step:", '%04d' % (i), \
                      "cost=", "{:.9f}".format(cc))
            
            #if we've got a lower cost than desired
            if cc <= minCost:
                print("Cose is lower than minCost")
                break
            
    if silent == False:         
        print("Optimization Finished!")
        training_cost = sess.run(cost, 
                                 feed_dict = {IN: trainX, 
                                              LABEL_IN:trainY, 
                                              keep_rate:1.0})
        print("Training cost = ", training_cost)
    
    #print the shape of the network
    if silent == False:
        network_Architecture(initVector)
    
    #gather the answers to the testing data from the network
    test_result = one_Hot(sess.run(OUTPUT, 
                                   feed_dict = {IN: testX, 
                                                keep_rate:1.0}))
    #correct the test submitted by the network
    percent_correct = check_Tests(testY, test_result)
    print("Got", str(percent_correct) + "% on the test cases.")
    
    # A or better?
    if percent_correct >= strictness:
        return "SUCCESS"
    
    return "Failure"


def main():
    
    '''
    Balance data contains categories and 4 features
    
    Class, Left Weight(1 to 5), Left Weight Distance(1 to 5), 
    Right Weight(1 to 5), Right Weight Distance (1 to 5)
    '''
    dataFileName = 'Balance_data.csv'  
    inputX, inputY = get_Samples(dataFileName, categorical=True)
    
    artificial_Net(inputX, inputY, initVector=[0, 35, 35, 0], 
                   training_epochs=3000, silent=True, batchSize=20)
    
    '''
    Mushrooms contains many features, all with various levels 
    denoted by characters. Characters are mapped by get_Samples to integers
    '''
    dataFileName = 'Mushrooms.csv'
    inputX, inputY = get_Samples(dataFileName, categorical=True)
    artificial_Net(inputX, inputY, initVector=[0, 33, 33, 0], 
                   training_epochs=200, silent=True, batchSize=100)
    
    '''
    Ionosphere contains 34 features, all continuous.  get_Samples will 
    normalize the continuous data before returning it
    '''
    dataFileName = 'Ionosphere.csv'
    inputX, inputY = get_Samples(dataFileName, categorical=False)
    artificial_Net(inputX, inputY, initVector = [0, 40, 40, 0], 
                   training_epochs=2500, silent=True,
                   alpha=0.0005)
    
    
    #mnist example
    
    print("\n")
    #typically scores around 98.5%
    inputX, inputY = MNIST_Data_Get()
    artificial_Net(inputX, inputY, initVector = np.array([0, 784, 0]), 
                   training_epochs=20000, batchSize=100)
    



main()
