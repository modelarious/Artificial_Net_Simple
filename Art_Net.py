import tensorflow as tf
import numpy as np

from neuralFuncs import add_art_layers
from inputFuncs import get_Samples, MNIST_Data_Get, generate_Random_Samples
from printFuncs import network_Architecture
from errorCheck import error_Check

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

def set_Display_Step(step):
    rv = min(100, int(step/20))
    if rv == 0:
        rv += 1
    return rv


def optimize(IN, LABEL_IN, OUTPUT, LOGITS, keep_rate, trainX, trainY, testX, 
             testY, batchSize=100, keepPercent=0.5, batching=True, minCost=1e-6,
             training_epochs=20000, alpha=0.02, silent=False):
    
    displayStep = set_Display_Step(training_epochs)
    
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
                     feed_dict = {IN:sampleX, 
                                  LABEL_IN:sampleY, 
                                  keep_rate:keepPercent})
            
            cc = sess.run(cost, 
                          feed_dict = {IN:sampleX, 
                                       LABEL_IN:sampleY, 
                                       keep_rate:1.0})            
            
        else:
            # train on entire train data
            sess.run(optimizer,
                     feed_dict = {IN:trainX, 
                                  LABEL_IN:trainY, 
                                  keep_rate:keepPercent})
            cc = sess.run(cost,
                          feed_dict = {IN: trainX,
                                       LABEL_IN:trainY,
                                       keep_rate:1.0})            
        # log training
        if i % displayStep == 0:
            
            if silent == False:
                print("Training step:", '%04d' % (i), \
                      "cost=", "{:.9f}".format(cc))
            
        #if we've got a lower cost than desired
        if cc <= minCost:
            print("Cost is lower than minCost")
            break
            
    if silent == False:         
        print("Optimization Finished!")
        training_cost = sess.run(cost, 
                                 feed_dict = {IN: trainX, 
                                              LABEL_IN:trainY, 
                                              keep_rate:1.0})
        print("Training cost = ", training_cost)
    
    #gather the answers to the testing data from the network
    test_result = one_Hot(sess.run(OUTPUT, 
                                   feed_dict = {IN: testX, 
                                                keep_rate:1.0}))
    #correct the test submitted by the network
    percent_correct = check_Tests(testY, test_result)
    print("Got", str(percent_correct) + "% on the test cases.")
    return IN, LABEL_IN, OUTPUT, keep_rate    

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
            nSamples = int(0.9*inputX.shape[0])
        
            if (int(0.9*total) + int(0.1*total)) < total:
                # if the sum of 70% of the data and 30% of the data is 1 less 
                # than the actual total (floating point error), adjust by 1
                nSamples += 1

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
    
    #print the shape of the network
    if silent == False:
        network_Architecture(initVector)
        
    return optimize(IN, LABEL_IN, OUTPUT, LOGITS, keep_rate, trainX, trainY, 
                    testX, testY, batchSize, keepPercent, batching, minCost,
                    training_epochs, alpha, silent)
    


def main():
    '''
    Mushrooms contains many features, all with various levels 
    denoted by characters. Characters are mapped by get_Samples to integers
    '''
    dataFileName = 'Mushrooms.csv'
    inputX, inputY = get_Samples(dataFileName, categorical=True)
    #IN, LABEL_IN, OUTPUT, keep_rate = artificial_Net(inputX, inputY, 
    #                                                 initVector=[0, 33, 33, 0], 
    #                                                 training_epochs=200, 
    #                                                 silent=True, batchSize=100)

    IN, LABEL_IN, OUTPUT, keep_rate = artificial_Net(inputX, inputY, 
                                                     initVector=[0, 33, 33, 0], 
                                                     training_epochs=20000, 
                                                     silent=True, alpha=0.0005,
                                                     batchSize=100)  
    
    '''
    Ionosphere contains 34 features, all continuous.  get_Samples will 
    normalize the continuous data before returning it
    '''
    dataFileName = 'Ionosphere.csv'
    inputX, inputY = get_Samples(dataFileName, categorical=False)
    IN, LABEL_IN, OUTPUT, keep_rate = artificial_Net(inputX, inputY, 
                                                     initVector=[0, 80, 80, 0], 
                                                     training_epochs=10000, 
                                                     silent=True, alpha=0.0005, 
                                                     batchSize=100)
    
    
    '''
    Balance data contains categories and 4 features
    
    Class, Left Weight(1 to 5), Left Weight Distance(1 to 5), 
    Right Weight(1 to 5), Right Weight Distance (1 to 5)
    '''
    dataFileName = 'Balance_data.csv'  
    inputX, inputY = get_Samples(dataFileName, categorical=True)
    
    IN, LABEL_IN, OUTPUT, keep_rate = artificial_Net(inputX, inputY, 
                                                     initVector=[0, 35, 35, 0], 
                                                     training_epochs=600, 
                                                     silent=True, batchSize=200)
    
    #mnist example
    print("\n")
    inputX, inputY = MNIST_Data_Get()
    artificial_Net(inputX, inputY, initVector = np.array([0, 784, 0]), 
                   training_epochs=20000, batchSize=100)


main()
