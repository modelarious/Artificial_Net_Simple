import tensorflow as tf
import numpy as np

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

