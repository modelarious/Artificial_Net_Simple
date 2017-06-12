import tensorflow as tf
from printFuncs import formatted_print

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

def Quad_obj(fcShape, output_layer, silent):

    
    keep_prob = tf.placeholder(tf.float32)
    weights = instantiate_weights(fcShape)
    biases = instantiate_biases(fcShape, silent)    
    # additional variable for the quadratic function
    second_degree_weights = instantiate_weights(fcShape) 
    
    
    # Version with nx^2 + mx + b
    for i in range(fcShape.size - 1):
       
        if i == (fcShape.size - 2):
            #last layer
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
            output_layer = tf.nn.dropout(output_layer, keep_prob)
    
    return OUTPUT, LOGITS, keep_prob

def linear_obj(fcShape, output_layer, silent):
    keep_prob = tf.placeholder(tf.float32)
    weights = instantiate_weights(fcShape)
    biases = instantiate_biases(fcShape, silent)    
    for i in range(fcShape.size - 1):
        
        if i == (fcShape.size - 2):
            #last layer
            #need to store logits at output layer
            LOGITS = tf.add(tf.matmul(output_layer, weights[str(i)]), 
                            biases[str(i)])
            OUTPUT = tf.nn.softmax(LOGITS) 
         
        else:
            #flow through i^th layer
            output_layer = tf.add(tf.matmul(output_layer, weights[str(i)]), 
                                  biases[str(i)])
            output_layer = tf.nn.softmax(output_layer) 
            output_layer = tf.nn.dropout(output_layer, keep_prob)
    return OUTPUT, LOGITS, keep_prob   

def add_art_layers(IN, fcShape, objFunc, silent):    
    
    #flow through the layers
    if objFunc == "QUAD":
        OUTPUT, LOGITS, keepProb = Quad_obj(fcShape, IN, silent)
           
    else:
        #Version with just mx + b
        OUTPUT, LOGITS, keepProb = linear_obj(fcShape, IN, silent)
        
    return OUTPUT, LOGITS, keepProb
