import pandas as pd
import numpy as np
import sys

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

def MNIST_Data_Get():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    inputX = np.concatenate((mnist.test.images, mnist.train.images, mnist.validation.images), axis=0)
    inputY = np.concatenate((mnist.test.labels, mnist.train.labels, mnist.validation.labels), axis=0)
    
    return inputX, inputY
    
    
def generate_Random_Samples(num, IN_X, IN_Y, overlap=False):
    total_samples = IN_X.shape[0]
    testingNum = int(total_samples*0.1)
    
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
