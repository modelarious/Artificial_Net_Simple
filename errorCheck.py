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
