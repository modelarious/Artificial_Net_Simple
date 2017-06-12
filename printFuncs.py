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
