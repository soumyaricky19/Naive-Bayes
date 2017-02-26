from Network import Network 
import sys
import csv
import numpy as np
import math

max_iterations=1

def readData(data_file,per):
    file=open(data_file)
    read=csv.reader(file)
    is_header=True

    map=[]
    attribute_vector_list=[]
    attribute_num=0
    row_num=0
    for row in read:
        if is_header:
            
            attribute_num=len(row)-1
            map=row[0:attribute_num]
            is_header=False       
        else:        
            row_num+=1
            train=(row[0:len(row)-1],row[len(row)-1])
            attribute_vector_list.append(train)

    set=np.vsplit(attribute_vector_list,np.array([row_num*per/100,]))
    return set

def startBuildNN(num_hidden,num_neurons,dataset,allowed_err):
    # form neural network structure
    neural_struct=[]
    #attribute layer
    attribute_len=len(dataset[0][0])
    neural_struct.append(attribute_len)
    #hidden layers
    for x in range(int(num_hidden)):
        neural_struct.append(int(num_neurons))
    #classification layer
    classes=1
    neural_struct.append(classes)
    nn=Network(neural_struct)
    # print("Final")
    # print(nn.weights[0][1][1])
    buildNN(nn,neural_struct,dataset,allowed_err)
    return nn

def buildNN(nn,neural_struct,ds,allowed_err):
    global max_iterations
    iterations=0
    error=100
    # while iterations <= max_iterations or error >= allowed_err:
    while iterations < max_iterations:
        iterations+=1
        print("Iteration: %d"%iterations)
        for x in range(len(ds)):   
            nodes=[]
        # for x in range(1):
            for l in range(len(neural_struct)): 
                if(l == 0): 
                    nodes.append(ds[x][0])
                else:
                    n_list=[]
                    net=0                      
                    for i in range(neural_struct[l]):
                        for a in range(neural_struct[l-1]):
                            net+=float(nodes[l-1][a]) * float(nn.weights[l-1][a][i])
                        net+=float(nn.weights[l-1][0][i])
                        n_list.append(sigmoid(net))
                    nodes.append(n_list)   

            for l in range(len(neural_struct)):       
                print("Neurons at level %d:-"%l)                 
                print(nodes[l])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def main(args):
    print("Program executing...")
    # in_data=args[1]
    # train_per=int(args[2])
    # err_tol=int(args[3])
    # num_hidden=int(args[4])
    # num_neurons=int(args[5])
    in_data="training_set.csv"
    train_per=80
    err_tol=80
    num_hidden=1
    num_neurons=2
    train,test=readData(in_data,train_per)
    nn1=startBuildNN(num_hidden,num_neurons,train,err_tol)

main(sys.argv)