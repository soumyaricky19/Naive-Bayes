from Network import Network 
import sys
import csv
import numpy as np

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

def startBuildNN(num_hidden,num_neurons,dataset):
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
    buildNN(nn,dataset)
    return nn

def buildNN(nn,ds):
    for x in range(len(ds)):
        nn.nodes[0]=ds[x][0]
        

def main(args):

    print("Program executing...")
    in_data=args[1]
    train_per=int(args[2])
    err_tol=int(args[3])
    num_hidden=int(args[4])
    num_neurons=int(args[5])
    train,test=readData(in_data,train_per)
    nn1=startBuildNN(num_hidden,num_neurons,train)

main(sys.argv)