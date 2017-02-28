from Network import Network 
import sys
import csv
import numpy as np
import math

max_iterations=5000
ita=1

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
    return nn,neural_struct

def buildNN(nn,neural_struct,ds,allowed_err):
    print("Training started...")
    global max_iterations
    global ita
    iterations=0
    error=100
    accuracy=0.0
    # while iterations <= max_iterations or error >= allowed_err:
    while (iterations < max_iterations and accuracy <= 100-allowed_err):
        iterations+=1
        correct=0.0
        wrong=0.0
        # print("Iteration: %d"%iterations)
        for x in range(len(ds)):   
            nodes=[]
            delta=[]
        # for x in range(1):
        # FORWARD PROPAGATION
            for l in range(len(neural_struct)): 
                n_list=[]
                delta_list=[]
                n_list.append(1.0)
                delta_list.append(0.0)
                if(l == 0):    
                    for i in range(neural_struct[l]):
                        n_list.append(float(ds[x][0][i]))
                        delta_list.append(0.0)
                    nodes.append(n_list)
                    delta.append(delta_list)    
                else:
                    #initialize
                    for i in range(neural_struct[l]):
                        n_list.append(0)
                        delta_list.append(0)
                    nodes.append(n_list)    
                    delta.append(delta_list)                                    
                    for i in range(1,neural_struct[l]+1):
                        net=0
                        for a in range(neural_struct[l-1]+1):
                            net+=float(nodes[l-1][a]) * float(nn.weights[l-1][a][i-1])
                        nodes[l][i]=sigmoid(net)
            
            # for l in range(len(neural_struct)):       
            #     print("Neurons at level %d:-"%l)                 
            #     print(nodes[l])

                 
            # BACKWARD PROPAGATION
            for l in range(len(neural_struct)-1,0,-1): 
                for i in range(1,neural_struct[l]+1):
                    if(l == len(neural_struct)-1):
                        delta[l][i]=(nodes[l][i])*(1-nodes[l][i])*(float(ds[x][1])-nodes[l][i])
                    else:
                        sum=0.0
                        for j in range(1,neural_struct[l+1]+1):
                            sum+=nn.weights[l][i][j-1]*delta[l+1][j]
                        delta[l][i]=(nodes[l][i])*(1-nodes[l][i])*sum

                    for a in range(neural_struct[l-1]+1):
                        change_delta=float(ita)*delta[l][i]*float(nodes[l-1][a])
                        # print(change_delta)
                        nn.weights[l-1][a][i-1]+=change_delta
            
            l=len(neural_struct)-1
            # print(delta[l])
            if round(nodes[l][1],1) == round(float(ds[x][1]),1):
                correct+=1        
            else:
                wrong+=1
                # print(nodes[l][1],1),round(float(ds[x][1]),1)
        
        accuracy=float(correct/(correct+wrong))*100
        print (iterations,accuracy)
            # if iterations == max_iterations:
                # print("Instance %d: "%x)
                # print(round(nodes[len(neural_struct)-1][1],1))
                # print(nn.weights)
    print("Training complete.")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def testAccuracy(nn,neural_struct,ds):
    correct=0.0
    wrong=0.0
    for x in range(len(ds)):   
            nodes=[]
            for l in range(len(neural_struct)): 
                n_list=[]
                n_list.append(1.0)
                if(l == 0):    
                    for i in range(neural_struct[l]):
                        n_list.append(float(ds[x][0][i]))
                    nodes.append(n_list)
                else:
                    #initialize
                    for i in range(neural_struct[l]):
                        n_list.append(0)
                    nodes.append(n_list)    
                    for i in range(1,neural_struct[l]+1):
                        net=0
                        for a in range(neural_struct[l-1]+1):
                            net+=float(nodes[l-1][a]) * float(nn.weights[l-1][a][i-1])
                        nodes[l][i]=sigmoid(net)
                if l == (len(neural_struct)-1):
                    # print(round(nodes[l][1],1),round(float(ds[x][1]),1))

                    if round(nodes[l][1],0) == round(float(ds[x][1]),0):
                        correct+=1        
                    else:
                        wrong+=1    

    print("Accuracy: ",str(float(correct/(correct+wrong)*100)))

def main(args):
    print("Program executing...")
    # in_data=args[1]
    # train_per=int(args[2])
    # err_tol=int(args[3])
    # num_hidden=int(args[4])
    # num_neurons=int(args[5])
    # in_data="Boston2.csv"
    in_data="training_set.csv"
    train_per=95
    err_tol=5
    num_hidden=1
    num_neurons=10
    train,test=readData(in_data,train_per)
    nn,n_struct=startBuildNN(num_hidden,num_neurons,train,err_tol)
    testAccuracy(nn,n_struct,test)
main(sys.argv)