from Network import Network 
import sys
import csv
import numpy as np
import math

max_iterations=100000
ita=0.9
precision=1

def preProcess(arr):
    string_att_map=[]
    attribute_vector_list=[]

    arr_np=np.array(arr)
    string_att_map,data_arr_str=np.vsplit(arr_np,[1])
    # numbering,data_arr_str=np.hsplit(arr_np,[1])
    data_arr=data_arr_str.astype(np.float)
    
    mean_list=np.mean(data_arr,axis=0)
    std_list=np.std(data_arr,axis=0)
    max_list=np.max(data_arr,axis=0)
    min_list=np.min(data_arr,axis=0)

    result=[]
    for j in range(len(data_arr[0])):
        for i in range(len(data_arr)):
            # data=float(data_arr[i][j]-mean_list[j])/std_list[j]
            data=float(data_arr[i][j]-min_list[j])/(max_list[j]-min_list[j])
            data_arr[i][j]=float(data)
        
#Final data conversion        
    for row in data_arr:
        split_rows=(row[1:len(row)-1],row[len(row)-1])
        attribute_vector_list.append(split_rows)
    # print(attribute_vector_list)
    return attribute_vector_list


def readData(data_file):
    file=open(data_file)
    read=csv.reader(file)
    attribute_vector_list=[]
    for row in read:
        attribute_vector_list.append(row)
    return attribute_vector_list
    

def startBuildNN(num_hidden,num_neurons,dataset,allowed_err):
    # form neural network structure
    neural_struct=[]
    #attribute layer
    attribute_len=len(dataset[0][0])
    neural_struct.append(attribute_len)
    #hidden layers
    # for x in range(int(num_hidden)):
    #     neural_struct.append(int(num_neurons))
    neural_struct.append(8)
    # neural_struct.append(5)
    #classification layer
    classes=1
    neural_struct.append(classes)
    print(neural_struct)
    nn=Network(neural_struct)
    # print("Final")
    # print(nn.weights[0][1][1])
    buildNN(nn,neural_struct,dataset,allowed_err)
    return nn,neural_struct

def buildNN(nn,neural_struct,ds,allowed_err):
    print("Training started...")
    global max_iterations
    global ita
    global precision
    iterations=0
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
                        # ita=iterations/100000.0
                        change_delta=float(ita)*delta[l][i]*float(nodes[l-1][a])
                        # print(change_delta)
                        nn.weights[l-1][a][i-1]+=change_delta
            
            l=len(neural_struct)-1
            # print(delta[l])
            if round(nodes[l][1]-float(ds[x][1]),precision) == round(float(0),precision):
                correct+=1        
            else:
                wrong+=1
                # print(nodes[l][1],1),round(float(ds[x][1]),1)
        
        accuracy=float(correct/(correct+wrong))*100
        print (iterations,"Training...accuracy: ",accuracy)
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
                     
                    if round(nodes[l][1]-float(ds[x][1]),precision) == round(float(0),precision):    
                        correct+=1  
                        print("correct",(nodes[l][1]),float(ds[x][1]))
                    else:
                        wrong+=1
                        print("Wrong",(nodes[l][1]),float(ds[x][1]))

    print("Test accuracy: ",str(float(correct/(correct+wrong)*100)))

def main(args):
    print("Program executing...")
    # in_data=args[1]
    # train_per=int(args[2])
    # err_tol=int(args[3])
    # num_hidden=int(args[4])
    # num_neurons=int(args[5])
    in_data="Boston.csv"
    # in_data="training_set.csv"
    train_per=90.0
    err_tol=35.0
    num_hidden=1
    num_neurons=15
    raw_list=readData(in_data)
    processed_list=preProcess(raw_list)
    train,test=np.vsplit(processed_list,np.array([len(raw_list)*train_per/100,]))
    nn,n_struct=startBuildNN(num_hidden,num_neurons,train,err_tol)
    testAccuracy(nn,n_struct,test)
main(sys.argv)