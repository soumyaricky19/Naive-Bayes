from Network import Network 
import sys
import csv
import numpy as np
import math

max_iterations=5
class_threshold=0.05
ita=0.5

def preProcess(data_in,data_out):
    global class_threshold
    string_att_map=[]
    attribute_vector_list=[]
    col_to_delete=[]
    is_classfication=False
    num_cols=0
    num_rows=0

    #Read data
    file=open(data_in)
    read=csv.reader(file)
    arr=[]
    for row in read:
        arr.append(row)
    file.close()

    #Separate header
    is_header=True
    has_header=True
    no_header_data=[]
    string_att_map=[]
    
    for row in arr:
        if is_header:
            is_header=False
            for j in range(len(row)):
                try:
                    temp1=float(row[j])
                    has_header=False
                except ValueError:
                    pass  # it was a string
            if has_header == True:
                string_att_map=row[0:len(row)]
            
        else:
            if(len(row) >0):
                num_rows+=1
                no_header_data.append(row)
        
    #Ignore columns with no header  
    valid_string_att_index=[]
    for j in range(len(string_att_map)):
        if len(string_att_map[j]) != 0:
            valid_string_att_index.append(j)
    
    
    if has_header == False:
        for j in range(len(arr[0])):
            valid_string_att_index.append(j)
    
    num_cols=len(valid_string_att_index)

    data_arr_str=[]
    for i in range(num_rows):
        valid_arr_list=[]
        for j in valid_string_att_index:
            valid_arr_list.append(no_header_data[i][j])
        data_arr_str.append(valid_arr_list)

    #Assign proper datatype to values
    for j in range(num_cols):
        unique_list=[]
        temp_arr_str=[]
        is_str=False
        for i in range(num_rows):
            if data_arr_str[i][j]=='' or data_arr_str[i][j]==' ':
                data_arr_str[i][j]='0'
            
            #Initialize
            temp_arr_str.append(0)      
            try:
                temp = float(data_arr_str[i][j])
            except ValueError:
                pass  # it was a string
                is_str=True

            if data_arr_str[i][j] in unique_list:
                temp_arr_str[i]=unique_list.index(data_arr_str[i][j])+1
            else:
                temp_arr_str[i]=len(unique_list)+1
                unique_list.append(data_arr_str[i][j])
                 
        if (len(unique_list) < class_threshold*num_rows and (is_str or (j == num_cols-1))):
            # print("Encoded column:",j)
            # print("Distinct:",len(unique_list))
            for i in range(num_rows):
                data_arr_str[i][j]=temp_arr_str[i]

    #If number of distinct classes are less than 5% of the instances, it can be considered classification
    if (len(unique_list) < class_threshold*num_rows):
        is_classfication=True
    
    print("Classification:",is_classfication)

    np_array=np.array(data_arr_str)
    data_arr=np_array.astype(np.float)

    #Normalize the attributes
    max_list=np.max(data_arr,axis=0)
    min_list=np.min(data_arr,axis=0)

    # print(num_cols)
    
    for j in range(num_cols-1):
        for i in range(num_rows):
            data=float(data_arr[i][j]-min_list[j])/(max_list[j]-min_list[j])
            data_arr[i][j]=data

    # print(max_list[13])
    # print(min_list[13])
    # print(data_arr[0][13])        
    # print(data_arr)    
    temp_arr=data_arr.tolist()
    # print(temp_arr)


    #Separate attribute list and class
    if is_classfication:
        for row in temp_arr:
            split_rows=(row[0:num_cols-1],row[num_cols-1])
            attribute_vector_list.append(split_rows)

        # for n in range(num_rows):
        #     print(attribute_vector_list[n][1])

    else: 
        for row in temp_arr:
            split_rows=(row[0:num_cols-1],((row[num_cols-1]-min_list[num_cols-1])/(max_list[num_cols-1]-min_list[num_cols-1])))
            attribute_vector_list.append(split_rows)

    #write data into file 
    file=open(data_out,'w')
    writer=csv.writer(file)
    if has_header == True:
        header=[]
        for j in valid_string_att_index:
                header.append(string_att_map[j])
        writer.writerow(header)
    for i in range(len(attribute_vector_list)):
        row=[]
        for j in range(len(attribute_vector_list[i][0])):
            row.append(attribute_vector_list[i][0][j])

        row.append(attribute_vector_list[i][1])
        writer.writerow(row)

    
    file.close()
    # print(attribute_vector_list[0][1])
    return attribute_vector_list,is_classfication,len(unique_list),num_rows,num_cols
    

def startBuildNN(neurons,dataset,allowed_err,is_classification,num_classes):
    # form neural network structure
    neural_struct=[]
    #attribute layer
    neural_struct.append(len(dataset[0][0]))
    #hidden layers
    for x in range(len(neurons)):
        neural_struct.append(neurons[x])

    #classification layer
    if(is_classification): 
        classes=num_classes
    else:
        classes=1
    
    neural_struct.append(classes)
    print("Neural structure: ",neural_struct)
    nn=Network(neural_struct)
    # print("Final")
    # print(nn.weights[0][1][1])
    buildNN(nn,neural_struct,dataset,allowed_err,is_classification)
    return nn,neural_struct

def buildNN(nn,neural_struct,ds,err_tol,is_classification):
    print("Training started...")
    global max_iterations
    global ita
    iterations=0
    accuracy=0.0
    if (is_classification):
        precision=0
    else:
        precision=1 
    # while iterations <= max_iterations or error >= allowed_err:
    err_tol=False
    while (iterations < max_iterations and err_tol==False):
        iterations+=1
        correct=[]
        # wrong=0.0
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
                        if(is_classification):
                            if(i == int(ds[x][1])):
                                delta[l][i]=(nodes[l][i])*(1-nodes[l][i])*(1-nodes[l][i])
                            else:
                                delta[l][i]=(nodes[l][i])*(1-nodes[l][i])*(-nodes[l][i])
                        else:
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
            if (is_classification):
                flag_1=False
                flag_2=True
                for a in range(1,neural_struct[l]+1):
                    if a == ds[x][1]:
                        if (1 - nodes[l][a] ) <= err_tol:
                            flag_1=True
                    else:
                        if (nodes[l][a] - 0) > err_tol:
                            flag_2=False

                if flag_1 and flag_2:
                    correct.append(True)
                else:
                    correct.append(False)
            else:
                if (float(ds[x][1])-nodes[l][1] <= err_tol):
                    correct.append(True)     
                else:
                    correct.append(False)
            
        err_tol=True
        for x in range(len(ds)):
            if correct[x]==False:
                err_tol=False
                break

        # accuracy=float(correct/(correct+wrong))*100
        # print ("Training...iteration:",iterations)
        # print(nodes[2])
        # print(ds[x][1])
            # if iterations == max_iterations:
                # print("Instance %d: "%x)
                # print(round(nodes[len(neural_struct)-1][1],1))
                # print(nn.weights)
    printWeights(nodes,neural_struct)            
    print("Training complete. Iterations:",iterations)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def printWeights(nodes,neural_struct):
    for l in range(1,len(neural_struct)):
        print(neural_struct[l])
        if(l == len(neural_struct)-1):
            print("Output Layer:")
        else:
            print("Hidden Layer%d:"%(l))
        
        for i in range(neural_struct[l]):
                print("Neuron%d weights"%(i+1),nodes[l][i+1])

def testAccuracy(nn,neural_struct,ds,is_classification,err_tol,par):
    if (is_classification):
        precision=0
    else:
        precision=1 

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
                    if (is_classification): 
                        flag_1=False
                        flag_2=True
                        for a in range(1,neural_struct[l]+1):
                            # if par=="Training":
                            #     print(ds[x][1])
                            #     print(err_tol)
                            if a == ds[x][1]:
                                # if par=="Training":
                                #     print("a",a)
                                #     print("1",1 - nodes[l][a])
                                if (1 - nodes[l][a] ) <= err_tol:
                                    flag_1=True
                            else:
                                # if par=="Training":
                                #     print("a",a)
                                #     print("0",nodes[l][a] - 0)
                                if (nodes[l][a] - 0) > err_tol:
                                    flag_2=False
                        # if par=="Training":            
                            # print(flag_1)
                            # print(flag_2)
                        if flag_1 and flag_2:
                            correct+=1
                            # if par=="Training":
                                # print("correct: ",nodes[l],int(ds[x][1]))
                        else:
                            wrong+=1
                            # if par=="Training":
                            #     print("wrong: ",nodes[l], int(ds[x][1]))
                    else:
                        if (float(ds[x][1])-nodes[l][1] <= err_tol):
                            correct+=1   
                        else:
                            wrong+=1

    print(par," accuracy: ",str(float(correct/(correct+wrong)*100)))

def main(args):
    print(args)
    print("Program executing...")
    in_data=args[1]
    out_data=args[2]
    train_per=int(args[3])
    err_tol=float(args[4])
    num_hidden=int(args[5])
    neurons=[]
    for i in range(num_hidden):
        neurons.append(int(args[6+i]))
    
    # in_data="iris.csv"
    # out_data="rough.csv"
    # train_per=80
    # err_tol=0.5
    # num_hidden=1
    # neurons=[5]

    processed_list,is_classification,num_classes,num_rows,num_cols=preProcess(in_data,out_data)
    # print(processed_list)  
    train,test=np.vsplit(processed_list,np.array([int(num_rows*train_per)/100,]))
    nn,n_struct=startBuildNN(neurons,train,err_tol,is_classification,num_classes)
    testAccuracy(nn,n_struct,train,is_classification,err_tol,"Training")
    testAccuracy(nn,n_struct,test,is_classification,err_tol,"Test")
main(sys.argv)