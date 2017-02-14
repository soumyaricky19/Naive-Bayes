from __future__ import print_function
import sys
import csv
from Node import Node
from math import log
import random

accuracy_improvement_threshold=0
node_num=0
leaf_node_num=0

def read_data(data_file):
    file=open(data_file)
    read=csv.reader(file)
    is_header=True
    global row_num
    global attribute_vector_list
    global attribute_num
    global map

    attribute_vector_list=[]
    attribute_list=[]
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

    
def startTree():
    global root
    root=Node()
    p=0
    n=0
    # print("Number %d" %row_num)
    for x in range(row_num):
            cl=attribute_vector_list[x][1][0]
            if cl=='0':
                p+=1
            else:
                n+=1
    # print("root +: %d"%p)
    # print("root -: %d"%n)
    root.p_num=p
    root.n_num=n
    root.possible_attributes_index=range(attribute_num)
    root=buildTree(root,attribute_vector_list)     
    

def buildTree(node,attribute_list):
    if node.decision==2:
        global node_num
        global leaf_node_num
        node_num+=1
        is_pure=False
        best_ig=0
        possible_attributes_index=[]
        for x in range(len(node.possible_attributes_index)):
            possible_attributes_index.append(node.possible_attributes_index[x])         
        
        best_count=[[0,0],[0,0]]
        ig=0
        node_entropy=calcEntropy(node.n_num,node.p_num)
        if (node_entropy == 0 ):
            is_pure=True
        if ((not node.possible_attributes_index) or is_pure ):
            leaf_node_num+=1
            if (node.n_num > node.p_num):
                node.decision=0
            else:
                if(node.n_num < node.p_num):
                    node.decision=1
                else:
                    node.decision=random.randrange(1)    
            # print('CLASSIFIED as :%d'%node.decision)
            return node
        best_attribute_index=possible_attributes_index[0]
        best_false_attribute_list=[]
        best_true_attribute_list=[]

        # print("Node +: %d"%node.p_num)
        # print("Node -: %d"%node.n_num)
        # print("Node Entropy: %f"%node_entropy)
    
        # print("Possible attribute index: "+str(possible_attributes_index))
    
        for attribute_index in possible_attributes_index:
            false_attribute_list=[]
            true_attribute_list=[]
            count=[[0,0],[0,0]]
            for x in range(len(attribute_list)):
                at=attribute_list[x][0][attribute_index]
                cl=attribute_list[x][1][0]
                if at=='0':
                    false_attribute_list.append(attribute_list[x])
                    if cl=='0':
                        count[0][0]+=1     
                    else:
                        count[0][1]+=1    
                else:
                    true_attribute_list.append(attribute_list[x])
                    if cl=='0':
                        count[1][0]+=1     
                    else:
                        count[1][1]+=1
            #Information Gain
            #print('Count status: '+str(count))
            false_entropy=calcEntropy(count[0][0],count[0][1])
            true_entropy=calcEntropy(count[1][0],count[1][1])
            false_weight=(count[0][0]+count[0][1])/float(count[0][0]+count[0][1]+count[1][0]+count[1][1])
            true_weight=(count[1][0]+count[1][1])/float(count[0][0]+count[0][1]+count[1][0]+count[1][1])
            entropy=false_entropy*false_weight+true_entropy*true_weight
            ig=node_entropy-entropy
            #print('Entropy with -'+str(attribute_index)+ ': '+str(entropy))

            if(ig>best_ig):
                best_ig=ig
                best_attribute_index=attribute_index
                best_count=count
                best_false_attribute_list=false_attribute_list
                best_true_attribute_list=true_attribute_list
          
        # print("Best:")
        # print(best_ig)
        # print(best_attribute_index)
        # print(best_count)
        # print("Possible"+str(possible_attributes_index))
    
        possible_attributes_index.remove(best_attribute_index)    
        node.att_to_split_on_index=best_attribute_index
        node.false_child=buildTree(Node(best_attribute_index,0,possible_attributes_index,best_count[0][0],best_count[0][1]),best_false_attribute_list)

        node.true_child=buildTree(Node(best_attribute_index,1,possible_attributes_index,best_count[1][0],best_count[1][1]),best_true_attribute_list)
        node.n_num=best_count[0][0]+best_count[0][1]
        node.p_num=best_count[1][0]+best_count[1][1]

    return node


def calcEntropy(n,p):
    if (n==0 or p==0):
        return 0
    else:
        pr1=n/float(n+p)
        pr2=p/float(n+p)
        p1=log(pr1,2)
        p2=log(pr2,2)

        e=-pr1*p1-pr2*p2
        return e

def findAccuracy(node,attribute_vector_list):
    correct=0
    wrong=0
    for x in attribute_vector_list:
        n=node
        while (n.decision == 2):
            i=n.att_to_split_on_index
            if x[0][i] == '0':
                n=n.false_child
            if x[0][i] == '1':
                n=n.true_child
        if x[1] == str(n.decision):
            # print("Correct: ",str(x[0])," class:",str(x[1])," Pred: ",str(n.decision))
            correct+=1
        else:
            wrong+=1
            # print("Wrong: ",str(x[0])," class:",str(x[1])," Pred: ",str(n.decision))

    # print(correct)
    # print(wrong)
    return correct/float(correct+wrong)*100

def copyTree(node):
    if (node!=None):
        return Node(node.split_att_index,node.split_att_value,node.possible_attributes_index,node.p_num,node.n_num
        ,node.decision,node.att_to_split_on_index,copyTree(node.false_child),copyTree(node.true_child))
    else:
        return None

def prune(factor):
    global node_num
    global leaf_node_num
    global root
    global best_tree
    global attribute_vector_list
    global count
    n=int(factor*node_num)
    val_accuracy=findAccuracy(root,attribute_vector_list)
    curr_accuracy=val_accuracy 
    while curr_accuracy <= val_accuracy+accuracy_improvement_threshold:       
        tree=copyTree(root)  
        for x in range(n):
            node_num=0
            leaf_node_num=0
            countTree(tree)
            k=random.randrange(node_num-leaf_node_num)
            count=0
            pruneTree(tree,k)           
        curr_accuracy=findAccuracy(tree,attribute_vector_list)
    best_tree=tree

def pruneTree(node,k):
    global count
    if(node.decision==2):  
        count+=1   
        if (count==k and node.split_att_value !=2):
            # print(node.att_to_split_on_index)
            if (node.n_num > node.p_num):
                node.decision=0
            else:
                if(node.n_num < node.p_num):
                    node.decision=1
                else:
                    node.decision=random.randrange(1)
        pruneTree(node.false_child,k)
        pruneTree(node.true_child,k)
    
def countTree(node):
    global node_num
    global leaf_node_num
    node_num+=1
    if(node.decision!=2):
        leaf_node_num+=1
    else:
      countTree(node.false_child)  
      countTree(node.true_child)


def printTree(node,k):   
    if(node.decision!=2):
        print(str(map[node.split_att_index]) + " = " + str(node.split_att_value),end="")
        print(": %d" %node.decision)
    else:
        if (node.split_att_value != 2):           
            print(str(map[node.split_att_index]) + " = " + str(node.split_att_value))    
            k+=1        
            for i in range(k):
                print("|",end='') 
                print(" ",end="")
                    
            
        printTree(node.false_child,k)
            
        if (node.split_att_value != 2):          
            for i in range(k):
                print("|",end='') 
                print(" ",end="")
                   
        
        printTree(node.true_child,k)

def printAccuracy(node,str):
    print("\nNumber of",str,"instances = %d"%row_num)
    print("Number of",str,"attributes = %d"%attribute_num)
    print("Accuracy of the model on the",str,"dataset = %f"%findAccuracy(node,attribute_vector_list))


def main(args):
    global root
    global best_tree
    global node_num
    global leaf_node_num
    print("Program executing...")
    training=args[1]
    validation=args[2]
    test=args[3]
    factor=float(args[4])
    
    read_data(training)
    startTree()
    printTree(root,0) 

    node_num=0
    leaf_node_num=0
    countTree(root)
    print('\nPre-Pruned Accuracy')
    print('-------------------------------------')
    print ('Total number of nodes in the tree = %d'%node_num)
    print('Number of leaf nodes in the tree = %d'%leaf_node_num)
    printAccuracy(root,"training")
    read_data(validation)
    printAccuracy(root,"validation")
    read_data(test)
    printAccuracy(root,"testing")
    
    read_data(validation)
    prune(factor)
       
    node_num=0
    leaf_node_num=0
    # printTree(best_tree,0)
    countTree(best_tree)
    print('\nPost-Pruned Accuracy')
    print('-------------------------------------')
    print ('Total number of nodes in the tree = %d'%node_num)
    print('Number of leaf nodes in the tree = %d'%leaf_node_num)
    read_data(training)
    printAccuracy(best_tree,"training")
    read_data(validation)
    printAccuracy(best_tree,"validation")
    read_data(test)
    printAccuracy(best_tree,"testing")


main(sys.argv)
