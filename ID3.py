from __future__ import print_function
import sys
import csv
from Node import Node
from math import log

attribute_vector_list=[]
attribute_list=[]
attribute_num=0
num_rows=0


def create_training(training):
    file=open(training)
    read=csv.reader(file)
    is_header=True
    global num_rows
    global attribute_vector_list
    global attribute_num
    global map

    for row in read:
        if is_header:
            attribute_num=len(row)-1
            map=row[0:attribute_num]
            is_header=False
        else:
            num_rows+=1
            train=(row[0:len(row)-1],row[len(row)-1])
            attribute_vector_list.append(train)
    #attribute_vector_list[row num][column type][column num]
    
    #print(attribute_vector_list[1][1][0])
    #print(attribute_num)
    #print(map[5])
    #print('Original List length: '+str(attribute_vector_list[0]))
    print("Number of examples found: %d" %num_rows)
    print('Done feature')
    
def startTree():
    root=Node()
    p=0
    n=0
    print("Number %d" %num_rows)
    for x in range(num_rows):
            cl=attribute_vector_list[x][1][0]
            if cl=='0':
                p+=1
            else:
                n+=1
    print("root +: %d"%p)
    print("root -: %d"%n)
    root.p_num=p
    root.n_num=n
    #global possible_attributes_index
    root.possible_attributes_index=range(attribute_num)
    #print('Root list length: %d'%len(attribute_vector_list[0][0]))
    root=buildTree(root,attribute_vector_list)     
    print("Tree:- ")    
    printtree(root,0)

def buildTree(node,attribute_list):
    if node.decision==2:
        is_pure=False
        best_ig=0
        possible_attributes_index=[]
        for x in range(len(node.possible_attributes_index)):
            possible_attributes_index.append(node.possible_attributes_index[x])         
        
        best_count=0
        ig=0
        node_entropy=calcEntropy(node.n_num,node.p_num)
        if (node_entropy == 0 ):
            is_pure=True
        if ((not node.possible_attributes_index) or is_pure ):
            if (node.n_num >= node.p_num):
                node.decision=0
            else:
                node.decision=1
            print('CLASSIFIED as :%d'%node.decision)
            return node
        best_attribute_index=possible_attributes_index[0]

        print("Node +: %d"%node.p_num)
        print("Node -: %d"%node.n_num)
        print("Node Entropy: %f"%node_entropy)
        #count[attributeValue][classValue]
    
        print("Possible attribute index: "+str(possible_attributes_index))
        #print("attribute_list: "+str(attribute_list))
    
        for attribute_index in possible_attributes_index:
            false_attribute_list=[]
            true_attribute_list=[]
            count=[[0,0],[0,0]]
            for x in range(len(attribute_list)):
                #print("x: %d" %x)
                #print("attribute_index: %d"%attribute_index)
                #print('List length: %d'%len(attribute_list))
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
            print('Count status: '+str(count))
            false_entropy=calcEntropy(count[0][0],count[0][1])
            true_entropy=calcEntropy(count[1][0],count[1][1])
            false_weight=(count[0][0]+count[0][1])/float(count[0][0]+count[0][1]+count[1][0]+count[1][1])
            true_weight=(count[1][0]+count[1][1])/float(count[0][0]+count[0][1]+count[1][0]+count[1][1])
            entropy=false_entropy*false_weight+true_entropy*true_weight
            ig=node_entropy-entropy
            print('Entropy with -'+str(attribute_index)+ ': '+str(entropy))
            # print('False_entropy %f'%false_entropy)
            # print('false_weight %f'%false_weight)
            # print('true_weight %f'%true_weight)
            # print('True_entropy %f'%true_entropy)
            if(ig>best_ig):
                best_ig=ig
                best_attribute_index=attribute_index
                best_count=count
                best_false_attribute_list=false_attribute_list
                best_true_attribute_list=true_attribute_list
          
        print("Best:")
        #print(best_ig)
        print(best_attribute_index)
        #print(best_count)
        #print("Possible"+str(possible_attributes_index))
    
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
        # print("print n:%d "%n)
        # print("print p:%d "%p)
        # print("print pr1:%f "%pr1)
        # print("print pr2:%f "%pr2)
        p1=log(pr1,2)
        p2=log(pr2,2)

        e=-pr1*p1-pr2*p2
        return e


def printtree(node,k):   
    if(node.decision!=2):
        print(str(map[node.split_att_index]) + " = " + str(node.split_att_value),end="")
        print(": %d" %node.decision)
    else:
        # print("Split index: "+str(node.split_att_index))
        # print(node)
        # print(node.false_child)
        # print(node.true_child)
        if (node.split_att_value != 2):           
            print(str(map[node.split_att_index]) + " = " + str(node.split_att_value))    
            k+=1        
            for i in range(k):
                print("|",end='') 
                print(" ",end="")
                    
            
        printtree(node.false_child,k)
            
        if (node.split_att_value != 2):          
            for i in range(k):
                print("|",end='') 
                print(" ",end="")
                   
        
        printtree(node.true_child,k)


def main(args):
    print("Program executing...")
    #att_info=args[1]
    #training_data=
    training="training_set.csv"
    #test=args[3]
    
    create_training(training)  
    startTree()
    
    #tree=start_tree()
    #print('Number of nodes in tree',number_of_nodes)
    #display_tree(tree,0)

main(sys.argv)