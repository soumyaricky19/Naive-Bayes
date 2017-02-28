import numpy as np

class Network:
    weights=[]

    def __init__(self,structure):
        self.structure=structure    
        count=0  
        print("structure",str(structure))
        for (a,b) in zip(structure[:-1],structure[1:]):
            # print("Level: %d"%count)
            count+=1
            x=np.random.random_sample((a+1,b))
            self.weights.append(x)
            # print(x)
            
