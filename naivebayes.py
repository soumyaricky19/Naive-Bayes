import os
import sys
import re
import math

ignore=["","a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]


def getFiles(path):
    d=dict()
    total=0
    for filename in os.scandir(path):
        if filename.is_file():
            file=open(filename)
            i=0
            lines=0
            for row in file:
                if row.__contains__("Lines: "):
                    # if len(row.split("Lines: ")) == 2:
                    lines=int(row.split("Lines:")[1])
                    continue
                if (lines !=0 and i<=lines):
                    i+=1
                    if row != "\n":
                        total+=process(row,d,total)
                elif(lines !=0):
                    break
    return d,total

def process(row,d,total):
    words=[]
    global ignore
    # row = re.sub(r"-","",row)
    row = re.sub(r"[^a-zA-Z0-9]+"," ",row)
    row = row.split(" ")
    for x in row:
        if (x.lower() not in ignore):
            words.append(x.lower())
    
    for word in words:
        if word not in d:
            d[word]=1
        else:
            d[word]=d[word]+1
            
    return len(words)

def readTestData(data,struct,path,file_count):
    correct_wrong=[0,0]
    correct=0
    wrong=0
    for filename in os.scandir(path):
        if filename.is_file():
            print(filename)
            file=open(filename)
            i=0
            lines=0
            w=[]
            for row in file:
                if row.__contains__("Lines: "):
                    lines=int(row.split("Lines:")[1])
                    continue
                if (lines !=0 and i<=lines):
                    i+=1
                    if row != "\n":
                        findWords(row,w)
                elif(lines !=0):
                    break
            # print(w)
            correct,wrong=checkClass(w,data,struct,file_count,correct,wrong)
    #Accuracy
    print(correct,wrong)

def checkClass(words,data,struct,file_count,correct,wrong):
    prob_list=[]
    prior=[]
    total=0
    for i in range(len(struct)):
        total+=struct[i]
    
    for i in range(len(data)):
        prior.append(struct[i]/total)

    # print(prior)
    # print(struct)
    i=0
    for d in data:
        likelihood=0.0
        for word in words:
            # print("Probabilty",word, d.get(word,1.0)/struct[i])
            likelihood+=-math.log(d.get(word,1.0)/struct[i])
            # print(likelihood)
        likelihood+=-math.log(prior[i])
        prob_list.append(likelihood)
        i+=1
    print(prob_list.index(max(prob_list)))
    if prob_list.index(max(prob_list)) == file_count:
        correct+=1
    else:
        wrong+=1

    return correct,wrong

def findWords(row,words):
    global ignore
    row = re.sub(r"[^a-zA-Z0-9]+"," ",row)
    row = row.split(" ")
    for x in row:
        if (x.lower() not in ignore):
            words.append(x.lower())

def sortRev(d):
    return sorted(d.items(), key=lambda value: value[1], reverse=True)

def main(args):
    l=["20news-bydate-train\\alt.atheism","20news-bydate-train\\comp.graphics"]
    # l=["20news-bydate-train\\alt.atheism"]
    data=[]
    struct=[]
    d=()
    # for i in range(5):
    for i in range(2):
        d,t=getFiles(l[i])
        # d=sortRev(d)
        # print(d)
        data.append(d)
        struct.append(t)
        print(t)
    
    test_path="20news-bydate-test\\alt.atheism"
    for i in range(2):
        readTestData(data,struct,test_path,i)
    # print(data)

main(sys.argv)




