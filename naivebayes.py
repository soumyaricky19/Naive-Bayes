import os
import sys
import re
# def readData:
#     file=open(data_in)
#     read=csv.reader(file)
    # arr=[]
words=[]


def getFiles(path):
    d=dict()
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
                        process(row,d)
                elif(lines !=0):
                    break
    return d

def process(row,d):
    global words
    ignore=[
        "",
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves"
        ]
    # row = re.sub("\s+\n+[^a-zA-Z]*"," ",row)

    #row = re.sub("[^a-zA-Z]*"," ",row)
    
    row = re.sub(r"[^a-zA-Z0-9']+"," ",row)
    row = row.split(" ")
    for x in row:
        if (x.lower() not in ignore):
            words.append(x)
    
    for word in words:
        if word not in d:
            d[word]=1
        else:
            d[word]=d[word]+1
    
def sortRev(d):
    return sorted(d.items(), key=lambda value: value[1], reverse=True)

def main(args):
    l=["20news-bydate-train\\alt.atheism","20news-bydate-train\\comp.graphics"]
    data=[]
    d=()
    for i in range(len(l)):
        d=getFiles(l[i])
        d=sortRev(d)
        
        data.append(d)

    #print(reversed(sortD))
    print(data)

main(sys.argv)