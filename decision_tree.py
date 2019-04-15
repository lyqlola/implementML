'''
Decision Tree
'''

import math
from collections import Counter

def entropy(rows):
    # rows: list of records, the last col indicates the label y
    N = len(rows)
    cnt = Counter([row[-1] for row in rows])
    entropy = 0.0
    for key in cnt:
        p = cnt[key]/N
        entropy -= p * math.log(p, 2)
    return entropy

def gini(rows):
    # rows: list of records, the last col indicates the label y
    N = len(rows)
    cnt = Counter([row[-1] for row in rows])
    gini = 0.0
    for key in cnt:
        p = cnt[key]/N
        gini += p * (1-p)
    return gini

class DTnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col # splitting attribute index
        self.value = value # splitting attribute value
        self.results = results # dictionary, counts of labels on the current node 
        self.tb = tb # child node when answer is YES
        self.fb = fb # child node when answer is NO

def splitNode(rows, col, value):
    splitFunc = None
    # attribute is numerical or categorical?
    if isinstance(value, int) or isinstance(value, float):
        splitFunc = lambda row: row[col] >= value
    else:
        splitFunc = lambda row: row[col] == value
    set1 = [row for row in rows if splitFunc(row)]
    set2 = [row for row in rows if not splitFunc(row)]
    return (set1, set2)

# recursively build the tree
def constructDT(rows, scoref = entropy):
    if len(rows)==0: return DTnode()
    cur_entropy = scoref(rows)

    # choose the best splitting attribute and splitting value
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    colNum = len(rows[0])-1 # number of attributes
    for col in range(colNum):
        col_values = set([row[col] for row in rows])
        for value in col_values:
            set1, set2 = splitNode(rows, col, value)
            # compute information gain
            p = len(set1)/len(rows)
            gain = cur_entropy - p * scoref(set1) - (1-p) * scoref(set2)    
            if gain > best_gain and len(set1)>0 and len(set2)>0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    if best_gain>0: # if gain passes a threshold
        trueBranch = constructDT(best_sets[0])
        falseBranch = constructDT(best_sets[1])
        return DTnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return DTnode(results=Counter([row[-1] for row in rows]))

#display the DT, recursive func
def printDT(dt, indent=''):   
    if dt.results: # if leaf node
        print(str(dt.results))
    else:
        print(str(dt.col)+':'+str(dt.value)+'?') # print the condition
        print(indent+'T->', end=' ')
        printDT(dt.tb, indent+' ')
        print(indent+'F->', end=' ')
        printDT(dt.fb, indent+' ')

def predict(sample, dt):
    while not dt.results: # if not leaf node
        v = sample[dt.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= dt.value: dt = dt.tb
            else: dt = dt.fb
        else:
            if v == dt.value: dt = dt.tb
            else: dt = dt.fb
    ans = None
    for key in dt.results:
        if not ans or dt.results[key] > dt.results[ans]: 
            ans = key
    return (ans, dt.results[ans])

def prune(dt, threshold):
    # if branch is not leaf, prune it recursively
    if not dt.tb.results:
        prune(dt.tb, threshold)
    if not dt.fb.results:
        prune(dt.fb, threshold)
    # if both branches are leaves
    if dt.tb.results and dt.fb.results:
        tb, fb = [], []
        for v, c in dt.tb.results.items():
            tb += [v]*c
        for v, c in dt.fb.results.items():
            fb += [v]*c
        p = len(tb)/(len(tb)+len(fb))
        delta = entropy(tb+fb) - p*entropy(tb) - (1-p)*entropy(fb)
        if delta < threshold: # merge branches
            dt.tb, dt.fb = None, None
            dt.results = Counter(row[-1] for row in tb+fb)
        

my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

if __name__ == "__main__":
    dt = constructDT(my_data)
    printDT(dt)
    test_sample = ['(direct)','USA','yes',5]
    print(predict(test_sample, dt))
    dt1 = constructDT(my_data, gini)
    prune(dt1, 0.1)
    printDT(dt1)