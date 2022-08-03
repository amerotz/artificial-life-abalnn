import argparse
import json
import ast

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str)
args = parser.parse_args()

with open(args.input, 'r') as f:
    string = f.read()

genes = ast.literal_eval(string)

#print(genes)

'''
genes = {
    'a': ['a1', 'a2', 'a3'],
    'a1': ['a11', 'a12'],
    'a2': [],
    'a3': [],
    'a11': [],
    'a12': [],
    'b': ['b1'],
    'b1': ['b11', 'b12'],
    'b11':[],
    'b12': ['b121', 'b122'],
    'b121': [],
    'b122': ['bob'],
}

'''
def printDic(dic, indent, survivors):
    for i, d in enumerate(dic):
        s = ''
        for i in range(indent):
            s += '\t'
        for c in survivors:
            if c == d:
                s += '* '
                break
        s += d[:8] + ' ... '
        print(s)
        if type(dic[d]) is dict:
            printDic(dic[d], indent+1, survivors)

def rename(dic, prefix, names):
    if type(dic) is list:
        new = []
    else:
        new = {}
    for i, d in enumerate(dic):
        if d in names:
            name = names[d]
        else:
            name = prefix + str(i)
            names[d] = name

        if type(dic) is list:
            new.append(name)
        else:
            new[name] = rename(dic[d], name, names)
    return new

stop = False
genealogy = {}
old = {}
#while not stop:
for i in range(5):
    #print(genes)
    #printDic(genes, 0)
    delete = []
    for g1 in genes:
        for g2 in genes:
            if g1 in genes[g2]:
                if not g2 in genealogy:
                    genealogy[g2] = {}
                genealogy[g2][g1] = genes[g1]
                delete.append(g1)

    for d in delete:
        if d in genealogy:
            del genealogy[d]

    if genealogy == old:
        stop = True
    else:
        old = genealogy
        genes = genealogy
    #print('--------------')


genes = rename(genes, '', {})
printDic(genes, 0, survivors)
