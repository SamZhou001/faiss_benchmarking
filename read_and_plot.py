#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

def str_to_lst(s):
    if type(s) == float:
        return 0
    if s[0] == '[':
        s = s[1:]
    if s[-1] == ']':
        s = s[:-1]
    sp = s.split(', ')
    ret = []
    inSub = False
    for x in sp:
        if x[0] == '[':
            x = x[1:]
            inSub = True
            sublist = []
            sublist.append(float(x))
        elif x[-1] == ']':
            x = x[:-1]
            sublist.append(float(x))
            ret.append(sublist)
            inSub = False
        elif inSub:
            sublist.append(float(x))
        else:
            ret.append(float(x))
    return ret

def read(path):
    df = pd.read_csv(path)
    benchmarks = {}
    for i in range(df.shape[0]):
        benchmarks[df['Unnamed: 0'][i]] = [str_to_lst(df['QPS'][i]), str_to_lst(df['Recall'][i]), str_to_lst(df['Build_Time'][i]), str_to_lst(df['Hyp'][i])]
    return benchmarks

def save(d, fname):
    df = pd.DataFrame.from_dict(d, orient="index")
    df.columns = ['QPS', 'Recall', 'Build_Time', 'Hyp']
    df.to_csv(fname)

def plotQPS(bench, path):
    labels = []
    for i in sorted(bench.keys()):
        if i == 'Flat' or i == 'Binary':
            plt.plot(bench[i][1], bench[i][0], 'o')
        else:
            plt.plot(bench[i][1], bench[i][0])
        labels.append(i)
    plt.xlabel('Recall@20')
    plt.ylabel('Queries Per Second')
    plt.yscale('log')
    plt.title("Faiss Algorithm Query Time vs. Recall")
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(path, bbox_inches='tight')

