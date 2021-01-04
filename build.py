#!/usr/bin/env python
# coding: utf-8

import time
import faiss
import numpy as np

d = 2048

def chooseMax(durs, recalls, build_ts, hyp):
    points = []
    for i in range(len(durs)):
        points.append([durs[i], recalls[i], build_ts[i], hyp[i]])
    points.sort(reverse = True)
    back = 0
    for i in range(len(points)):
        if i != 0:
            if points[i - back][1] <= points[i - 1 - back][1]:
                points.pop(i - back)
                back += 1
    filtered_durs = [x[0] for x in points]
    filtered_recalls = [x[1] for x in points]
    filtered_build = [x[2] for x in points]
    filtered_hyp = [x[3] for x in points]
    return filtered_durs, filtered_recalls, filtered_build, filtered_hyp

def buildHNSW(xb, M, efConstruction, efSearch, tp, metric):
    t0 = time.time()
    if tp == 'flat':
        if metric == 'L1':
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L1)
        if metric == 'L2':
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    if tp == 'SQ':
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
    index.hnsw.efSearch = efSearch
    index.hnsw.efConstruction = efConstruction
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    return index, build_time

def buildIVFFlat(xb, nlist, nprobe, metric):
    t0 = time.time()
    if metric == 'L1':
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L1)
    if metric == 'L2':
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.nprobe = nprobe
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    return index, build_time

def buildIVFPQ(xb, nlist, M, nprobe, metric):
    t0 = time.time()
    if metric == 'L1':
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L1)
    if metric == 'L2':
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
    index.nprobe = nprobe
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    return index, build_time

def buildLSH(xb, nbits):
    t0 = time.time()
    index = faiss.IndexLSH(d, nbits)
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    return index, build_time

def buildPQ(xb):
    t0 = time.time()
    index = faiss.IndexPQ(d, 16, 8)
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    return index, build_time

