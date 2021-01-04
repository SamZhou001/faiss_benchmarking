#!/usr/bin/env python
# coding: utf-8

from build import *
import time
import faiss
import os

def search(index, xq, k, gt):
    t2 = time.time()
    D, I = index.search(xq, k)
    t3 = time.time()
    duration = (t3 - t2) / len(xq)
    recall = recall_at_r(I, gt, k)
    return duration, I, recall

def recall_at_r(I, gt, r):
    assert I.ndim == 2
    assert gt.ndim == 2
    nq, topk = I.shape
    assert r <= topk
    sum = 0
    for i in range(len(I)):
        sum += len(set(I[i][:r]) & set(gt[i][:r]))/float(r)
    return sum / float(nq)

def runFlat(xb, xq, k, metric):
    t0 = time.time()
    if metric == 'L2':
        index = faiss.IndexFlatL2(d)
    if metric == 'L1':
        index = faiss.IndexFlat(d, faiss.METRIC_L1)
    index.do_polysemous_training = True
    index.train(xb)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    t2 = time.time()
    for i in range(5):
        D, I = index.search(xq, k)
    t3 = time.time()
    duration = (t3 - t2) / len(xq) / 5
    return duration, build_time, I

def runBinary(xb, xq, gt):
    t0 = time.time()
    adjusted_arr = []
    for i in range(len(xb)):
        mean = sum(xb[i]) / len(xb[i])
        binary = []
        term = 0
        for j in range(len(xb[i])):
            term *= 2
            if xb[i][j] >= mean:
                term += 1
            if j % 8 == 7:
                binary.append(term)
                term = 0
        adjusted_arr.append(binary)
    adjusted_arr = np.array([np.array(vec) for vec in adjusted_arr])
    adjusted_arr = adjusted_arr.astype('uint8')
    xb = adjusted_arr
    adjusted_arr_q = []
    for i in range(len(xq)):
        mean = sum(xq[i]) / len(xq[i])
        binary = []
        term = 0
        for j in range(len(xq[i])):
            term *= 2
            if xq[i][j] >= mean:
                term += 1
            if j % 8 == 7:
                binary.append(term)
                term = 0
        adjusted_arr_q.append(binary)
    adjusted_arr_q = np.array([np.array(vec) for vec in adjusted_arr_q])
    adjusted_arr_q = adjusted_arr_q.astype('uint8')
    xq = adjusted_arr_q
    index = faiss.IndexBinaryFlat(d)
    index.add(xb)
    t1 = time.time()
    build_time = t1-t0
    total_dur = 0
    total_recall = 0
    for i in range(5):
        duration, I, recall = search(index, xq, 20, gt)
        total_dur += duration
        total_recall += recall
    duration = total_dur / 5
    recall = total_recall / 5
    return duration, recall, build_time

def runHNSW(test, query, gt, metric, bench):
    durs_flat = []
    durs_SQ = []
    recalls_flat = []
    recalls_SQ = []
    build_ts_flat = []
    build_ts_SQ = []
    hyp_flat = []
    hyp_SQ = []
    idx = 0
    for M in [4, 8, 16]:
        for efSearch in [4, 8, 16, 32, 64]:
            for efConstruction in [40, 80, 128, 256]:
                index, build_time = buildHNSW(test, M, efConstruction, efSearch, 'flat', metric)
                duration, I, recall = search(index, query, 20, gt)
                durs_flat.append(1/duration)
                recalls_flat.append(recall)
                build_ts_flat.append(build_time)
                hyp_flat.append([M, efSearch, efConstruction])
                index, build_time = buildHNSW(test, M, efConstruction, efSearch, 'SQ', metric)
                duration, I, recall = search(index, query, 20, gt)
                durs_SQ.append(1/duration)
                recalls_SQ.append(recall)
                build_ts_SQ.append(build_time)
                hyp_SQ.append([M, efSearch, efConstruction])
                idx += 1
                print('HNSW: ' + str(idx) + '/60', end = '\r')
    filtered_durs_flat, filtered_recalls_flat, filtered_build_flat, filtered_hyp_flat = chooseMax(durs_flat, recalls_flat, build_ts_flat, hyp_flat)
    filtered_durs_SQ, filtered_recalls_SQ, filtered_build_SQ, filtered_hyp_SQ = chooseMax(durs_SQ, recalls_SQ, build_ts_SQ, hyp_SQ)
    bench['HNSWFlat'] = [filtered_durs_flat, filtered_recalls_flat, filtered_build_flat, filtered_hyp_flat]
    bench['HNSWSQ'] = [filtered_durs_SQ, filtered_recalls_SQ, filtered_build_SQ, filtered_hyp_SQ]
    os.system('cls')
    return bench

def runIVF(test, query, gt, metric, bench):
    durs = []
    recalls = []
    build_ts = []
    hyp = []
    idx = 0
    for nlist in [64, 128, 256, 512]:
        for nprobe in [1, 2, 4, 8, 16, 32, 64, 128]:
            if nprobe > nlist:
                continue
            if nlist > len(test):
                continue
            else:
                index, build_time = buildIVFFlat(test, nlist, nprobe, metric)
                duration, I, recall = search(index, query, 20, gt)
                durs.append(1/duration)
                recalls.append(recall)
                build_ts.append(build_time)
                hyp.append([nlist, nprobe])
                idx += 1
                print('IVF: ' + str(idx) + '/31', end = '\r')
    filtered_durs, filtered_recalls, filtered_build, filtered_hyp = chooseMax(durs, recalls, build_ts, hyp)
    bench['IVFFlat'] = [filtered_durs, filtered_recalls, filtered_build, filtered_hyp]
    os.system('cls')
    return bench

def runIVFPQ(test, query, gt, metric, bench):
    durs = []
    recalls = []
    build_ts = []
    hyp = []
    idx = 0
    for M in [4, 8, 16]:
        for nlist in [512, 1024, 2048]:
            for nprobe in [1, 5, 10, 50]:
                if nlist > len(test):
                    continue
                index, build_time = buildIVFPQ(test, nlist, M, nprobe, metric)
                duration, I, recall = search(index, query, 20, gt)
                durs.append(1/duration)
                recalls.append(recall)
                build_ts.append(build_time)
                hyp.append([M, nlist, nprobe])
                idx += 1
                print('IVFPQ: ' + str(idx) + '/36', end = '\r')
    filtered_durs, filtered_recalls, filtered_build, filtered_hyp = chooseMax(durs, recalls, build_ts, hyp)
    bench['IVFPQ'] = [filtered_durs, filtered_recalls, filtered_build, filtered_hyp]
    os.system('cls')
    return bench

def runLSH(test, query, gt, bench):
    durs = []
    recalls = []
    build_ts = []
    hyp = []
    idx = 0
    for nbit in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        index, build_time = buildLSH(test, nbit)
        duration, I, recall = search(index, query, 20, gt)
        durs.append(1/duration)
        recalls.append(recall)
        build_ts.append(build_time)
        hyp.append(nbit)
        idx += 1
        print('LSH: ' + str(idx) + '/8', end = '\r')
    filtered_durs, filtered_recalls, filtered_build, filtered_hyp = chooseMax(durs, recalls, build_ts, hyp)
    bench['LSH'] = [filtered_durs, filtered_recalls, filtered_build, filtered_hyp]
    os.system('cls')
    return bench

def runPQ(test, query, gt, bench):
    durs = []
    recalls = []
    build_ts = []
    hyp = []
    idx = 0
    for ht in [36, 40, 44, 48, 52, 56]:
        index, build_time = buildPQ(test)
        index.search_type = faiss.IndexPQ.ST_polysemous
        index.polysemous_ht = ht
        duration, I, recall = search(index, query, 20, gt)
        durs.append(1/duration)
        recalls.append(recall)
        build_ts.append(build_time)
        hyp.append(ht)
        idx += 1
        print('PQ: ' + str(idx) + '/6', end = '\r')
    filtered_durs, filtered_recalls, filtered_build, filtered_hyp = chooseMax(durs, recalls, build_ts, hyp)
    bench['PQ'] = [filtered_durs, filtered_recalls, filtered_build, filtered_hyp]
    os.system('cls')
    return bench
