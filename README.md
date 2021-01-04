# Benchmarking Faiss Algorithms

## Introduction

The following benchmarking experiments aim to test the speed and accuracy of k-nearest neighbor search algorithms from the [Faiss](https://github.com/facebookresearch/faiss) library for similarity search, developed by Facebook AI Research. The dataset used in this experiment, which is not included in this repo due to confidentiality, contains a vectorized list of around 14500 items from different Shopify stores, and the Faiss algorithms are used to determine the most similar products for each Shopify item.

## Terminology
* **Query vectors**: A list of input vectors for which most similar vectors are searched.
* **Base vectors**: The pool of vectors searched by k-nearest neighbor algorithms. For each query vector, a list of the most similar base vectors are returned.
* **Queries per second**: The number of query vectors that an algorithm can search in one second.
* **Recall@k**: The accuracy of the algorithms when returning the top k most similar base vectors.
* **Faiss index**: Another name for Faiss algorithm.
* **Metric**: The method used to compute the distance between each vector. For example, L1 distance returns the sum of the absolute difference within each component of  two vectors. L2 distance, or Euclidean distance, returns the straight line distance between two vectors.

## Algorithms
* **Flat**: Brute force search. The distance between a query vector and each base vector is computed, and the base vectors with the least distances are returned.
* **Binary**: Each vector is encoded in the following way — each component is reassigned 1 if the component's value is greater or equal to the average of all components of the vector, and is assigned 0 otherwise. Then, every eight components (bits) are encoded into an uint8 byte, and brute force search is used.
* **Inverted File Index (IVFFlat)**: The vectors are grouped into a number of lists, where each list is stored as an inverted file. Then, given a query vector, only the most relevant lists of vectors are searched.
* **Product Quantizer (PQ)**: The base vectors are sliced into many subvectors. Then all subvectors are divided into groups, and the centroid for each group is computed. Each subvector is replaced by the id of its nearest centroid, and therefore each base vector is encoded as a list of centroids. After the quantization, brute force search is used.
* **IVFPQ**: PQ quantization with IVF search.
* **Hierarchical Navigable Small World (HNSWFlat)**: The base vectors are put into a layered, connected graph. For each query vector, the graph is traversed until the distance between the node (base vector) and the query reaches a local minimum.
* **HNSW with Scalar Quantizer (HNSWSQ)**: Each base vector is encoded as a scalar, and then HNSW search is used.
* **Locality Sensitive Hashing (LSH)**: The vectors are put into a buckets based on their proximity, so closer vectors are more likely to be in the same bucket. Give a query vector, only the most relevant buckets are searched.

## Method
Four experiments were conducted. In Experiments 1 and 2, the entire dataset of vectors is used as both base and query. Experiment 1 evaluates algorithms with L1 metric, while Experiment 2 evaluates them with L2 metric. In Experiments 3 and 4, the base vectors are assigned the first 80% of the dataset, and the query vectors are the remaining 20%. Experiment 3 evaluates algorithms with L1 metric, while Experiment 4 evaluates them with L2 metric.

For each algorithm, the base vectors are first added into the Faiss index and trained (see `build.py`), and its build time is measured. Then, a certain number of parameters are adjusted, and for each adjustment, the index searches the 20 nearest neighbors for all query vectors (see `run_algos.py` and  `make_bench.py`). The index's speed and accuracy are then recorded: the speed (QPS) is measured by the total time elapsed divided by the number of queries, and the accuracy (recall@20) is measured by the fraction of matching return values between the index and the brute force search result. After all parameters for an algorithm are adjusted and tested, and the times and accuracies recorded, the suboptimal sets of parameters are removed. A set of parameters is suboptimal is there exists another set of parameters that has both greater QPS and greater recall@20.

After each algorithm is tested, the optimal speeds and accuracies for each algorithm are plotted and compared against each other.

## Results
Below are the speed-accuracy graphs for all experiments. Note that Binary and Flat are both represented by a dot, since there are no parameters to tune for these algorithms.

Experiment 1:

![Experiment 1 graph](https://github.com/SamZhou001/faiss_benchmarking/blob/main/plots/allureL1_full.png)

Experiment 2:

![Experiment 2 graph](https://github.com/SamZhou001/faiss_benchmarking/blob/main/plots/allureL2_full.png)

Experiment 3:

![Experiment 3 graph](https://github.com/SamZhou001/faiss_benchmarking/blob/main/plots/allureL1.png)

Experiment 4:

![Experiment 4 graph](https://github.com/SamZhou001/faiss_benchmarking/blob/main/plots/allureL2.png)

All four graphs show a trade-off between speed and accuracy: as the accuracy increases, the speed seems to drop off. Furthermore, in all four experiments, it seems that HNSWFlat performs the best, as its graph is the most upper-right out of all algorithms. This demonstrates that HNSWFlat can achieve the best accuracy in a short amount of time. IVFFlat also seems to do well, although slightly worse than HNSWFlat. Interestingly, the brute force algorithm runs significantly faster (by nearly 30 times) in L2 distance compared to L1 distance.

## Discussion

Although HNSWFlat seems to be the optimal algorithm, one must keep in mind that because it creates a layered connected graph, it requires significantly more memory than other algorithms. Because the dataset used in these experiments is small, HNSWFlat can run in a fast time. It would be a good idea to explore the effect of dataset size on the performance of HNSWFlat or any other algorithm. Although PQ and IVFPQ seem to perform poorly on this dataset, they might do better on bigger datasets because the product quantizer, which acts as a lossy compression, significantly reduces an index's memory usage.

Another performance metric to consider is each algorithm's ability to approximate results. Although some algorithms might not be completely matching the brute force results, the base vectors that they return might still be similar. Instead of comparing each algorithm's top 20 results to the brute force top 20 results, which was done in these experiments, one can see how much of an algorithm's top 20 results appears in the top 25 results generated by brute force.
