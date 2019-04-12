"""
 ====================================
||  Classification on the PROTEINS  ||
||          Kernel generator        ||
||  ------------------------------  ||
||  Giacomo Chiarot                 ||
||  giacomochiarot@gmail.com        ||
 ====================================
"""
from __future__ import print_function
print(__doc__)

from sklearn.model_selection import cross_val_score
from matplotlib import pylab as plt
from sklearn.manifold import Isomap
from grakel import GraphKernel
from grakel import Graph
from sklearn import svm
import numpy as np

"""
Reads the list of files and returns the name of the files and their value which represent the class of each protein
"""
def readProteins():
    print("-- reading file list")
    f = open("trainingSet/fileList.txt", "r")
    proteinNames = []
    labels = []
    for line in f:
        lineDivided = line.split(' ')
        proteinNames.append(lineDivided[0])
        labels.append(int(lineDivided[1]))
    f.close()
    return proteinNames, labels

"""
Reads the list of arcs for each protein and stores them as graphs
"""
def readGraphs(proteinNames):
    print("-- reading graphs")
    graphs = []
    for name in proteinNames:
        file = "trainingSet/" + name
        f = open(file, "r")
        graph = {}
        labels = {}
        first = True
        for line in f:
            if first:
                first = False
            else:
                values = line.split(';')
                edges = (values[0], values[1])
                graph[edges] = float(values[2])
                labels[values[0]] = values[0]
                labels[values[1]] = values[1]
        f.close()
        graphs.append(Graph(graph, labels))
    return graphs

"""
Computes the weisfeiler_lehman kernel
"""
def computeKernel(graphs):
    print("-- computing kernel")
    wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 5}, {"name": "subtree_wl"}], normalize=True)
    return  wl_kernel.fit_transform(graphs)

"""
Computes 10-times cross validation with svm and returns the mean of results
"""
def runSVM(K, labels):
    print("-- computing scores with SVM")
    mod = svm.SVC(kernel='precomputed')
    scores = cross_val_score(mod, K, labels, cv=10)
    return np.mean(scores)

def main():
    proteinNames, labels  = readProteins()
    graphs = readGraphs(proteinNames)
    K = computeKernel(graphs)
    np.savetxt("kernel.txt", np.array(K), fmt="%s")
    np.savetxt("labels.txt", np.array(labels), fmt="%s")
    result = runSVM(K, labels)
    print("Accuracy is: " + str(result))

if __name__ == "__main__":
    main()
