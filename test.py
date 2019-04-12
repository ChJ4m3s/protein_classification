"""
 ====================================
||  Classification on the PROTEINS  ||
||          Kernel testing          ||
||  ------------------------------  ||
||  Giacomo Chiarot                 ||
||  giacomochiarot@gmail.com        ||
 ====================================
"""
from __future__ import print_function
print(__doc__)

from grakel import GraphKernel
from grakel import Graph
from sklearn import svm
import numpy as np

"""
Reads the list of files and returns the name of the files and their value which represent the class of each protein
"""
def readProteins():
    print("-- reading file list")
    proteinNames = []
    labels = []
    testLabels = []
    f = open("testSet/fileList.txt", "r")
    for line in f:
        lineDivided = line.split(' ')
        proteinNames.append('testSet/' + lineDivided[0])
        labels.append(int(lineDivided[1]))
    f.close()
    f = open("trainingSet/fileList.txt", "r")
    for line in f:
        lineDivided = line.split(' ')
        proteinNames.append('trainingSet/' + lineDivided[0])
        testLabels.append(int(lineDivided[1]))
    f.close()
    return proteinNames, labels, testLabels

"""
Reads the list of arcs for each protein and stores them as graphs
"""
def readGraphs():
    proteinNames, labelValues, testLavelValues = readProteins()
    print("-- reading graphs")
    graphs = []
    for file in proteinNames:
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
    return graphs, labelValues, testLavelValues

"""
Create the model for SVM
"""
def createModel(K, labels):
    print("-- computing the score with SVM")
    mod = svm.SVC(kernel='precomputed')
    mod.fit(K, labels)
    return mod

"""
Computing the score
"""
def computingScores(mod, features, labels):
    predictions = mod.predict(features)
    score = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            score += 1
    np.savetxt("labelsPredicted.txt", np.array(predictions), fmt="%s")
    np.savetxt("labelsTest.txt", np.array(labels), fmt="%s")
    return score / len(labels)

"""
Computes the weisfeiler_lehman kernel
"""
def computeKernel(graphs):
    print("-- computing kernel")
    wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 5}, {"name": "subtree_wl"}], normalize=True)
    return  wl_kernel.fit_transform(graphs)

def getFeatures():
    graphs, labels, testLabels = readGraphs()
    K = computeKernel(graphs)
    n = len(K) - len(labels)
    features = K[n:]
    result = []
    for i in range(len(features)):
        row = features[i]
        row = row[: len(row) - len(labels)]
        result.append(row)
    return result, labels, testLabels

def main():
    features, labels, testLabels = getFeatures()
    K = np.loadtxt("kernel.txt")
    model = createModel(K, testLabels)
    result = computingScores(model, features, labels)
    print("Accuracy is: " + str(result))

if __name__ == "__main__":
    main()
