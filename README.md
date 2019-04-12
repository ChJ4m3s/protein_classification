# Protein classification
Dependancies:
* pip install -U scikit-learn
* pip install -U matplotlib
* https://github.com/ysig/GraKeL

# Files & folders
* GenKernel.py: generate the kernel from testing set and measures accuracy
* test.py: test the model with a testing set
* kernel.txt: the output of GenKernel.py
* labelsPredicted.txt: the predicted labels
* labelsTest.txt: the real labels of test set
* labels.txt: labels of training set
* trainingSet: folder containing proteins used for training
* testSet: folder containing proteins used for testing
* trainingSet/fileList.txt: list of filenames
* testSet/fileList.txt: list of filenames

# Usage
* launch GenKernel.py to generate similarity matrix (kernel.txt), labels.txt and see the accuracy of the model
* launch test.py to generate labelsPredicted.tx, labelsTest.txt and see the accuracy of the model on the test set (it needs kernel.txt to be generated before and the proteins used for training in trainingSet)
