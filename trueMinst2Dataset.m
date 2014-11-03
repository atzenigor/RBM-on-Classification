testdata = loadMNISTImages('t10k-images.idx3-ubyte')';
testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
data = loadMNISTImages('train-images.idx3-ubyte')';
labels = loadMNISTLabels('train-labels.idx1-ubyte');

labels = labels + 1;
testlabels = testlabels + 1;