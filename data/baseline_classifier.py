import numpy as np
from collections import Counter
import operator

ppdb_training_file = 'random_sample_ppdb-1.0-1-o2m.gz_500_orig'
mwe_indices_file = 'data/random_sample/mwe_indices.txt'

def load_ppdb(ppdb_file):
  entries = []
  f = open(ppdb_file, 'r')
  line = f.readline()
  while line != '':
    entries.append(line[:-1].split(' ||| '))
    line = f.readline()
  f.close()
  return entries

def load_mwe_indices(mwe_indices_file):
  indices = []
  f = open(mwe_indices_file, 'r')
  line = f.readline()
  while line != '':
    indices.append(int(line[:-1]))
    line = f.readline()
  f.close()
  return set(indices)


entries = load_ppdb(ppdb_training_file)
x = [1 for entry in entries]
y = [1 if i in load_mwe_indices(mwe_indices_file) else -1 for i in range(len(entries))]

accuracies = []
for fold in range(10):
  start = int(fold*(len(entries)/float(10)))
  end = int((fold+1)*(len(entries)/float(10)))
  test = entries[start:end]
  train = entries[:start] + entries[end:]
  test_labels = y[start:end]
  train_labels = y[:start] + y[end:]
  majority_label = sorted(Counter(train_labels).iteritems(), key=operator.itemgetter(1))[-1][0]
  accuracies.append(sum([int(test_label == majority_label) for test_label in test_labels])/float(len(test_labels)))

#find average accuracy:
accuracy = sum(accuracies)/float(len(accuracies))
print accuracy
