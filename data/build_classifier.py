from svmutil import *
import numpy as np

#ppdb_file = '/home1/a/amensch/research/ppdb/ppdb-1.0-l-o2m_sample'
ppdb_training_file = 'random_sample_ppdb-1.0-1-o2m.gz_500_orig'
prepositions = ['about', 'around', 'back', 'down', 'in', 'off', 'on', 'out', 'over', 'up']
light_verbs = ['give', 'have', 'hold', 'make', 'take']
mwe_indices_file = 'mwe_indices.txt'
#verb_particle_indices = [121,157,184,205,206,207,222,279,288,373,391]
#verb_particle_indices = [113, 169, 243, 289, 490, 492] #fixed expressions
indices_list = [[121,157,184,205,206,207,222,279,288,373,391], [113, 169, 243, 289, 490, 492], [246, 321], [24, 35, 132, 142, 331, 365, 483], [114, 267, 456, 476], [126, 129, 195, 216]]

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
  return indices

def normalize(indices):
  return [index-2 for index in indices]

def in_expansion(expansion_split, word):
  return int(word in expansion_split)

def ppdb_scores(entry):
  vector = []
  spl = entry[3].split('=')
  for i in range(1, len(spl)):
    vector.append(float(spl[i].split()[0]))
  return vector

def feature_vector(entry):
  expansion_split = entry[2].split()
  #look for prepositions
  vector = [in_expansion(expansion_split, prep) for prep in prepositions]
  #look for light verbs
  vector.extend([in_expansion(expansion_split, lightv) for lightv in light_verbs])
  #look for original word in expansion
  vector.append(in_expansion(expansion_split, entry[1]))
  #include PPDB scores
  vector.extend(ppdb_scores(entry))
  return vector

#do stuff with your feature vectors
entries = load_ppdb(ppdb_training_file)
x = [feature_vector(entry) for entry in entries]
y = [1 if i in load_mwe_indices(mwe_indices_file) else -1 for i in range(len(entries))]

#partition
test_accuracy = 0
div = len(x)/float(10)
for i in range(10):
  start = int(div*i)
  end = start + int(div)
  part = x[start:end]
  part_y = y[start:end]
  rest = x[:start] + x[end:]
  rest_y = y[:start] + y[end:]

  prob = svm_problem(rest_y, rest)
  param = svm_parameter('-t 0 -c 4 -b 1 -h 0') 
  m = svm_train(prob, param)
  p_label_test, p_acc_test, p_val_test = svm_predict(part_y, part, m, options="-b 1")
  test_accuracy += p_acc_test[0]
  #get accuracy per class
  f = open('temp', 'a')
  f.write('iteration: %d\n'%i)
  for ili in range(len(indices_list)):
    index_list = []
    f.write('index list: %d\n'%ili)
    verb_particle_indices = indices_list[ili]
    for index in normalize(verb_particle_indices):
      if index in range(start, end):
        index_list.append(index-start)
    if len(index_list)>0:
      correct = 0
      for i in index_list:
        if p_label_test[i] == part_y[i]:
          correct += 1
      f.write('%d\n'%correct)
  f.close()

test_accuracy = test_accuracy/float(10)
print test_accuracy

