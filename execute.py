import csv
import os
from pymorphy2 import MorphAnalyzer
from sklearn.metrics.pairwise import euclidean_distances


def show_results(data, request):
    morph = MorphAnalyzer()
    dist_data = {k:v for (k,v) in data.items()}
    for label in dist_data.keys():
        dist_data[label].append(euclidean_distances(dist_data[label][0]))
    for word in request:
        print('word ' + word + ' is similar to:')
        for label, things in dist_data.items():
            if label in ['full', 'nonstop']:
                if word not in things[1].keys():
                    print('Word %s not in text' % word)
                    continue
                print(' '.join([things[2][idx] for idx in things[3][things[1][word] - 1].argsort()[1:6] + 1]) + ' in ' + label)
            else:
                word = morph.parse(word)[0].normal_form
                if word not in things[1].keys():
                    print('Word %s not in text' % word)
                    continue
                print(' '.join([things[2][idx] for idx in things[3][things[1][word] - 1].argsort()[1:6] + 1]) + ' in ' + label)

for root, dirs, files in os.walk(r'.\processed'):
    labels = {os.path.splitext(f)[0].split('_')[1]:['.csv', '.txt', '.txt'] for f in files}
data = {k:v for (k,v) in labels.items()}
for label in labels.keys():
    with open(os.path.join(root, 'weights_' + label + '.csv'), encoding='utf-8-sig') as f:
        weights = []
        dic = csv.DictReader(f)
        for row in dic:
            word = []
            for i in range(100):
                word.append(float(row[str(i)]))
            weights.append(word)
        data[label][0] = weights
    with open(os.path.join(root, 'word2ind_' + label + '.txt'), encoding='utf-8-sig') as f:
        words = f.read().split('\t')
    word2ind = {words[i]: int(words[i + 1]) for i in range(0, len(words), 2)}
    data[label][1] = word2ind
    data[label][2] = {v:k for (k, v) in word2ind.items()}
request = input('What words you are interested in?\n').split()
show_results(data, request)
