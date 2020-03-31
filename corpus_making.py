from collections import defaultdict
from string import punctuation
#from pymorphy2 import analyzer


def parse_file(file):
    with open(file, encoding='utf-8') as f:
        text = f.read().lower().split()
    for i in range(len(text)):
        text[i] = text[i].strip(punctuation)
    return text


def create_stopwords():
    with open('stopwords.txt', encoding='utf-8') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip('\n')
    return stopwords


def remove_stopwords(words, stopwords):
    res = []
    for word in words:
        if word not in stopwords:
            res.append(word)
    return res


def create_vocabulary(words):
    dict1 = defaultdict(int)
    for word in words:
        dict1[word] += 1
    items = list(dict1.items())
    vocab = {items[i][0]:i for i in range(len(items))}
    return vocab


def cbow_pairs_generator(text, window_size):
    pairs = []
    for i in range(window_size,len(text) - window_size):
        pairs.append(([text[j] for j in range(i - window_size, i + window_size + 1) if j != i], text[i]))
    return pairs


def preprocessing(file):
    text = parse_file(file)
    stopwords = create_stopwords()
    text = remove_stopwords(text, stopwords)
    vocab = create_vocabulary(text)
    return vocab, text


def main():
    file = 'sample.txt'
    vocab, text = preprocessing(file)
    pairs = cbow_pairs_generator(text, 2)
    print(pairs)


if __name__ == '__main__':
    main()