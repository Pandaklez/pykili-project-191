from sklearn.metrics.pairwise import euclidean_distances
from string import punctuation
from keras_preprocessing import text, sequence
import keras.utils as np_utils
import keras.backend as K
from keras.models import  Sequential
from keras.layers import Embedding, Dense, Lambda
import pandas as pd
import csv
import os
import re
from  pymorphy2 import MorphAnalyzer


def collect_files(path):
    text = ''
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            with open(os.path.join(path, file), encoding='utf-8') as f:
                text += f.read()
    return text


def parse_file(utext):
    text = re.sub('[^а-яА-Я\-]', ' ', utext).lower().split()
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
        if word not in stopwords and word != '':
            res.append(word)
    return res


def create_lemmas(text):
    morph = MorphAnalyzer()
    for i in range(len(text)):
        text[i] = morph.parse(text[i])[0].normal_form
    return text


def build_vocab(texts):
    tokenizer = text.Tokenizer(filters='')
    tokenizer.fit_on_texts(texts)
    word2ind = tokenizer.word_index
    word2ind['PAD'] = 0
    print(word2ind)
    print(len(word2ind))
    ind2word = {v:k for (k, v) in word2ind.items()}
    textinds = [word2ind[word] for word in texts]
    return word2ind, ind2word, textinds


def cbow_pairs_generator(textinds, window_size, vocab_size):
    for ind, word in enumerate(textinds):
        target_word = [word]
        context_words = []
        context_words.append([textinds[i] for i in range(ind - window_size, ind + window_size + 1) if 0 <= i < len(textinds) and i != ind])
        x = sequence.pad_sequences(context_words, maxlen=window_size*2)
        y = np_utils.to_categorical(target_word, vocab_size)
        yield x, y


def preprocessing(start_path):
    utext = collect_files(start_path)
    text = parse_file(utext)
    stopwords = create_stopwords()
    text = remove_stopwords(text, stopwords)
    text = create_lemmas(text)
    word2ind, ind2word, textinds = build_vocab(text)
    return word2ind, ind2word, textinds


def build_network(vocab_size, embed_size, window_size):
    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size * 2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return cbow


def train(cbow, textinds, window_size, vocab_size):
    for epoch in range(1, 6):
        loss = 0
        i = 0
        for x, y in cbow_pairs_generator(textinds=textinds, window_size=window_size, vocab_size=vocab_size):
            i += 1
            loss += cbow.train_on_batch(x, y)
            if i % 100000 == 0:
                print('Processed {} (context, word) pairs'.format(i))
        print('Epoch:', epoch, '\tloss:', loss)
    return cbow


def get_cbow_weights(cbow):
    return cbow.get_weights()[0][1:]


def show_results(weights, ind2word, word2ind):
    distances = euclidean_distances(weights)
    similar = {target: [ind2word[idx] for idx in distances[word2ind[target] - 1].argsort()[1:6] + 1] for target in ['муму', 'герасим']}
    return similar


def save_weights(weights, ind2word):
    file = pd.DataFrame(weights, index=list(ind2word.values())[:-1])
    print(file)
    file.to_csv(r'C:\Users\raski\.PyCharmCE2019.2\config\scratches\weights.csv', encoding='utf-8-sig')


def export_texts(word2ind):
    res = []
    for word, ind in word2ind.items():
        res.append(word)
        res.append(ind)
    with open('word2ind.txt', 'w', encoding='utf-8') as f:
        f.write('\t'.join(map(str, res)))


def main():
    file_path = r'C:\Users\raski\.PyCharmCE2019.2\config\scratches\samples'
    embed_size = 100
    window_size = 2
    word2ind, ind2word, textinds = preprocessing(file_path)
    export_texts(word2ind)
    cbow = build_network(vocab_size=len(word2ind), embed_size=embed_size, window_size=window_size)
    trained = train(cbow=cbow, textinds=textinds, window_size=window_size, vocab_size=len(word2ind))
    weights = get_cbow_weights(trained)
    print(weights)
    save_weights(weights, ind2word)
    print(pd.DataFrame(weights, index=list(ind2word.values())[1:]).head())
    print(show_results(weights=weights, word2ind=word2ind, ind2word=ind2word))


if __name__ == '__main__':
    main()