from __future__ import print_function
import os
import re
import csv
import json
import nltk
import string
import argparse
import numpy as np
from operator import itemgetter
# from unidecode import unidecode
from collections import defaultdict

path_this = os.path.dirname(os.path.abspath(__file__))
path_corpus = os.path.abspath(os.path.join(path_this, 'corpus'))
path_embedding = os.path.abspath(os.path.join(path_this, 'embedding'))
path_model = os.path.abspath(os.path.join(path_this, 'model'))


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                yield line.split()


class Helper(object):
    def __init__(self, lang='id'):
        self.topic = {'advertisement': 0, 'animals' : 1, 'automotive': 2,
                    'business and economy': 3, 'careers': 4, 'conflict and attack': 5,
                    'disaster and accident': 6, 'education': 7,
                    'environment': 8, 'hate speech': 9,
                    'hobbies and interests': 10, 'international relation': 11,
                    'porn': 12, 'real estate': 13,
                    'religion and spirituality': 14, 'weather and climate': 15}
        self.lang = lang
        self.text_size = 150
        self.input_size = 150

        self.word2idx_contents = None
        self.idx2word_contents = None
        self.unknown_word_idx_contents = None
        self.nb_words_contents = None

        self.char2idx = self.load_chars()
        self.unknown_char_idx = len(self.char2idx) + 1
        self.nb_chars = len(self.char2idx)

    def decode_topic(self, topic):
        for dname, did in self.topic.items():
            if did == topic:
                return dname

    def remove_url(self, text):
        def regex_or(*items):
            r = '|'.join(items)
            r = '(' + r + ')'
            return r

        def pos_lookahead(r):
            return '(?=' + r + ')'

        def optional(r):
            return '(%s)?' % r

        punct_chars = r'''['".?!,:;]'''
        entity = '&(amp|lt|gt|quot);'

        url_start1 = regex_or('https?://', r'www\.')
        common_TLDs = regex_or('com', 'co\\.uk', 'org', 'net', 'info', 'ca')
        url_start2 = r'[a-z0-9\.-]+?' + r'\.' + common_TLDs + pos_lookahead(r'[/ \W\b]')
        url_body = r'[^ \t\r\n<>]*?'
        url_extra_crap_before_end = '%s+?' % regex_or(punct_chars, entity)
        url_end = regex_or(r'\.\.+', r'[<>]', r'\s', '$')
        url = (r'\b' +
               regex_or(url_start1, url_start2) +
               url_body +
               pos_lookahead(optional(url_extra_crap_before_end) + url_end))

        url_re = re.compile('(%s)' % url, re.U | re.I)
        text = re.sub(url_re, '', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def preprocessing(self, text):
        # text = unidecode(text)
        text = str(text)
        is_retweet = re.findall(r"^rt\b ?@([A-Za-z0-9_]+)", text, re.IGNORECASE)
        if is_retweet:
            return None
        else:
            try:
                tweet_with_no_link = ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())

                if len(tweet_with_no_link) > 0:
                    # Tweet telah bersih dari URL/link, cek keberadaan username
                    tweet_wihtout_username = ' '.join(re.sub(r"@[A-Za-z0-9_]+", " ", tweet_with_no_link).split())

                    if len(tweet_wihtout_username) > 0:
                        # Tweet telah bersih dari URL/link dan username, kemudian cek hashtag
                        tweet_with_no_hashtag = ' '.join(re.sub(r"#(\w+)", " ", tweet_wihtout_username).split())

                        if len(tweet_with_no_hashtag) > 0:
                            # Tweet telah bersih dari URL/link dan username, dan juga hashtag. Kemudian cek retweet (RT)
                            tweet_with_no_rt = ''.join(re.sub(r"(RT : ?|via : |Cc : )", "", tweet_with_no_hashtag))

                            if len(tweet_with_no_rt) > 0:
                                if len(tweet_with_no_rt.split()) > 1:
                                    clean_tweet = re.sub(' +', ' ', tweet_with_no_rt)
                text = clean_tweet
            except:
                pass
            text = ''.join(list(filter(lambda x: x in string.printable, text)))
            text = re.sub(r"\n", ' ', text)
            text = re.sub(r"\s+", ' ', text)
            text = self.remove_url(text)
            text = re.sub(r"\s+", ' ', text)
            text = re.sub('<\w+.*>:', "", text)
            text = re.sub('@\w+', "", text)
            text = re.sub('https?:\w+.*', "", text)
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"pic.twitter.com\S+", "", text)
            text = re.sub(r'\[pic\]', '', text)
            text = re.sub(r"RT\s\S+.", "", text)
            text = re.sub(r"@\S+", "", text)
            text = re.sub(r'\.', ' ', text)
            return text.strip()

    def load_chars(self):
        with open(os.path.join(path_embedding, 'chars.json'), 'r') as f:
            return json.loads(f.read())

    def load_vocab(self, nb_words=None):
        with open(os.path.join(path_embedding, self.lang, 'vocab.json'), 'r') as f:
            data = json.loads(f.read())
        sorted_word2idx = sorted(data.items(), key=itemgetter(1))
        if nb_words is not None:
            sorted_word2idx = sorted_word2idx[:nb_words]
        word2idx = dict(((k, v) for (k, v) in sorted_word2idx))
        idx2word = dict(((v, k) for (k, v) in word2idx.items()))
        return word2idx, idx2word

    def load_weights(self):
        with open(os.path.join(path_embedding, self.lang, 'weight.npy'), 'rb') as f:
            words_weights = np.load(f)
        return words_weights

    def set_vocab_contents(self, nb_words=None):
        self.nb_words_contents = nb_words
        if self.nb_words_contents is None:
            raise ValueError('There is no vocab!')
        self.word2idx_contents, self.idx2word_contents = self.load_vocab(nb_words=self.nb_words_contents)
        self.unknown_word_idx_contents = len(self.word2idx_contents) + 1

    def load_text_and_label(self):
        corpus = defaultdict(list)
        with open(os.path.join(path_corpus, self.lang + '.csv'), newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                corpus[row[1].lower()].append(row[0].lower())
            csvfile.close()
        return corpus

    def split_train_and_test_corpus(self, corpus, rasio=0.75):
        train_contents = []
        train_char_contents = []
        train_label = []
        test_contents = []
        test_char_content = []
        test_label = []
        for topic in corpus.keys():
            split_train = int(len(corpus[topic]) * rasio)

            all_contents = [_c for _c in corpus[topic]]

            train_contents.extend(all_contents[:split_train])
            train_char_contents.extend([list(_) for _ in all_contents[:split_train]])

            test_contents.extend(all_contents[split_train:])
            test_char_content.extend([list(_) for _ in all_contents[split_train:]])

            train_label.extend([self.topic[topic] for _ in range(len(all_contents[:split_train]))])
            test_label.extend([self.topic[topic] for _ in range(len(all_contents[split_train:]))])

        return {
            'train_contents': train_contents,
            'train_char_contents': train_char_contents,
            'train_label': train_label,
            'test_contents': test_contents,
            'test_char_contents': test_char_content,
            'test_label': test_label
        }

    def text_to_sequences(self, contents, chars_contents):
        sequences_contents = []
        sequences_char_contents = []

        for line in contents:
            tmp_word = []
            for word in nltk.word_tokenize(line):
                idx = self.word2idx_contents.get(word)
                if idx is not None:
                    tmp_word.append(self.word2idx_contents[word])
                else:
                    tmp_word.append(self.unknown_word_idx_contents)

            tmp_word = tmp_word[:self.input_size]
            sequences_contents.append(tmp_word)

        for line in chars_contents:
            tmp_char = []
            for c in line:
                idx = self.char2idx.get(c)
                if idx is not None:
                    tmp_char.append(self.char2idx[c])
                else:
                    tmp_char.append(self.unknown_char_idx)

            tmp_char = tmp_char[:self.input_size]
            sequences_char_contents.append(tmp_char)

        return sequences_contents, sequences_char_contents

    def load_dataset(self, nb_words):
        # check if word2idx is not None
        if self.word2idx_contents is None:
            self.set_vocab_contents(nb_words=nb_words)

        corpus = self.load_text_and_label()
        split_result = self.split_train_and_test_corpus(corpus)

        X_train, CX_train = self.text_to_sequences(split_result['train_contents'], split_result['train_char_contents'])
        X_test, CX_test = self.text_to_sequences(split_result['test_contents'], split_result['test_char_contents'])

        x_train = X_train
        cx_train = CX_train
        y_train = split_result['train_label']
        x_test = X_test
        cx_test = CX_test
        y_test = split_result['test_label']

        return (x_train, cx_train, y_train), (x_test, cx_test, y_test)


def word2vec(lang='id'):
    import sys
    import logging
    from gensim.models import Word2Vec

    if not os.path.exists(os.path.join(path_embedding, lang)):
        os.makedirs(os.path.join(path_embedding, lang))
    if not os.path.exists(os.path.join(path_model, 'w2v', lang)):
        os.makedirs(os.path.join(path_model, 'w2v', lang))

    train_dir = os.path.join(path_corpus, 'txt', lang + '.txt')
    weights_path = os.path.join(path_embedding, lang, 'weight.npy')
    vocab_path = os.path.join(path_embedding, lang, 'vocab.json')
    model_path = os.path.join(path_model, 'w2v', lang, 'model.h5')

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    corpus = Sentences(train_dir)

    model = Word2Vec(corpus,
                     size=500,
                     window=5,
                     min_count=20,
                     workers=10)

    weights = model.wv.syn0
    np.save(weights_path, weights)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

    model.save(model_path)


def extract_corpus(bhs='id'):
    from openpyxl import load_workbook

    language = bhs
    contents = []
    final_corpus_pc = []

    path_xlsx = os.path.join(path_corpus, 'xlsx')

    print("creating corpus {} ...".format(language))

    # load file txt corpus language_detector
    read_txt = open(os.path.join(path_xlsx, language + '.txt'), 'r').read().splitlines()
    read_txt = [_.strip() for _ in read_txt if _.strip()]
    for from_txt in read_txt:
        prepro_txt = obj.preprocessing(from_txt)
        if prepro_txt != None:
            contents.append(prepro_txt.lower())

    print("read xlsx file ...")
    list_file_corpus = [_ for _ in os.listdir(path_xlsx) if _.endswith('.xlsx')]

    # load file corpus
    for file_pc in list_file_corpus:
        _workbook = load_workbook(os.path.join(path_xlsx, file_pc))
        _sheet = _workbook.get_active_sheet()
        for _i_, row in enumerate(_sheet.rows):
            if _i_ != 0:
                if row[ord("A") - 65].value:
                    con = row[ord("A") - 65].value
                    top = row[ord("B") - 65].value
                    contents.append(con)
                    final_corpus_pc.append((con, top))
        _workbook.close()

    final_corpus = list(set(final_corpus_pc))
    contents = list(set(contents))

    print("create csv file ...")
    # create corpus .csv
    if not os.path.exists(os.path.join(path_corpus)):
        os.makedirs(os.path.join(path_corpus))
    with open(os.path.join(path_corpus, language + '.csv'), 'w', newline='') as _filecsv:
        spamwriter = csv.writer(_filecsv)
        for c in final_corpus:
            content = c[0]
            tops = str(c[1])
            content = obj.preprocessing(content)
            if content != None:
                spamwriter.writerow([content, tops.lower()])
        _filecsv.close()
        print("corpus csv has been created.")

    if not os.path.exists(os.path.join(path_corpus, 'txt')):
        os.makedirs(os.path.join(path_corpus, 'txt'))

    # create file txt contents for w2v
    with open(os.path.join(path_corpus, 'txt', language + '.txt'), 'w') as _file:
        for con in contents:
            if con != None:
                _file.write("{}\n".format(obj.preprocessing(con)))
        _file.close()
        print("file contents for w2v has been created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-extract', default='w2v', help='w2v or corpus')
    parser.add_argument('-lang', default='id', help='Specify Language!')
    args = vars(parser.parse_args())

    lang = args['lang']
    extract = args['extract']

    obj = Helper(lang=lang)

    # build w2v
    if extract == 'w2v':
        word2vec(lang)
    else:
        # create corpus csv, issue.txt, content.txt
        extract_corpus(lang)