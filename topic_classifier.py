from __future__ import print_function
import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model, model_from_json
from keras.layers import Embedding, LSTM, Dense, Input, Dropout, concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import precision, recall, f_measure, accuracy

path_this = os.path.dirname(os.path.abspath(__file__))
path_root = os.path.abspath(os.path.join(path_this, '..'))
path_model = os.path.abspath(os.path.join(path_this, 'model'))
path_logs = os.path.abspath(os.path.join(path_this, 'logs'))
sys.path.append(path_root)

from classifier16.helper import Helper


class TopicClassifier(object):
    def __init__(self, lang='id', mode='test', evaluate=False, **kwargs):
        self.kwargs = kwargs
        self.lang = lang
        self.epochs = self.kwargs.get('epochs', 30)
        self.patience = self.kwargs.get('patience', 10)
        self.monitor = self.kwargs.get('monitor', 'val_loss')
        self.callback = self.kwargs.get('callback', 'earlystopping')
        self.max_features = 30000
        self.word_contents_embedding_size = 500
        self.char_embedding_size = 60
        self.words_maxlen = 150
        self.output_dim = 500
        self.batch_size = 256
        self.filename_model = 'topic_model.json'
        self.filename_weight = 'topic_weight.h5'
        self.helper = Helper(lang=self.lang)
        if mode == 'train':
            self.train()
        if evaluate:
            self.evaluate()

        try:
            self.model = self.load_model()
            if self.helper.word2idx_contents is None:
                self.helper.set_vocab_contents(nb_words=self.max_features)
        except Exception:
            raise ValueError('model was not found!')

    def load_model(self):
        with open(os.path.join(path_model, self.lang, self.filename_model), 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(path_model, self.lang, self.filename_weight))
        return model

    def classify(self, text):
        text = self.helper.preprocessing(text)
        X_test, CX_test = self.helper.text_to_sequences([text], [list(text)])
        x_test = np.array(X_test)
        cx_test = np.array(CX_test)
        x_test = pad_sequences(x_test, maxlen=self.words_maxlen, padding='pre')
        cx_test = pad_sequences(cx_test, maxlen=self.words_maxlen, padding='pre')
        result = np.argmax(self.model.predict_on_batch([x_test, cx_test]))
        predicted = self.helper.decode_topic(result)
        return predicted, text

    def evaluate(self):
        datenow = datetime.now()
        if not os.path.exists(os.path.join(path_logs, self.lang)):
            os.makedirs(os.path.join(path_logs, self.lang))
        with open(os.path.join(path_logs, self.lang, 'eval_' + datenow.strftime("%Y%m%d%M%S") + '.txt'), 'w') as _filetxt:
            with open(os.path.join(path_model, self.lang, self.filename_model), 'r') as f:
                loaded_model_json = f.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(path_model, self.lang, self.filename_weight))

            (x_train, cx_train, y_train), (x_test, cx_test, y_test) = self.helper.load_dataset(nb_words=self.max_features)

            x_train = np.array(x_train)
            cx_train = np.array(cx_train)
            x_test = np.array(x_test)
            cx_test = np.array(cx_test)

            x_train = pad_sequences(x_train, maxlen=self.words_maxlen, padding='pre')
            cx_train = pad_sequences(cx_train, maxlen=self.words_maxlen, padding='pre')
            x_test = pad_sequences(x_test, maxlen=self.words_maxlen, padding='pre')
            cx_test = pad_sequences(cx_test, maxlen=self.words_maxlen, padding='pre')

            y_train = to_categorical(np.asarray(y_train))
            y_test = to_categorical(np.asarray(y_test))

            print('-' * 50)
            print('TRAIN SET EVALUATION')
            print('-' * 50)
            _filetxt.write("{}\n".format('-' * 50))
            _filetxt.write("{}\n".format('TRAIN SET EVALUATION'))
            _filetxt.write("{}\n".format('-' * 50))

            predictions = model.predict([x_train, cx_train], batch_size=self.batch_size)
            predictions = [self.helper.decode_topic(np.argmax(p)) for p in predictions]
            reference = [self.helper.decode_topic(np.argmax(y)) for y in y_train]

            cm = ConfusionMatrix(reference, predictions)
            print(cm.pretty_format(sort_by_count=True))
            print('Accuracy: {}\n'.format(accuracy(reference, predictions)))
            _filetxt.write("{}\n".format(cm.pretty_format(sort_by_count=True)))
            _filetxt.write("{}\n".format('Accuracy: {}\n'.format(accuracy(reference, predictions))))
            refsets = defaultdict(set)
            testsets = defaultdict(set)
            for i, (y, y_) in enumerate(zip(reference, predictions)):
                refsets[y].add(i)
                testsets[y_].add(i)
            for label in self.helper.topic.keys():
                print('{} precision: {}'.format(label, precision(refsets[label], testsets[label])))
                print('{} recall: {}'.format(label, recall(refsets[label], testsets[label])))
                print('{} F-measure: {}\n'.format(label, f_measure(refsets[label], testsets[label])))
                _filetxt.write("{}\n".format('{} precision: {}'.format(label, precision(refsets[label], testsets[label]))))
                _filetxt.write("{}\n".format('{} recall: {}'.format(label, recall(refsets[label], testsets[label]))))
                _filetxt.write("{}\n".format('{} F-measure: {}\n'.format(label, f_measure(refsets[label], testsets[label]))))

            print('-' * 50)
            print('TEST SET EVALUATION')
            print('-' * 50)
            _filetxt.write("{}\n".format('-' * 50))
            _filetxt.write("{}\n".format('TEST SET EVALUATION'))
            _filetxt.write("{}\n".format('-' * 50))

            predictions = model.predict([x_test, cx_test], batch_size=self.batch_size)
            predictions = [self.helper.decode_topic(np.argmax(p)) for p in predictions]
            reference = [self.helper.decode_topic(np.argmax(y)) for y in y_test]
            cm = ConfusionMatrix(reference, predictions)
            print(cm.pretty_format(sort_by_count=True))
            print('Accuracy: {}\n'.format(accuracy(reference, predictions)))
            _filetxt.write("{}\n".format(cm.pretty_format(sort_by_count=True)))
            _filetxt.write("{}\n".format('Accuracy: {}\n'.format(accuracy(reference, predictions))))
            refsets = defaultdict(set)
            testsets = defaultdict(set)
            for i, (y, y_) in enumerate(zip(reference, predictions)):
                refsets[y].add(i)
                testsets[y_].add(i)
            for label in self.helper.topic.keys():
                print('{} precision: {}'.format(label, precision(refsets[label], testsets[label])))
                print('{} recall: {}'.format(label, recall(refsets[label], testsets[label])))
                print('{} F-measure: {}\n'.format(label, f_measure(refsets[label], testsets[label])))
                _filetxt.write("{}\n".format('{} precision: {}'.format(label, precision(refsets[label], testsets[label]))))
                _filetxt.write("{}\n".format('{} recall: {}'.format(label, recall(refsets[label], testsets[label]))))
                _filetxt.write("{}\n".format('{} F-measure: {}\n'.format(label, f_measure(refsets[label], testsets[label]))))
            _filetxt.close()

    def train(self):
        (x_train, cx_train, y_train), (x_test, cx_test, y_test) = self.helper.load_dataset(nb_words=self.max_features)

        x_train = np.array(x_train)
        cx_train = np.array(cx_train)
        x_test = np.array(x_test)
        cx_test = np.array(cx_test)

        x_train = pad_sequences(x_train, maxlen=self.words_maxlen, padding='pre')
        cx_train = pad_sequences(cx_train, maxlen=self.words_maxlen, padding='pre')
        x_test = pad_sequences(x_test, maxlen=self.words_maxlen, padding='pre')
        cx_test = pad_sequences(cx_test, maxlen=self.words_maxlen, padding='pre')

        y_train = to_categorical(np.asarray(y_train))
        y_test = to_categorical(np.asarray(y_test))

        print('x_train shape:', x_train.shape)
        print('cx_train shape:', cx_train.shape)
        print('y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape)
        print('cx_test shape:', cx_test.shape)
        print('y_test shape:', y_test.shape)
        print()
        # exit()
        print('Build the model...')

        # for weights embedding contents
        word2idx_contents, idx2word_contents = self.helper.load_vocab(self.max_features)
        words_weights_contents = self.helper.load_weights()
        n_vocabs_contents = len(word2idx_contents) + 3
        word_vectors = {}
        for word, idx in word2idx_contents.items():
            word_vectors[word] = words_weights_contents[idx]
        word_embedding_weights_contents = np.zeros((n_vocabs_contents + 2, self.word_contents_embedding_size))
        for word, idx in word2idx_contents.items():
            idx = idx + 3
            word_embedding_weights_contents[idx, :] = word_vectors[word]

        chars2idx = self.helper.load_chars()
        n_chars = len(chars2idx) + 3

        text_contents_input = Input(shape=(self.words_maxlen,))
        x2 = Embedding(input_dim=n_vocabs_contents + 2,
                       output_dim=self.word_contents_embedding_size,
                       weights=[word_embedding_weights_contents],
                       input_length=self.words_maxlen,
                       name='embed_word_contents')(text_contents_input)
        x2 = Dropout(0.25)(x2)
        x2 = Conv1D(64, 5,
                    padding='valid',
                    activation='relu',
                    strides=1)(x2)
        x2 = MaxPooling1D(pool_size=4)(x2)
        x2 = LSTM(250, return_sequences=True)(x2)
        x2 = Dense(300)(x2)
        x2 = Dropout(0.5)(x2)
        x2 = Dense(100)(x2)
        x2 = Dropout(0.5)(x2)
        text_contents_output = Dropout(0.2)(x2)

        chars_contents_input = Input(shape=(self.words_maxlen,))
        x4 = Embedding(input_dim=n_chars + 2,
                       output_dim=self.char_embedding_size,
                       input_length=self.words_maxlen,
                       name='embed_char_contents')(chars_contents_input)
        x4 = Dropout(0.25)(x4)
        x4 = Conv1D(64, 5,
                    padding='valid',
                    activation='relu',
                    strides=1)(x4)
        x4 = MaxPooling1D(pool_size=4)(x4)
        x4 = LSTM(250, return_sequences=True)(x4)
        x4 = Dense(300)(x4)
        x4 = Dropout(0.5)(x4)
        x4 = Dense(100)(x4)
        x4 = Dropout(0.5)(x4)
        chars_contents_output = Dropout(0.2)(x4)

        concatenated_input_contents = concatenate([text_contents_output, chars_contents_output])
        concatenated_input_contents = LSTM(250)(concatenated_input_contents)
        concatenated_input_contents = Dense(300)(concatenated_input_contents)
        concatenated_input_contents = Dropout(0.5)(concatenated_input_contents)
        concatenated_input_contents = Dense(100)(concatenated_input_contents)
        concatenated_input_contents = Dropout(0.2)(concatenated_input_contents)

        main_output = Dense(len(self.helper.topic), activation='softmax')(concatenated_input_contents)

        model = Model([text_contents_input, chars_contents_input], main_output)
        model.summary()
        model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=model_optimizer,
                      metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor=self.monitor,
                                       patience=self.patience,
                                       verbose=1,
                                       mode='auto')
        filepath_weight = os.path.join(path_model, self.lang, self.filename_weight)
        model_checkpoint = ModelCheckpoint(filepath=filepath_weight,
                                           monitor=self.monitor,
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)
        if self.callback == "earlystopping":
            use_callback = early_stopping
        else:
            use_callback = model_checkpoint

        print('Train the model...')
        model.fit([x_train, cx_train], y_train,
                        epochs=self.epochs,
                        validation_split=0.10,
                        batch_size=self.batch_size,
                        callbacks=[use_callback],
                        verbose=1
                  )

        print('Serialize the model...')
        # serialize model architecture to JSON
        model_json = model.to_json()
        if not os.path.exists(os.path.join(path_model, self.lang)):
            os.makedirs(os.path.join(path_model, self.lang))
        with open(os.path.join(path_model, self.lang, self.filename_model), 'w') as json_file:
            json_file.write(model_json)

        if self.callback == "earlystopping":
            # serialize model weights to HDF5
            model.save_weights(filepath_weight)

        print('Test the model...')
        loss, acc = model.evaluate([x_test, cx_test], y_test,
                                   batch_size=self.batch_size)
        print()
        print('Test loss:', loss)
        print('Test accuracy:', acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang', default='id', help='Specify Language!')
    parser.add_argument('-mode', default='test', help='train or test')
    parser.add_argument('-epochs', default=50, help='how many epochs')
    parser.add_argument('-callback', default='earlystopping', help='Specify Callback (earlystopping or modelcheckpoint')
    parser.add_argument('-patience', default=10, help='how many patience for earlystopping')
    parser.add_argument('-monitor', default='val_loss', help='how use monitor for earlystopping or modelcheckpoint (val_loss or val_acc)')
    parser.add_argument('-text', default=None, help='text for single test')
    args = vars(parser.parse_args())

    lang = args['lang']
    mode = args['mode']
    epochs = args['epochs']
    callback = args['callback']
    patience = args['patience']
    monitor = args['monitor']
    eval = True if mode == 'train' else False
    text = args['text']

    tc = TopicClassifier(lang=lang, mode=mode, evaluate=eval, epochs=epochs, callback=callback, patience=patience, monitor=monitor)

    texts_id = [
        "Nilai saham PT eBdesk hari ini turun 3 persen.",
        "Cuaca pagi ini sangat cerah . Awan tidak terlihat mendung",
        "Pekerjaan diposisi sebagai data scientist sangat diminati karena prospek karir yang baik kedepannya",
        "Team badminton Indonesia melaju ke babak final Asian Games 2018",
        "Penyanyi dangdut kondang tersebut melantunkan lagu dengan indah",
        "Rumah di Jakarta sangat mahal untuk dijangkau",
        "Kajian rutin masjid Istiqlal selalu ramai dan dipenuhi orang",
        "Prabowo mengalahkan Jokowi di pemilihan umum presiden 2019",
        "Oppo merilis handphone baru nya oppo f9 pada 23 Agustus mendatang",
        "Pelaku pembunuhan sudah ditangkap polisi"
    ]

    texts_my = [

    ]
    texts_en = [

    ]
    if lang == 'en':
        texts = texts_en
    elif lang == 'my':
        texts = texts_my
    else:
        texts = texts_id

    if text is not None:
        topic, clean_text = tc.classify(text)
        print("{} | {}".format(topic, clean_text))
    else:
        print("Examples:")
        for text in texts:
            topic, clean_text = tc.classify(text)
            print("{} | {}".format(topic, clean_text))