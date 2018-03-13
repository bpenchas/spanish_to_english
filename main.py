import tensorflow as tf
import numpy as np
import random

def make_train_dict():
    train_dict = {}
    with open("train.vocab") as f:
        for line in f:
            (key, val) = line.split(',')
            train_dict[key] = val
    return train_dict

def make_word2vec_dict(file_name):
    d = {}
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split(' ', 1)
            vec = val.split(' ')
            vec.pop(-1)
            num_list = [float(i) for i in vec]
            d[key] = np.reshape(num_list, (len(num_list), 1))
    return d

class SpanEngNet(object):
    def __init__(self):
        self.train_dict = make_train_dict()
        self.english_word2vec = make_word2vec_dict('train.en.wv')
        self.spanish_word2vec = make_word2vec_dict('train.es.wv')
        self.eval_spanish_word2vec = make_word2vec_dict('eval.es.wv')
        self.search_word2vec = make_word2vec_dict('search.en.wv')
        self.vec_shape = self.spanish_word2vec['mano'].shape
        # print(self.vec_shape)
        # vec_shape = 300
        self.input_spanish_vec = tf.placeholder(tf.float32, shape=self.vec_shape)
        self.correct_english_vec = tf.placeholder(tf.float32, shape=self.vec_shape)
        
        self.batch_size = 64
        self.lr = 0.001
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def create_model(self):
        #spanish_vecs = tf.layers.Input(shape=self.vec_shape, dtype=tf.float32)
        self.logits = tf.layers.dense(inputs=self.input_spanish_vec, units=1, activation=None)
        self.loss = tf.reduce_sum(tf.square(self.logits - self.correct_english_vec))        
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
    
    def train(self, sess):
        for epoch in range(5):
            loss_sum = 0
            items = self.train_dict.items()
            random.shuffle(items)
            for tup in items:
                spanish_vec = self.spanish_word2vec[tup[0].replace('\n', '')]
                english_vec = self.english_word2vec[tup[1].replace('\n', '')]

                loss, _ = sess.run([self.loss, self.opt], feed_dict={self.input_spanish_vec: spanish_vec, self.correct_english_vec: english_vec})
                loss_sum += loss
            print(loss_sum)
            

    def eval(self, spanish_word, sess):
        print(self.eval_spanish_word2vec[spanish_word].shape)
        y_hat = sess.run(self.logits, feed_dict={self.input_spanish_vec: self.eval_spanish_word2vec[spanish_word]})
        print(y_hat.shape)
        #lowest_dist = tf.convert_to_tensor(1000000000.0, dtype=tf.float64)
        lowest_dist = 1000000000.0
        max_dist = 0
        result = 'NO_WORD_YO'
        for word,english_vec in self.search_word2vec.items():
            dist = np.sum((y_hat - english_vec) ** 2)
            # if max_dist < dist:
            #     max_dist = dist
            #     result = word
            # if word == 'return':
            #     print(dist)
            #     print(word)
            # elif word == 'and':
            #     print(dist)
            #     print(word)
            if dist < lowest_dist and word != 'and':
                lowest_dist = dist
                result = word
        return result


if __name__ == "__main__":
    
    with tf.Session() as sess:
        model = SpanEngNet()
        model.create_model()
        sess.run(tf.global_variables_initializer())
        model.train(sess)
        print(model.eval('regresar', sess))


