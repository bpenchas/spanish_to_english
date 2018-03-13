import tensorflow as tf
import numpy as np

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
            for tup in self.train_dict.items():
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
        result = 'NO_WORD_YO'
        for word,english_vec in self.english_word2vec.items():
            dist = np.sum((y_hat - english_vec) ** 2)
            if word == 'return':
                print(dist)
                print(word)
            elif word == 'and':
                print(dist)
                print(word)
            if dist < lowest_dist:
                lowest_dist = dist
                result = word
        return result


if __name__ == "__main__":
    model = SpanEngNet()
    model.create_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
        print(model.eval('regresar', sess))
    
    # and -0.031533 0.046278 -0.12534 0.19165 -0.1266 -0.012853 0.10342 -0.0098085 0.15189 0.27582 0.13695 0.0088799 0.14132 -0.12 -0.063439 -0.15178 0.09809 -0.1201 -0.069086 0.014666 -0.023041 0.03043 -0.12664 -0.063282 -0.082246 0.036718 0.22698 -0.096025 -0.011699 0.066158 -0.18542 0.19223 -0.061685 0.27049 0.075116 -0.054928 -0.086027 -0.19387 0.14677 -0.06013 0.068269 0.071613 -0.094414 0.036158 0.002782 -0.081711 -0.013369 -0.053017 0.052227 -0.079682 -0.00031768 0.030397 -0.16847 0.021828 -0.19577 -0.050109 -0.0096879 0.085536 -0.28135 0.17001 -0.049194 -0.16721 0.19018 -0.0474 -0.00036412 0.026316 -0.22135 -0.061583 -0.21854 -0.021669 -0.2963 -0.071949 0.010638 -0.19055 -0.11292 -0.099072 0.19357 0.14115 0.068346 -0.00045947 0.072621 -0.021192 -0.1242 -0.041933 -0.028386 0.049083 -0.073574 0.073525 0.088135 -0.032184 0.029903 -0.070025 0.15323 -0.17236 0.073502 0.13232 0.090191 0.0079023 -0.027887 -0.046971 0.039198 -0.12567 0.19803 -0.075995 -0.21353 0.031964 -0.17346 0.055884 -0.055404 -0.0083924 -0.024104 0.0023894 -0.1057 -0.10604 -0.061323 -0.041473 0.0060497 0.055896 -0.071338 0.1375 0.094781 0.048121 -0.071236 0.26263 0.07257 -0.00020344 0.1864 0.066703 0.055229 0.11258 0.047647 0.085482 -0.14489 0.0098078 0.082585 0.039254 -0.10044 0.16532 -0.030841 0.10315 -0.046584 0.11211 0.15416 -0.050309 0.14853 0.2287 -0.056036 -0.072966 0.0018167 -0.015694 -0.06022 -0.19044 -0.075073 -0.0032815 -0.079256 -0.078324 -0.11073 -0.093705 0.26284 0.01034 -0.095 0.17295 -0.053949 0.15056 0.22815 -0.16589 -0.080074 -0.076248 0.13423 -0.093626 -0.065384 -0.014181 -0.067937 -0.038283 -0.084514 0.11082 0.068804 0.19402 -0.069373 -0.043398 0.15402 -0.10172 0.049785 -0.010005 -0.03371 0.29018 0.025405 -0.094919 0.093876 -0.055423 -0.059419 -0.082542 0.094048 0.059422 -0.032564 -0.0062017 -0.0095274 0.092439 -0.16995 0.00038904 0.19187 -0.025048 -0.11844 0.027879 -0.034024 -0.046866 -0.09009 -0.034417 0.25534 0.096778 0.20841 0.029693 -0.015943 -0.035779 0.0021559 0.080246 -0.031355 -0.22676 -0.11579 -0.059579 -0.07442 -0.12871 -0.10199 0.064969 -0.070388 -0.040131 -0.1474 -0.098839 0.11614 0.15871 0.0693 0.031897 -0.028738 -0.084634 -0.14864 0.11398 0.072688 -0.065752 -0.013296 0.085164 0.025053 0.016867 -0.045257 -0.042925 0.12329 0.13012 -0.01532 -0.13943 0.089764 0.082172 -0.081918 -0.011688 -0.11742 0.029242 -0.065814 0.029959 -0.010941 -0.0183 0.05718 0.068436 0.0072271 0.0057584 0.071466 -0.083164 -0.01501 -0.07806 0.0033293 0.099132 0.061188 -0.097815 -0.14008 -0.0026304 0.0022269 0.083496 -0.14334 -0.037447 0.061564 0.21536 -0.036836 0.038629 0.13031 0.045944 0.027701 0.061679 0.062921 0.068453 -0.026292 0.17342 -0.14421 -0.013124 0.15494 -0.10786 0.18314 0.13881 0.02757 -0.035073 -0.017829 0.11163 -0.058231 0.011977 





