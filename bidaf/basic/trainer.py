import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients
from my.nltk_utils import span_f1
from squad.utils import get_phrase, get_best_span
from basic.evaluator import F1Evaluator 
from collections import defaultdict 
import numpy as np

class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        _, ds = batch
        feed_dict = self.model.get_feed_dict(ds, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        yps = []
        yp2s = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                yps.append(model.yp)
                yp2s.append(model.yp2)
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
                losses.append(loss)
                grads_list.append(grads)

        self.yps = yps
        self.yp2s = yp2s
        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_scores(self, sess, batches, get_summary=False, k=10):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        yis = []
        for batch, model in zip(batches, self.models):
            _, ds = batch
            paragraph_pointers = ds.data['*p'][0]
            par = ds.shared['p'][paragraph_pointers[0]]
            
            """
            print(par)
            print(paragraph_pointers)
            print(ds.data.keys())
            print(ds.shared.keys())
            assert(False)
            print(ds.data['span_answerss'][0])
            print(ds.data['answerss'][0])
            print(ds.data['*p'][0])
            print(ds.shared.keys())
            print(ds.shared['p'])
            assert(False)
            """
            yis.append(ds.data['y'])
            feed_dict.update(model.get_feed_dict(ds, True))

        # ASSUMPTION: Only 1 replica model
        # NOTE: sometimes y < batch size. If so pad y with zeros
        y = yis[0]
        loss_mask = [True] * len(y)
        while len(y) < model.config.batch_size:
            y.append([[[0, 0], [0, 1]]])
            loss_mask.append(False)

        yps, yp2s = sess.run([self.yps, self.yp2s], feed_dict=feed_dict)

        top_k_spans = []
        top_k_scores = []
        top_k_matches = []

        start_features = []
        end_features = []
        span_lengths = []

        yp = yps[0]
        yp2 = yp2s[0]
        _, data_set = batches[0]
        for _ in range(k):
            spans_, scores_ = zip(*[get_best_span(ypi, yp2i) for ypi, yp2i in zip(yp, yp2)])
            if len(top_k_spans) == 0:
                print("Appending spans")
                for _ in range(len(spans_)):
                    start_features.append([])
                    end_features.append([])
                    top_k_spans.append([])
                    top_k_scores.append([])
                    top_k_matches.append([])
                    span_lengths.append([])

            for i in range(0, len(spans_)):
                cur_span = spans_[i]
                cur_score = scores_[i]

                yp[i][cur_span[0][0]][cur_span[0][1]] = 0  
                yp2[i][cur_span[1][0]][cur_span[1][1] - 1] = 0  

                top_k_spans[i].append(cur_span)
                top_k_scores[i].append(cur_score)
                span_lengths[i].append(cur_span[1][1] - cur_span[0][1])


        top_k_f1s = np.array([list(map(lambda sp: F1Evaluator.span_f1(yi, sp), \
            top_k_span)) for yi, top_k_span in zip(y, top_k_spans)])
        top_k_matches = np.array([list(map(lambda sp: F1Evaluator.compare2(yi, sp), \
            top_k_span)) for yi, top_k_span in zip(y, top_k_spans)])
        top_k_scores = np.array(top_k_scores)
        best_f1_indices = np.argmax(top_k_f1s, axis=1)

        k_scores = [] 
        predicted_f1_scores = top_k_f1s[:, 0]
        predicted_matches = top_k_matches[:, 0]
        predicted_scores = top_k_scores[:, 0]
        predicted_spans = np.array(top_k_spans)[:, 0]

        print("Span shape %s f1 indices shape %s" % (len(top_k_spans), len(best_f1_indices)))
        best_spans = np.array(top_k_spans)[range(len(top_k_spans)), best_f1_indices]
        best_scores = top_k_scores[range(len(top_k_scores)), best_f1_indices]
        best_matches = np.max(top_k_matches, axis=1)
        best_f1_scores = np.max(top_k_f1s, axis=1)


        def _get(xi, span):
            if len(xi) <= span[0][0]:
                return [""]
            if len(xi[span[0][0]]) <= span[1][1]:
                return [""]
            return xi[span[0][0]][span[0][1]:span[1][1]]

        top_k_answers = np.array([list(map(lambda sp: " ".join(_get(xi, sp)), spans))
                          for xi, spans in zip(data_set.data['x'], top_k_spans)])
        best_answers = top_k_answers[range(len(top_k_answers)), best_f1_indices[0:len(top_k_answers)]]
        predicted_answers = top_k_answers[:, 0]



        results = {}
        results['q'] = data_set.data['q']
        results['answerss'] = data_set.data['answerss']
        results['x'] = data_set.data['x']
        results['predicted_answers'] = predicted_answers 
        results['best_answers'] = best_answers
        results['best_f1_indices'] = best_f1_indices
        results['top_k_f1_scores'] = top_k_f1s
        results['best_f1_scores'] = best_f1_scores 
        results['predicted_f1_scores'] = predicted_f1_scores
        results['best_spans'] = best_spans
        results['predicted_spans'] = predicted_spans
        results['top_k_spans'] = top_k_spans
        results['loss_mask'] = loss_mask
        return [results] #Hack since we only have one batch

        """
        print(top_k_f1s[0:4])
        print(predicted_f1s[0:4])
        print(best_f1s[0:4])
        print(predicted_matches[0:4])
        print(best_matches[0:4])

        print(np.sum(best_matches))
        print(np.sum(predicted_matches))
        assert(False)
        """

    def margin_step(self, sess, batches, top_k_batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, top_k_batch, model in zip(batches, top_k_batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))
            feed_dict.update(model.get_margin_feed_dict(top_k_batch, True))
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))
            
        self.get_features(sess, batches, get_summary=get_summary)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
