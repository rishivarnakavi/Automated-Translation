# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import, division, print_function

import logging
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model
from logFile import handleException, handleInfo
from six.moves import xrange  # pylint: disable=redefined-builtin

currentDirectory = os.path.dirname(__file__)

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000,
                            "Spanish vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer(
    "max_train_data_size", 0,
    "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(10, 5), (15, 10), (25, 20), (50, 40)]

_lowestPerplexity = []


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 10000 == 0:
                    handleInfo("Reading data line : " + str(counter))
                sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        handleInfo(str("Reading model parameters from : " + str(ckpt.model_checkpoint_path)))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        handleInfo(str("Created model with fresh parameters."))
        session.run(tf.global_variables_initializer())
    return model


def train():
    """Train a en->fr translation model using WMT data."""
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None
    if FLAGS.from_train_data and FLAGS.to_train_data:
        from_train_data = FLAGS.from_train_data
        to_train_data = FLAGS.to_train_data
        from_dev_data = from_train_data
        to_dev_data = to_train_data
        if FLAGS.from_dev_data and FLAGS.to_dev_data:
            from_dev_data = FLAGS.from_dev_data
            to_dev_data = FLAGS.to_dev_data
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
            FLAGS.data_dir, from_train_data, to_train_data, from_dev_data,
            to_dev_data, FLAGS.from_vocab_size, FLAGS.to_vocab_size)
    else:
        # Prepare WMT data.
        handleInfo(str("Preparing WMT data in : " + str(FLAGS.data_dir)))
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)

    with tf.Session() as sess:
        # Create model.
        handleInfo(str("Creating " + str(FLAGS.num_layers) + " layers of " + str(FLAGS.size) + " units."))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        handleInfo(
            str("Reading development and training data (limit: " +
                str(FLAGS.max_train_data_size) + ")."))
        dev_set = read_data(from_dev, to_dev)
        train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [
            sum(train_bucket_sizes[:i + 1]) / train_total_size
            for i in xrange(len(train_bucket_sizes))
        ]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([
                i for i in xrange(len(train_buckets_scale))
                if train_buckets_scale[i] > random_number_01
            ])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                perplexityRound = round(perplexity, 1)

                if perplexityRound < 9.9:
                    if not _lowestPerplexity:
                        _lowestPerplexity.append(perplexityRound)
                        _lowestPerplexity.append(0)

                    if perplexityRound == _lowestPerplexity[0]:
                        if _lowestPerplexity[1] > 9:
                            break
                        else:
                            count = _lowestPerplexity[1]
                            count = count + 1
                            _lowestPerplexity[1] = count
                    else:
                        if perplexityRound < _lowestPerplexity[0]:
                            _lowestPerplexity[0] = perplexityRound
                            _lowestPerplexity[1] = 1

                    handleInfo("Lowest Perplexity List--" + str(_lowestPerplexity))

                message = "global step " + str(
                    model.global_step.eval()) + " learning rate " + str(
                        model.learning_rate.eval()) + " step-time " + str(
                            step_time) + " perplexity " + str(perplexity)
                handleInfo(message)
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir,"translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        handleInfo(str("Eval: empty bucket : " + str(bucket_id)))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    handleInfo(str("Eval: bucket " + str(bucket_id) + " perplexity : " + str(eval_ppx)))
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        es_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.es" % FLAGS.from_vocab_size)
        en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.en" % FLAGS.to_vocab_size)
        es_vocab, _ = data_utils.initialize_vocabulary(es_vocab_path)
        _, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)

        inputFileName = os.path.join(currentDirectory, "decode/es_decode.txt")
        outputFileName = os.path.join(currentDirectory, "decode/en_decode_output.txt")
        open(outputFileName, 'w').close()
        outputFile = open(outputFileName, 'a+', encoding="utf8")
        decodeFile = open(inputFileName, encoding="utf8")
        for lineno, sentence in enumerate(decodeFile):
            token_ids = data_utils.sentence_to_token_ids(
                tf.compat.as_bytes(sentence), es_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                # outputFile.write("Sentence truncated : " + str(sentence))
                logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({
                bucket_id: [(token_ids, [])]
            }, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs,
                                             decoder_inputs, target_weights,
                                             bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [
                int(np.argmax(logit, axis=1)) for logit in output_logits
            ]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            decodedOutput = " ".join(
                [tf.compat.as_str(rev_en_vocab[output]) for output in outputs])
            handleInfo(str(lineno) + " : " + str(decodedOutput))
            outputFile.write(str(decodedOutput) + "\n")
            outputFile.flush()

        decodeFile.close()
        outputFile.close()


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
