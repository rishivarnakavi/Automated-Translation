"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import, division, print_function

import gzip
import os
import re
import tarfile

import tensorflow as tf
from logFile import handleException, handleInfo

from tensorflow.python.platform import gfile

from six.moves import urllib

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    sentence = sentence.lower()
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path,
                      data_path,
                      max_vocabulary_size,
                      tokenizer=None,
                      normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        handleInfo("Path=========================" + str(os.path.dirname(os.path.realpath(__file__))))
        handleInfo(str("Creating vocabulary " + vocabulary_path + " from data " + data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    handleInfo("Processing line : " + str(counter))
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(
                    line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        handleInfo("Tokenizing data in : " + str(data_path))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        handleInfo("Tokenizing line : " + str(counter))
                    token_ids = sentence_to_token_ids(
                        tf.compat.as_bytes(line), vocab, tokenizer,
                        normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
    to_train_path = "corpus/en.txt"
    from_train_path = "corpus/es.txt"
    to_dev_path = "corpus/en_dev.txt"
    from_dev_path = "corpus/es_dev.txt"
    handleInfo(str(from_train_path + "   " + to_train_path))
    handleInfo(str(from_dev_path + "   " + to_dev_path))
    return prepare_data(data_dir, from_train_path, to_train_path,
                        from_dev_path, to_dev_path, en_vocabulary_size,
                        fr_vocabulary_size, tokenizer)

def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path,
                 from_vocabulary_size, to_vocabulary_size, tokenizer=None):
    to_vocab_path = os.path.join(data_dir, "vocab%d.en" % to_vocabulary_size)
    from_vocab_path = os.path.join(data_dir, "vocab%d.es" % from_vocabulary_size)
    create_vocabulary(to_vocab_path, to_train_path, to_vocabulary_size, tokenizer)
    create_vocabulary(from_vocab_path, from_train_path, from_vocabulary_size, tokenizer)

    to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
    from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)

    to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

    return (from_train_ids_path, to_train_ids_path, from_dev_ids_path,
            to_dev_ids_path, from_vocab_path, to_vocab_path)
