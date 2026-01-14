#! -*- coding:utf-8 -*-
import logging
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

DEFAULT_LABEL_LIST = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]


def load_config(config_path):
    config_path = config_path or "config.yaml"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logger(config=None):
    config = config or {}
    log_config = config.get("logging", {})
    level_name = log_config.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger("grte")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def print_config(args):
    config_path = os.path.join(args.base_path, args.dataset, "output", "config.txt")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            print(k, "=", v, file=f)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mix_prediction_sets(gold_set, pred_set, ratio, rng=None, logger=None):
    """混合预测结果，按样本比例选择ground truth或模型预测。"""
    rng = rng or random
    ratio = max(0.0, min(1.0, float(ratio)))
    gold_list = list(gold_set)
    pred_list = list(pred_set)
    total_gold = len(gold_list)
    total_pred = len(pred_list)
    roll = rng.random()
    use_groundtruth = roll < ratio

    if use_groundtruth and total_gold == 0:
        if logger:
            logger.debug(
                "混合预测: gold为空，无法使用ground truth，回退模型预测=%d",
                total_pred,
            )
        use_groundtruth = False

    if use_groundtruth:
        mixed = set(gold_list)
        gt_used, model_used = 1, 0
    else:
        mixed = set(pred_list)
        gt_used, model_used = 0, 1

    if logger:
        logger.debug(
            "混合预测: ratio=%.4f roll=%.4f use=%s gold=%d pred=%d mixed=%d",
            ratio,
            roll,
            "groundtruth" if use_groundtruth else "model",
            total_gold,
            total_pred,
            len(mixed),
        )
    return mixed, gt_used, model_used


def get_label_list(config):
    config = config or {}
    return config.get("label_list") or DEFAULT_LABEL_LIST


def build_label_maps(label_list):
    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i
    return id2label, label2id


class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.id2token = {v: k for k, v in self.vocab.items()}

    def _add_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.id2token[self.vocab[token]] = token
        return self.vocab[token]

    def encode(self, text, maxlen=None):
        tokens = ["[CLS]"] + list(text) + ["[SEP]"]
        token_ids = [self._add_token(token) for token in tokens]
        if maxlen is not None:
            token_ids = token_ids[:maxlen]
            tokens = tokens[:maxlen]
        mask = [1] * len(token_ids)
        return token_ids, mask

    def tokenize(self, text, maxlen=None):
        tokens = ["[CLS]"] + list(text) + ["[SEP]"]
        if maxlen is not None:
            tokens = tokens[:maxlen]
        return tokens

    def rematch(self, text, tokens):
        mapping = []
        text_index = 0
        for token in tokens:
            if token in ("[CLS]", "[SEP]"):
                mapping.append([])
                continue
            if text_index < len(text):
                mapping.append([text_index])
                text_index += 1
            else:
                mapping.append([])
        return mapping

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def mat_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[0] for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def tuple_mat_padding(inputs,dim=1, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[1] = (0, length - x.shape[dim])
        pad_width[2] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def sequence_padding(inputs,dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def judge(ex):
    for s,p,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d


class Vocab(object):
    def __init__(self, filename, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename), "Vocab file does not exist at " + filename
            # load from file and ignore all other params
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.id2word)
            print("Vocab size {} loaded from file".format(self.size))
        else:
            print("Creating vocab from scratch...")
            assert word_counter is not None, "word_counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than thres
                self.word_counter = dict([(k, v) for k, v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(self.word_counter, key=lambda k: self.word_counter[k], reverse=True)
            # add special tokens to the beginning
            self.id2word = ['**PAD**', '**UNK**'] + self.id2word
            self.word2id = dict([(self.id2word[idx], idx) for idx in range(len(self.id2word))])
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}".format(self.size, filename))

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id

    def save(self, filename):
        # assert not os.path.exists(filename), "Cannot save vocab: file exists at " + filename
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
        return

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.VOCAB_UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]

    def get_embeddings(self, word_vectors=None, dim=100):
        # self.embeddings = 2 * constant.EMB_INIT_RANGE * np.random.rand(self.size, dim) - constant.EMB_INIT_RANGE
        self.embeddings = np.zeros((self.size, dim))
        if word_vectors is not None:
            assert len(list(word_vectors.values())[0]) == dim, \
                "Word vectors does not have required dimension {}.".format(dim)
            for w, idx in self.word2id.items():
                if w in word_vectors:
                    self.embeddings[idx] = np.asarray(word_vectors[w])
        return self.embeddings
