import argparse
import os
import random
import time
import unicodedata
from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import HRLModel, PAD_token, EOS_token
from utils import AverageMeter
from utils import VisualizeLogger
from utils import get_logger

import re
import json

import pdb

USE_CUDA = torch.cuda.is_available()
global_step = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"x1": 3, "x2": 4, "x3": 5, "x4": 6}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "x1", 4: "x2", 5: "x3", 6: "x4"}
        self.n_words = 7  # Count default tokens

    def vocab_size(self):
        return len(self.word2index.keys())

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def data_file_process(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().lower().split('\t') for line in lines]
        lines_norm = []
        for line in lines:
            input = line[0].strip(' .')
            output = line[1].strip('"')
            input_tokens = input.split()
            if len(input_tokens) <= 1:
                continue
            line_norm = [input, output]
            lines_norm.append(line_norm)
    return lines_norm

def read_data(lang1, lang2, task_name):
    print("Reading dataset from task {}...".format(task_name))

    file_train = 'cogs_data/train.tsv'
    file_dev = 'cogs_data/dev.tsv'
    file_test = 'cogs_data/test.tsv'
    file_gen = 'cogs_data/gen.tsv'

    pairs_train = data_file_process(file_train)
    pairs_dev = data_file_process(file_dev)
    pairs_test = data_file_process(file_test)
    pairs_gen = data_file_process(file_gen)

    _input_lang = Lang(lang1)
    _output_lang = Lang(lang2)

    return _input_lang, _output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen

def prepare_dataset(lang1, lang2, task_name):
    global input_lang
    global output_lang
    assert task_name == "cogs"
    input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen = read_data(lang1, lang2, task_name)

    encode_token_filename = './preprocess/encode_tokens.txt'
    with open(encode_token_filename, 'r') as f:
        encode_tokens = f.readlines()
        for encode_token in encode_tokens:
            input_lang.index_word(encode_token.strip("\n"))

    decode_entity_filename = './preprocess/entity'
    with open(decode_entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))
    decode_caus_predicate_filename = './preprocess/caus_predicate'
    with open(decode_caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))
    decode_unac_predicate_filename = './preprocess/unac_predicate'
    with open(decode_unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))

    return input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen

def get_bound_idx(pairs, length):
    # Assume that pairs are already sorted.
    # Return the max index in pairs under certain length.
    # Warning: Will return empty when length is greater than max length in pairs.
    index = 0
    for i, pair in enumerate(pairs):
        if len(pair[0].split()) <= length:
            index = i
        else:
            return index + 1

def indexes_from_sentence(lang, sentence, type):
    if type == 'input':
        return [lang.word2index[word] for word in sentence.split(' ')]
    if type == 'output':
        return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def make_path_preparations(args, run_mode):
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if run_mode == 'train':
        log_dir = os.path.split(args.logs_path)[0]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        _logger = get_logger(f"{args.logs_path}.log")
        print(f"{args.logs_path}.log")
        _logger.info(f"random seed: {seed}")

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        _logger.info(f"checkpoint's dir is: {args.model_dir}")
        _visualizer = VisualizeLogger(summary_dir=args.model_dir)
    else:
        _logger = None
        _visualizer = None

    return _logger, _visualizer

def prepare_optimisers(args, logger, high_parameters, low_parameters):
    if args.high_optimizer == "adam":
        high_opt_class = torch.optim.Adam
    elif args.high_optimizer == "amsgrad":
        high_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.high_optimizer == "adadelta":
        high_opt_class = torch.optim.Adadelta
    else:
        high_opt_class = torch.optim.SGD

    if args.low_optimizer == "adam":
        low_opt_class = torch.optim.Adam
    elif args.low_optimizer == "amsgrad":
        low_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.low_optimizer == "adadelta":
        low_opt_class = torch.optim.Adadelta
    else:
        low_opt_class = torch.optim.SGD

    optimizer = {"high": high_opt_class(params=high_parameters, lr=args.high_lr, weight_decay=args.l2_weight),
                 "low": low_opt_class(params=low_parameters, lr=args.low_lr, weight_decay=args.l2_weight)}

    return optimizer

def perform_high_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_high_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["high"].step()
    optimizer["high"].zero_grad()


def perform_low_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_low_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["low"].step()
    optimizer["low"].zero_grad()

def test(test_data, model, example2type, device, log_file=None):
    loading_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    model.eval()
    start = time.time()

    file_right = 'gen_right.txt'
    file_wrong = 'gen_wrong.txt'

    type_right_count = {}

    with torch.no_grad():
        progress_bar = tqdm(range(len(test_data)))
        for idx in progress_bar:
            test_data_example = test_data[idx]
            # test_data_example[0] = 'a squirrel on a computer drew'
            # test_data_example[0] = 'a girl on the table helped emma'
            # pdb.set_trace()
            try:
                tokens_list = [indexes_from_sentence(input_lang, test_data_example[0], 'input')]
            except:
                continue
            tokens = Variable(torch.LongTensor(tokens_list))
            if USE_CUDA:
                tokens = tokens.cuda()
            # pdb.set_trace()
            batch_forward_info, pred_chain, label_chain = model(test_data_example, tokens, 1, is_test=True)
            # pdb.set_trace()
            normalized_entropy, log_prob, reward = batch_forward_info[0]

            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (reward == 1) else 0.]
            # pdb.set_trace()
            accuracy = torch.tensor(accuracy).mean()

            example_type = example2type[test_data_example[0]]
            if example_type not in type_right_count:
                type_right_count[example_type] = []
            type_right_count[example_type].append(accuracy.item())

            # if accuracy == 1:
            #     with open(file_right, 'a') as f:
            #         f.write(example_type + '\n')
            #         f.write(test_data_example[0] + '\n')
            #         f.write(test_data_example[1] + '\n')
            #         f.write(" ".join(pred_chain) + '\n')
            #         f.write(" ".join(label_chain) + '\n\n')
            # else:
            #     with open(file_wrong, 'a') as f:
            #         f.write(example_type + '\n')
            #         f.write(test_data_example[0] + '\n')
            #         f.write(test_data_example[1] + '\n')
            #         f.write(" ".join(pred_chain) + '\n')
            #         f.write(" ".join(label_chain) + '\n\n')

            ce_loss = accuracy
            n = 1
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            progress_bar.set_description("Test Acc {:.1f}%".format(accuracy_meter.avg * 100))

    return accuracy_meter.avg, type_right_count


def validate(valid_data, model, epoch, device, logger):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    # if len(valid_data) > 1000:
    #     # to accelerate
    #     valid_data = [random.choice(valid_data) for _ in range(1000)]

    visualizer.update_validate_size(len(valid_data))

    model.eval()
    start = time.time()

    file_right = 'dev_right.txt'
    file_wrong = 'dev_wrong.txt'
    skip_count = 0
    with torch.no_grad():
        for idx, valid_data_example in enumerate(valid_data):
            # print(idx)
            try:
                tokens_list = [indexes_from_sentence(input_lang, valid_data_example[0], 'input')]
            except:
                skip_count += 1
                continue
            tokens = Variable(torch.LongTensor(tokens_list))
            if USE_CUDA:
                tokens = tokens.cuda()

            # print('--' * 20)
            # print(valid_data_example[0])
            batch_forward_info, pred_chain, label_chain = model(valid_data_example, tokens, 1, is_test=True, epoch=epoch)
            normalized_entropy, log_prob, reward = batch_forward_info[0]

            """
            logging into visualizer
            """
            # debug_info['tree_sr_rewards'] = tree_sr_rewards
            # debug_info['decode_rewards'] = decode_rewards
            # seq = " ".join([input_lang.index2word[token.data.item()] for token in tokens[0]])
            # tree = seq2tree(seq, tree_actions, sr_actions, swr_actions)
            # visualizer.log_text(valid_data_example[1], tree, pred_labels, seq, debug_info)
            # visualizer.update_step()

            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (reward == 1) else 0.]
            accuracy = torch.tensor(accuracy).mean()

            # if accuracy == 1:
            #     with open(file_right, 'a') as f:
            #         f.write(valid_data_example[0] + '\n')
            #         f.write(valid_data_example[1] + '\n\n')
            # else:
            #     with open(file_wrong, 'a') as f:
            #         f.write(valid_data_example[0] + '\n')
            #         f.write(valid_data_example[1] + '\n\n')

            ce_loss = accuracy
            n = 1
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            batch_time_meter.update(time.time() - start)
            start = time.time()

            # print(accuracy, model.case)
            # logger.info(f"valid_case: {accuracy} {model.case}\n")


    # visualizer.log_performance(accuracy_meter.avg)
    # visualizer.update_epoch()

    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"n_entropy: {n_entropy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f} "
                f"skip_count: {skip_count} "
                )
    # print(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
    #         f"n_entropy: {n_entropy_meter.avg:.4f} "
    #         f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")
    model.train()
    return accuracy_meter.avg

def train(train_data, valid_data, model, optimizer, epoch, args, logger,
          total_batch_num, data_len, regular_weight):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    prob_ratio_meter = AverageMeter()
    reward_std_meter = AverageMeter()

    device = args.gpu_id
    model.train()
    start = time.time()

    if len(train_data) < 100:
        train_data = [pair for pair in train_data for _ in range(8)]
    elif len(train_data) < 500:
        train_data = [pair for pair in train_data for _ in range(2)]

    random.shuffle(train_data)
    batch_size = args.accumulate_batch_size

    if len(train_data) % batch_size == 0:
        batch_num = len(train_data) // batch_size
    else:
        batch_num = len(train_data) // batch_size + 1

    val_accuracy = 0.
    for batch_idx in range(batch_num):
        if (batch_idx + 1) * batch_size < len(train_data):
            train_pairs = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:
            train_pairs = train_data[batch_idx * batch_size:]
            batch_size = len(train_pairs)

        total_batch_num += batch_size

        loading_time_meter.update(time.time() - start)

        normalized_entropy_samples = []
        log_prob_samples = []
        rewards_samples = []

        accuracy_samples = []
        rewards_all = []

        sample_num = 10

        for example_idx in range(batch_size):
            train_pair = train_pairs[example_idx]

            # for test
            # train_pair[0] = "emma liked that a girl liked that the cat was rented a cake on a chair on the computer by michael"
            # for test

            tokens_list = [indexes_from_sentence(input_lang, train_pair[0], 'input')]
            tokens = Variable(torch.LongTensor(tokens_list))
            # pdb.set_trace()
            if USE_CUDA:
                tokens = tokens.cuda()

            batch_forward_info, pred_chain, label_chain = \
                model(train_pair, tokens, sample_num, is_test=False, epoch=epoch)

            for forward_info in batch_forward_info:
                normalized_entropy, log_prob, reward = forward_info

                accuracy = 1. if (reward == 1) else 0.

                normalized_entropy_samples.append(normalized_entropy)
                log_prob_samples.append(log_prob)
                rewards_samples.append(reward)

                rewards_all.append(reward)
                accuracy_samples.append(accuracy)

        normalized_entropy_samples = torch.cat(normalized_entropy_samples, dim=0)
        accuracy_samples = torch.tensor(accuracy_samples)
        rewards_all = torch.tensor(rewards_all)
        rewards_samples = torch.tensor(rewards_samples)

        if USE_CUDA:
            accuracy_samples = accuracy_samples.cuda()
            rewards_all = rewards_all.cuda()
            rewards_samples = rewards_samples.cuda()

        baseline = rewards_all.mean()
        accuracy = accuracy_samples.mean()

        if baseline:
            rewards_samples = rewards_samples - baseline
        # pdb.set_trace()
        log_prob_samples = torch.cat(log_prob_samples, dim=0)

        prob_ratio = (log_prob_samples - log_prob_samples.detach()).exp()
        # prob_ratio = log_prob_samples
        loss = (prob_ratio * rewards_samples).mean()
        # pdb.set_trace()

        loss = loss - regular_weight * normalized_entropy_samples.mean()

        loss.backward()
        perform_high_optimizer_step(optimizer, model, args)
        perform_low_optimizer_step(optimizer, model, args)

        normalized_entropy = normalized_entropy_samples.mean()
        n = batch_size * sample_num

        ce_loss = rewards_all.mean()
        accuracy_meter.update(accuracy.item(), n)
        ce_loss_meter.update(ce_loss.item(), n)
        reward_std_meter.update(rewards_all.std().item(), n)

        n_entropy_meter.update(normalized_entropy.item(), n)
        prob_ratio_meter.update((1.0 - loss.detach()).abs().mean().item(), n)
        batch_time_meter.update(time.time() - start)

        global global_step
        global_step += 1

        if batch_num <= 3000:
            val_num = batch_num
        else:
            val_num = 3000

        if (batch_idx + 1) % (val_num) == 0:
            logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                        f"reward_std: {reward_std_meter.avg:.4f} "
                        f"n_entropy: {n_entropy_meter.avg:.4f} loading_time: {loading_time_meter.avg:.4f} "
                        f"batch_time: {batch_time_meter.avg:.4f}")
            logger.info(f"total_batch_num: {total_batch_num} cir: {data_len}")

            # print(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
            #             f"reward_std: {reward_std_meter.avg:.4f} "
            #             f"n_entropy: {n_entropy_meter.avg:.4f} loading_time: {loading_time_meter.avg:.4f} "
            #             f"batch_time: {batch_time_meter.avg:.4f}")
            # print(f"total_batch_num: {total_batch_num} cir: {data_len}")

            val_accuracy = validate(valid_data, model, epoch, device, logger)

            global best_model_path
            logger.info("saving model...")
            best_model_path = f"{args.model_dir}/{epoch}-{batch_idx}.mdl"
            torch.save({"epoch": epoch, "batch_idx": batch_idx, "state_dict": model.state_dict()}, best_model_path)
            model.train()

        start = time.time()

    return val_accuracy, total_batch_num

def get_alignment(alignment_filename, input_lang, output_lang):
    with open(alignment_filename, 'r') as f:
        alignments = json.load(f)

    alignments_idx = {}
    for enct in alignments:
        assert alignments[enct] != ''
        dects = alignments[enct].split()
        dects = [" ".join(dect.split('-')) for dect in dects]
        enct_idx = input_lang.word2index[enct]
        dects_idx = [output_lang.word2index[dect] for dect in dects]
        alignments_idx[enct_idx] = dects_idx
    # pdb.set_trace()

    return alignments_idx

def train_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, pairs_dev, _, pairs_gen = prepare_dataset('nl', 'sparql', task_name)

    train_data, dev_data, gen_data = pairs_train, pairs_dev, pairs_gen

    train_data.sort(key=lambda p: len(p[0].split()))

    dev_data = list(set([tuple(item) for item in dev_data]))
    dev_data.sort(key=lambda p: len(p[0].split()))
    dev_data = [list(item) for item in dev_data]

    print(random.choice(train_data))
    print(random.choice(dev_data))

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    # read pre-alignment file
    alignment_filename = './preprocess/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    entity_list = []
    entity_filename = './preprocess/entity'
    with open(entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            entity_list.append(decode_token.strip("\n"))

    caus_predicate_list = []
    caus_predicate_filename = './preprocess/caus_predicate'
    with open(caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            caus_predicate_list.append(decode_token.strip("\n"))

    unac_predicate_list = []
    unac_predicate_filename = './preprocess/unac_predicate'
    with open(unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            unac_predicate_list.append(decode_token.strip("\n"))

    model = HRLModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     entity_list=entity_list,
                     caus_predicate_list=caus_predicate_list,
                     unac_predicate_list=unac_predicate_list)

    if USE_CUDA:
        model = model.cuda(args.gpu_id)

    optimizer = prepare_optimisers(args, logger,
                                   high_parameters=model.get_high_parameters(),
                                   low_parameters=model.get_low_parameters())

    # checkpoint_file = 'checkpoint/models/cogs_lenall_rand1/19-14999.mdl'
    # print("loading", checkpoint_file)
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint["state_dict"])
    # data_len = 'all'
    # if data_len == 'all':
    #     valid_data = dev_data
    # else:
    #     dev_lesson_idx = get_bound_idx(dev_data, data_len)
    #     valid_data = dev_data[:dev_lesson_idx]
    # validate(valid_data, model, 0, 0, logger)
    # return

    data_len = 'all'   # 7, 21 is the max length
    epoch_count = 0
    cir_epoch_dict = {
        data_len: args.max_epoch
    }

    regular_weight = args.regular_weight
    print('Start lesson ', data_len)
    total_batch_num = 0
    for epoch in range(args.max_epoch):

        if data_len in cir_epoch_dict:
            # training epochs
            cir_epoch_num = cir_epoch_dict[data_len]
        else:
            cir_epoch_num = 1

        if data_len == 'all':
            val_accuracy, total_batch_num = train(train_data,
                                                  dev_data, model, optimizer,
                                                  epoch, args, logger,
                                                  total_batch_num, data_len, regular_weight)
        else:
            train_lesson_idx = get_bound_idx(train_data, data_len)  # get the max index under the max_length
            dev_lesson_idx = get_bound_idx(dev_data, data_len)      # get the max index under the max_length

            val_accuracy, total_batch_num = train(train_data[:train_lesson_idx],
                                                  dev_data[:dev_lesson_idx], model, optimizer,
                                                  epoch, args, logger,
                                                  total_batch_num, data_len, regular_weight)

        if val_accuracy == 1.:
            final_dev_acc = validate(train_data+dev_data, model, epoch, 0, logger)
            if final_dev_acc == 1.:
                validate(gen_data, model, epoch, 0, logger)
                logger.info("saving model...")
                best_model_path = f"{args.model_dir}/{epoch}-final.mdl"
                torch.save({"epoch": epoch, "batch_idx": "final", "state_dict": model.state_dict()}, best_model_path)
                print("Finish Training. Training Succeed :)")
                break


def test_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, pairs_dev, pairs_test, pairs_gen = prepare_dataset('nl', 'sparql', task_name)

    test_data = pairs_gen
    test_data.sort(key=lambda p: len(p[0].split()))

    args.vocab_size = input_lang.n_words
    args.label_size = output_lang.n_words

    # read pre-alignment file
    alignment_filename = './preprocess/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    entity_list = []
    entity_filename = './preprocess/entity'
    with open(entity_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            entity_list.append(decode_token.strip("\n"))

    caus_predicate_list = []
    caus_predicate_filename = './preprocess/caus_predicate'
    with open(caus_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            caus_predicate_list.append(decode_token.strip("\n"))

    unac_predicate_list = []
    unac_predicate_filename = './preprocess/unac_predicate'
    with open(unac_predicate_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            unac_predicate_list.append(decode_token.strip("\n"))

    example2type_file = './preprocess/example2type'
    with open(example2type_file, 'r') as f:
        example2type = json.load(f)

    model = HRLModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=args.label_size,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     entity_list=entity_list,
                     caus_predicate_list=caus_predicate_list,
                     unac_predicate_list=unac_predicate_list)

    if USE_CUDA:
        model = model.cuda(args.gpu_id)

    checkpoint_file = args.checkpoint
    print("loading", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    print("loading finished...")
    max_length = 'all'
    if max_length == 'all':
        test_data = test_data
    else:
        test_lesson_idx = get_bound_idx(test_data, max_length)  # get the max index under the max_length
        # test_lesson_idx = -1
        test_data = test_data[:test_lesson_idx]
    random.shuffle(test_data)
    print("Start testing ..")
    log_file = './log/' + re.split('/|\.', checkpoint_file)[-3] + "_" + re.split('/|\.', checkpoint_file)[-2] + '.txt'
    # pdb.set_trace()
    test_acc, type_right_count = test(test_data, model, example2type, args.gpu_id, log_file)
    print("Test Acc: {} %".format(test_acc * 100))
    for type in type_right_count:
        type_acc = sum(type_right_count[type]) / len(type_right_count[type])
        print(type, 'acc: ', type_acc)


def prepare_arguments(checkpoint_folder, parser):
    high_lr = 1.0   # 1.0
    low_lr = 0.1     # 0.1
    accumulate_batch_size = 1
    regular_weight = 1e-1   # 1e-1
    regular_decay_rate = 0.5
    simplicity_reward_rate = 0.5
    hidden_size = 128
    encode_mode = 'seq'

    args = {"word-dim": hidden_size,
            "hidden-dim": hidden_size,
            "composer_leaf": "no_transformation",   # no_transformation | bi_lstm_transformation
            "composer-trans-hidden": hidden_size,
            "var-normalization": "True",
            "regular-weight": regular_weight,  # 0.0001
            "clip-grad-norm": 0.5,
            "env-optimizer": "adadelta",  # adadelta
            "pol-optimizer": "adadelta",  # adadelta
            "high-lr": high_lr,  # 1.
            "low-lr": low_lr,  # 0.1
            "epsilon": 0.2,
            "l2-weight": 0.0001,
            "batch-size": 1,
            "accumulate-batch-size": accumulate_batch_size,
            "max-epoch": 30,
            "gpu-id": 0,
            "model-dir": "checkpoint/models/" + checkpoint_folder,
            "logs-path": "checkpoint/logs/" + checkpoint_folder,
            "encode-mode": encode_mode,
            "regular-decay-rate": regular_decay_rate,
            "x-ratio-rate": simplicity_reward_rate}

    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--composer_leaf", required=False, default=args["composer_leaf"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--composer-trans-hidden", required=False, default=args["composer-trans-hidden"], type=int)

    parser.add_argument("--var-normalization", default=args["var-normalization"],
                        type=lambda string: True if string == "True" else False)
    parser.add_argument("--clip-grad-norm", default=args["clip-grad-norm"], type=float,
                        help="If the value is less or equal to zero clipping is not performed.")

    parser.add_argument("--high-optimizer", required=False, default=args["env-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--low-optimizer", required=False, default=args["pol-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--high-lr", required=False, default=args["high-lr"], type=float)
    parser.add_argument("--low-lr", required=False, default=args["low-lr"], type=float)
    parser.add_argument("--epsilon", required=False, default=args["epsilon"], type=float)
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)
    parser.add_argument("--accumulate-batch-size", required=False, default=args["accumulate-batch-size"], type=int)

    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
    parser.add_argument("--gpu-id", required=False, default=args["gpu-id"], type=int)
    parser.add_argument("--model-dir", required=False, default=args["model-dir"], type=str)
    parser.add_argument("--logs-path", required=False, default=args["logs-path"], type=str)
    parser.add_argument("--encode-mode", required=False, default=args["encode-mode"], type=str)

    parser.add_argument("--regular-weight", default=args["regular-weight"], type=float)
    parser.add_argument("--regular-decay-rate", required=False, default=args["regular-decay-rate"], type=float)
    parser.add_argument("--init-regular-weight", required=False, default=1e-1, type=float)
    # default no reward decay
    parser.add_argument("--decay-r", required=False, default=1.0, type=str)
    parser.add_argument("--x-ratio-rate", required=False, default=args["x-ratio-rate"], type=float)

    return parser.parse_args()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", required=True, default='train',
                            choices=['train', 'test'], type=str,
                            help="Determine whether to train a model or test using a trained weight file")
    arg_parser.add_argument("--checkpoint", required=True, type=str,
                            help="When training, it is the folder to store model weights; "
                                 "Otherwise it is the weight path to be loaded.")
    arg_parser.add_argument("--task", required=True, type=str,
                            choices=["addjump", "around_right", "simple", "length",
                                     "extend", "mcd1", "mcd2", "mcd3", "cfq", "cogs"],
                            help="All tasks on SCAN, the task name is used to load train or test file")
    arg_parser.add_argument("--random-seed", required=False, default=2, type=int)

    parsed_args = arg_parser.parse_args()
    if parsed_args.mode == 'train':
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger, visualizer = make_path_preparations(args, parsed_args.mode)
        train_model(args, parsed_args.task, logger)
    else:
        args = prepare_arguments(parsed_args.checkpoint, arg_parser)
        logger, visualizer = make_path_preparations(args, parsed_args.mode)
        test_model(args, parsed_args.task, logger)

