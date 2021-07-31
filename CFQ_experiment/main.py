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
from model import HRLModel
from utils import AverageMeter
from utils import VisualizeLogger
from utils import get_logger
import re
import json
import pdb

USE_CUDA = torch.cuda.is_available()
global_step = 0
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"x1": 3, "x2": 4, "x3": 5, "x4": 6}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "x1", 4: "x2", 5: "x3", 6: "x4"}
        self.n_words = len(self.index2word)  # Count default tokens

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


def read_data(lang1, lang2, task_name):
    print("Reading dataset from task {}...".format(task_name))
    lines_train_encode = open(PATH+'/train/train_encode.txt', encoding='utf-8'). \
        read().strip().split('\n')
    lines_train_decode = open(PATH+'/train/train_decode.txt', encoding='utf-8'). \
        read().strip().split('\n')
    lines_dev_encode = open(PATH+'/dev/dev_encode.txt', encoding='utf-8'). \
        read().strip().split('\n')
    lines_dev_decode = open(PATH+'/dev/dev_decode.txt', encoding='utf-8'). \
        read().strip().split('\n')
    lines_test_encode = open(PATH+'/test/test_encode.txt', encoding='utf-8'). \
        read().strip().split('\n')
    lines_test_decode = open(PATH+'/test/test_decode.txt', encoding='utf-8'). \
        read().strip().split('\n')

    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(s):
        s = unicode_to_ascii(s.lower().strip())
        return s

    train_encode_sent = [normalize_string(s) for s in lines_train_encode]
    dev_encode_sent = [normalize_string(s) for s in lines_dev_encode]
    test_encode_sent = [normalize_string(s) for s in lines_test_encode]

    train_decode_sent = [re.split(' { | }', normalize_string(s))[-2] for s in lines_train_decode]
    dev_decode_sent = [re.split(' { | }', normalize_string(s))[-2] for s in lines_dev_decode]
    test_decode_sent = [re.split(' { | }', normalize_string(s))[-2] for s in lines_test_decode]

    assert len(train_encode_sent) == len(train_decode_sent)
    assert len(dev_encode_sent) == len(dev_decode_sent)
    assert len(test_encode_sent) == len(test_decode_sent)

    pairs_train = [[train_encode_sent[i], train_decode_sent[i]] for i in range(len(train_encode_sent))]
    pairs_dev = [[dev_encode_sent[i], dev_decode_sent[i]] for i in range(len(dev_encode_sent))]
    pairs_test = [[test_encode_sent[i], test_decode_sent[i]] for i in range(len(test_encode_sent))]

    _input_lang = Lang(lang1)
    _output_lang = Lang(lang2)

    return _input_lang, _output_lang, pairs_train, pairs_dev, pairs_test


def prepare_dataset(lang1, lang2, task_name):
    global input_lang
    global output_lang
    assert task_name == "cfq"
    input_lang, output_lang, pairs_train, pairs_dev, pairs_test = read_data(lang1, lang2, task_name)

    encode_token_filename = PATH+'/encode_tokens.txt'
    with open(encode_token_filename, 'r') as f:
        encode_tokens = f.readlines()
        for encode_token in encode_tokens:
            input_lang.index_word(encode_token.strip("\n"))

    decode_token_filename = PATH+'/decode_tokens.txt'
    with open(decode_token_filename, 'r') as f:
        decode_tokens = f.readlines()
        for decode_token in decode_tokens:
            output_lang.index_word(decode_token.strip("\n"))

    return input_lang, output_lang, pairs_train, pairs_dev, pairs_test


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

def get_lower_bound_idx(pairs, length):
    # Assume that pairs are already sorted.
    # Return the max index in pairs under certain length.
    # Warning: Will return empty when length is greater than max length in pairs.
    if length==0:
        return 0
    for i, pair in enumerate(pairs):
        if len(pair[0].split()) == length:
            return i


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

    return _logger, _visualizer


def prepare_optimisers(args, logger, model):

    def get_optimiser(opt):
        if opt == "adam":
            opt_class = torch.optim.Adam
        elif opt == "amsgrad":
            opt_class = partial(torch.optim.Adam, amsgrad=True)
        elif opt == "adadelta":
            opt_class = torch.optim.Adadelta
        else:
            opt_class = torch.optim.SGD
        return opt_class

    opt_class = get_optimiser(args.optimizer)

    optimizer = {"abstractor": opt_class(params=model.abstractor.parameters(), lr=args.abstractor_lr, weight_decay=args.l2_weight),
                 "classifier": opt_class(params=model.classifier.parameters(), lr=args.classifier_lr, weight_decay=args.l2_weight),
                 "composer": opt_class(params=model.composer.parameters(), lr=args.composer_lr, weight_decay=args.l2_weight),
                 "solver": opt_class(params=model.rule_based_solver.parameters(), lr=args.solver_lr, weight_decay=args.l2_weight)
                }

    return optimizer


def perform_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["abstractor"].step()
    optimizer["abstractor"].zero_grad()
    optimizer["classifier"].step()
    optimizer["classifier"].zero_grad()
    optimizer["composer"].step()
    optimizer["composer"].zero_grad()
    optimizer["solver"].step()
    optimizer["solver"].zero_grad()


def get_alignment(alignment_filename, input_lang, output_lang):
    with open(alignment_filename, 'r') as f:
        alignments = json.load(f)

    alignments_idx = {}
    
    for enct in alignments:
        assert alignments[enct] != ''
        dects = alignments[enct].split()
        dects = [" ".join(dect.split('-')) for dect in dects]
        dects_idx = [output_lang.word2index[dect] for dect in dects]
        enct_idx = [input_lang.word2index[en]  for en in enct.split()]
        enct_idx = list(map(str, enct_idx))
        alignments_idx[" ".join(enct_idx)] = dects_idx
    
    # for key in alignments_idx:
    #     ens = map(int, key.split())
    #     print([input_lang.index2word[en] for en in ens])
    #     for de in alignments_idx[key]:
    #         print(output_lang.index2word[de])

    return alignments_idx


def test(test_data, model):
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    model.eval()

    all_cnt, true_cnt = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(range(len(test_data)))
        for idx in progress_bar:
            test_data_example = test_data[idx]
            tokens_list = [indexes_from_sentence(input_lang, test_data_example[0], 'input')]
            tokens = Variable(torch.LongTensor(tokens_list))
            if USE_CUDA:
                tokens = tokens.cuda()

            batch_forward_info = model(test_data_example, tokens, 1)
            normalized_entropy, log_prob, reward, token_true = batch_forward_info[0]

            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (reward == 1) else 0.]
            accuracy = torch.tensor(accuracy).mean()
            ce_loss = accuracy
            true_cnt += token_true
            all_cnt += 1
            
            accuracy_meter.update(accuracy.item())
            ce_loss_meter.update(ce_loss.item())
            n_entropy_meter.update(normalized_entropy.item())
            progress_bar.set_description("Test Acc {:.1f}%".format(accuracy_meter.avg * 100))
    
    return accuracy_meter.avg, true_cnt/all_cnt


def validate(valid_data, model, epoch, logger):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    if len(valid_data) > 3000:
        # to accelerate
        valid_data = random.choices(valid_data, k = 3000)

    # visualizer.update_validate_size(len(valid_data))

    model.eval()
    start = time.time()

    with torch.no_grad():
        for idx, valid_data_example in enumerate(valid_data):
            tokens_list = [indexes_from_sentence(input_lang, valid_data_example[0], 'input')]
            tokens = Variable(torch.LongTensor(tokens_list))
            if USE_CUDA:
                tokens = tokens.cuda()

            batch_forward_info = model(valid_data_example, tokens, 1, epoch=epoch)
            normalized_entropy, log_prob, reward, _ = batch_forward_info[0]
            normalized_entropy = normalized_entropy.mean()
            accuracy = [1. if (reward == 1) else 0.]
            accuracy = torch.tensor(accuracy).mean()
            ce_loss = accuracy
            
            accuracy_meter.update(accuracy.item())
            ce_loss_meter.update(ce_loss.item())
            n_entropy_meter.update(normalized_entropy.item())
            batch_time_meter.update(time.time() - start)
            start = time.time()

            # print(accuracy, model.case)
            # logger.info(f"valid_case: {accuracy} {model.case}\n")

    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"n_entropy: {n_entropy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f} "
                )

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

    model.train()

    start = time.time()

    random.shuffle(train_data)
    batch_size = args.accumulate_batch_size

    if len(train_data) % batch_size == 0:
        batch_num = len(train_data) // batch_size
    else:
        batch_num = len(train_data) // batch_size + 1

    val_accuracy, best_val_acc = 0. , 0.

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

        sample_num = args.sample_num

        for example_idx in range(batch_size):
            train_pair = train_pairs[example_idx]
            tokens_list = [indexes_from_sentence(input_lang, train_pair[0], 'input')]
            tokens = Variable(torch.LongTensor(tokens_list))
            if USE_CUDA:
                tokens = tokens.cuda()

            batch_forward_info = \
                model(train_pair, tokens, sample_num, epoch=epoch)

            for forward_info in batch_forward_info:
                normalized_entropy, log_prob, reward, _ = forward_info

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
        
        log_prob_samples = torch.cat(log_prob_samples, dim=0)
        prob_ratio = (log_prob_samples - log_prob_samples.detach()).exp()
        
        loss = (prob_ratio * rewards_samples).mean()
        loss = loss - regular_weight * normalized_entropy_samples.mean()
        loss.backward()

        perform_optimizer_step(optimizer, model, args)

        normalized_entropy = normalized_entropy_samples.mean()
        ce_loss = rewards_all.mean()
        n = batch_size * sample_num

        accuracy_meter.update(accuracy.item(), n)
        ce_loss_meter.update(ce_loss.item(), n)
        reward_std_meter.update(rewards_all.std().item(), n)
        n_entropy_meter.update(normalized_entropy.item(), n)
        prob_ratio_meter.update((1.0 - loss.detach()).abs().mean().item(), n)
        batch_time_meter.update(time.time() - start)

        global global_step
        global_step += 1

        if batch_num <= 500:
            val_num = batch_num
        else:
            val_num = 250

        if (batch_idx + 1) % (val_num) == 0:
            logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                        f"reward_std: {reward_std_meter.avg:.4f} "
                        f"n_entropy: {n_entropy_meter.avg:.4f} loading_time: {loading_time_meter.avg:.4f} "
                        f"batch_time: {batch_time_meter.avg:.4f}")
            logger.info(f"total_batch_num: {total_batch_num} cir: {data_len}")

            val_accuracy = validate(valid_data, model, epoch, logger)

            if val_accuracy >= 0.8 and best_val_acc < val_accuracy:
                best_val_acc = max(best_val_acc, val_accuracy)
                logger.info("saving model...")
                save_model_path = f"{args.model_dir}/{epoch}-{batch_idx}.mdl"
                torch.save({"epoch": epoch, "batch_idx": batch_idx, "state_dict": model.state_dict()}, save_model_path)

            model.train()

        start = time.time()

        if val_accuracy >= 0.99:
            break
    
    val_accuracy = validate(valid_data, model, epoch, logger)
    
    if args.lesson==10 and val_accuracy >= 0.6:
        logger.info("saving model...")
        save_model_path = f"{args.model_dir}/{epoch}-{batch_num}.mdl"
        torch.save({"epoch": epoch, "batch_idx": batch_num, "state_dict": model.state_dict()}, save_model_path)

    return val_accuracy, total_batch_num


def train_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, pairs_train, pairs_dev, _ = prepare_dataset('nl', 'sparql', task_name)

    train_data, dev_data = pairs_train, pairs_dev
    train_data.sort(key=lambda p: len(p[0].split()))
    dev_data.sort(key=lambda p: len(p[0].split()))

    # print("An example of training and validation data:")
    # print(random.choice(train_data))
    # print(random.choice(dev_data))

    # read pre-alignment file
    alignment_filename = PATH+'/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    model = HRLModel(x_ratio_rate=args.x_ratio_rate,
                     encode_mode=args.encode_mode,
                     decay_r=args.decay_r,
                     vocab_size=input_lang.n_words,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=output_lang.n_words,
                     composer_leaf=args.composer_leaf,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     logger=logger,
                     is_test=False,
                     lesson=args.lesson)

    if USE_CUDA:
        model = model.cuda()

    optimizer = prepare_optimisers(args, logger, model)

    if args.pretrained_model:
        checkpoint_file = args.pretrained_model
        print("loading pretrained model: ", checkpoint_file)
        logger.info(f"loading pretrained model: {checkpoint_file} ")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])

    data_len = args.lesson
    print('Start lesson ', data_len)
    total_batch_num = 0
    acc_trace = []

    for epoch in range(args.max_epoch):
        train_lesson_idx = get_bound_idx(train_data, data_len)  # get the max index under the max_length
        dev_lesson_idx = get_bound_idx(dev_data, data_len)      # get the max index under the max_length

        val_accuracy, total_batch_num = train(train_data[:train_lesson_idx],
                                              dev_data[:dev_lesson_idx], model, optimizer,
                                              epoch, args, logger,
                                              total_batch_num, data_len, args.init_regular_weight)

        acc_trace.append(val_accuracy)
        logger.info(f"acc_trace: {acc_trace} ")

        if val_accuracy >= 0.99:
            print("Training Succeed on lesson{}".format(args.lesson))
            val_accuracy_all = validate(dev_data, model, epoch, args.gpu_id, logger)
            if val_accuracy_all >= 0.99:
                print("Early Stopped. Training Succeed :)")
                break


def test_model(args, task_name, logger):
    global input_lang
    global output_lang

    input_lang, output_lang, _, _, pairs_test = prepare_dataset('nl', 'sparql', task_name)

    test_data = pairs_test
    test_data.sort(key=lambda p: len(p[0].split()))

    # read pre-alignment file
    alignment_filename = PATH+'/enct2dect'
    alignments_idx = get_alignment(alignment_filename, input_lang, output_lang)

    model = HRLModel(x_ratio_rate=args.x_ratio_rate,
                     encode_mode=args.encode_mode,
                     decay_r=args.decay_r,
                     vocab_size=input_lang.n_words,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     label_dim=output_lang.n_words,
                     composer_leaf=args.composer_leaf,
                     composer_trans_hidden=args.composer_trans_hidden,
                     var_normalization=args.var_normalization,
                     input_lang=input_lang,
                     output_lang=output_lang,
                     alignments_idx=alignments_idx,
                     logger=logger,
                     is_test=True,
                     lesson=args.lesson)

    if USE_CUDA:
        model = model.cuda()

    checkpoint_file = args.pretrained_model
    print("loading trained model: ", checkpoint_file)
    logger.info(f"loading trained model: {checkpoint_file} ")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])

    max_length = args.lesson
    test_lower_idx = get_lower_bound_idx(test_data, args.min_length) # get the first index of the min_length
    test_lesson_idx = get_bound_idx(test_data, max_length)  # get the max index under the max_length
    print("Start testing ..")
    print("number of test samples: {}".format(test_lesson_idx-test_lower_idx))
    test_acc, token_true = test(test_data[test_lower_idx:test_lesson_idx], model)
    print("Test Acc: {} %".format(test_acc * 100))
    print("Chains Precision: {} %".format(token_true*100))


def prepare_arguments(checkpoint_folder, parser):
    accumulate_batch_size = 1
    regular_weight = 1e-4
    regular_decay_rate = 0.5
    simplicity_reward_rate = 0.5
    hidden_size = 128
    encode_mode = 'seq'

    args = {"word-dim": hidden_size,
            "hidden-dim": hidden_size,
            "composer_leaf": "bi_lstm_transformation",   # no_transformation | bi_lstm_transformation
            "composer-trans-hidden": hidden_size,
            "var-normalization": "True",
            "regular-weight": regular_weight,  # 0.0001
            "clip-grad-norm": 0.5,
            "env-optimizer": "adadelta",  # adadelta
            "epsilon": 0.2,
            "l2-weight": 0.0001,
            "batch-size": 1,
            "accumulate-batch-size": accumulate_batch_size,
            "max-epoch": 300,
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

    parser.add_argument("--optimizer", required=False, default=args["env-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--epsilon", required=False, default=args["epsilon"], type=float)
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)
    parser.add_argument("--accumulate-batch-size", required=False, default=args["accumulate-batch-size"], type=int)

    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
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
    arg_parser.add_argument("--task", required=False, default='cfq', type=str,
                            choices=["cfq"],
                            help="All tasks on CFQ, the task name is used to load train or test file")
    arg_parser.add_argument("--random-seed", required=False, default=2, type=int)
    arg_parser.add_argument("--lesson", required=True, default=10, type=int,
                            help="The length of training and testing data, no larger than 28")
    arg_parser.add_argument("--sample-num", required=False, default=10, type=int)
    arg_parser.add_argument("--pretrained-model", required=False, default=None, type=str,
                            help="If not continuing training on a pre-trained model, keep it as None;"
                                 "Otherwise it is the path of a pretrained model")
    arg_parser.add_argument("--data-path", required=False, default='./data/mcd1', type=str)
    arg_parser.add_argument("--abstractor-lr", required=False, default=1, type=float)
    arg_parser.add_argument("--classifier-lr", required=False, default=1, type=float)
    arg_parser.add_argument("--composer-lr", required=False, default=0.1, type=float)
    arg_parser.add_argument("--solver-lr", required=False, default=0.1, type=float)
    arg_parser.add_argument("--min-length", required=False, default=0, type=int)
    
    parsed_args = arg_parser.parse_args()
    args = prepare_arguments(parsed_args.checkpoint, arg_parser)
    logger, visualizer = make_path_preparations(args, parsed_args.mode)
    PATH = args.data_path

    if parsed_args.mode == 'train':
        train_model(args, parsed_args.task, logger)
    else:
        test_model(args, parsed_args.task, logger)