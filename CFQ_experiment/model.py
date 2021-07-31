import pdb
import random
import statistics
from itertools import chain
import math
import torch.nn.functional as F
from torch import nn
from masked_cross_entropy import *
from utils import Categorical
from modules.BinaryTreeBasedModule import BinaryTreeBasedModule
from utils import clamp_grad
import time
import copy
import pdb

USE_CUDA = torch.cuda.is_available()

special_tokens = {"PAD": 0, "SOS": 1, "EOS": 2, "x1": 3, "x2": 4, "x3": 5, "x4":6}
all_entities = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
available_src_vars = ['x1', 'x2', 'x3', 'x4']


class E:    # Entities (M0~M9)
    def __init__(self, entities):
        self.entities = entities
        self.triplets = []

    def add_E(self, E0):    # E + E = E
        self.entities = self.entities + E0.entities
        self.triplets = self.triplets + E0.triplets


class R:    # Relations (?x0 relation ?x1)
    def __init__(self, relations):
        self.left = 0
        self.relations = relations
        self.attributes = []
        self.right = 1
        self.triplets = []

    def add_R(self, R0):    # R + R = R
        self.relations = self.relations + R0.relations
        self.attributes = self.attributes + R0.attributes
        self.triplets = self.triplets + R0.triplets

    def add_A(self, A0):    # R + A = R
        self.attributes = self.attributes + A0.attributes
        self.triplets = self.triplets + A0.triplets


class A:    # Attribute (?x0 A0 A1)
    def __init__(self, attributes):
        self.left = 0
        self.attributes = []
        self.attributes.append(attributes)
        self.triplets = []

    def add_A(self, A0):    # A + A = A
        self.attributes = self.attributes + A0.attributes
        self.triplets = self.triplets + A0.triplets

    def left_plus(self):
        self.left += 1


class P:    # a special E
    def __init__(self):
        self.left = 0
        self.attributes = []
        self.nest_attributes = []
        self.nest_Ps = []
        self.triplets = []

    def add_R_E(self, R0, E0):  # R + E = P
        self.attributes.append([R0.relations, E0.entities])
        self.attributes = self.attributes + R0.attributes
        self.triplets = self.triplets + R0.triplets + E0.triplets

    def add_R_A(self, R0, A0):  # R + A = P
        self.nest_attributes.append([R0.relations, A0.attributes])
        self.attributes = self.attributes + R0.attributes
        self.triplets = self.triplets + R0.triplets + A0.triplets

    def add_R_P(self, R0, P0):  # R + P = P
        self.nest_Ps.append([R0.relations, P0])
        self.attributes = self.attributes + R0.attributes
        self.triplets = self.triplets + R0.triplets + P0.triplets

    def add_A_P(self, A0, P0):  # A + P = P
        self.attributes = self.attributes + A0.attributes + P0.attributes
        self.nest_attributes = self.nest_attributes + P0.nest_attributes
        self.nest_Ps = self.nest_Ps + P0.nest_Ps
        self.triplets = self.triplets + A0.triplets + P0.triplets

    def add_P(self, P0):    # P + P = P
        # pdb.set_trace()
        self.attributes = self.attributes + P0.attributes
        self.nest_attributes = self.nest_attributes + P0.nest_attributes
        self.nest_Ps = self.nest_Ps + P0.nest_Ps
        self.triplets = self.triplets + P0.triplets


class T:    # a special E
    def __init__(self):
        self.triplets = []

    def add_E_A(self, E0, A0):  # E + A = T
        for attribute in A0.attributes:
            self.triplets.append([E0.entities, attribute[0], attribute[1]])
        self.triplets = self.triplets + E0.triplets + A0.triplets

    def add_E_P(self, E0, P0):  # E + P = T
        for p in P0.attributes + P0.nest_attributes + P0.nest_Ps:
            self.triplets.append([E0.entities, p[0], p[1]])
        self.triplets = self.triplets + E0.triplets + P0.triplets


class BottomAbstrator(BinaryTreeBasedModule):
    # To make bottom abstractions such as 'M0' and 'executive produce'
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation,
                 trans_hidden_dim, input_lang, alignments_idx, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.sr_linear = nn.Linear(in_features=hidden_dim, out_features=2)
        self.hidden_dim = hidden_dim
        self.input_lang = input_lang
        self.alignments_idx = alignments_idx
        self.span_linear1 = nn.Linear(input_dim, 2)
        self.span_linear2 = nn.Linear(input_dim, 2)
        self.span_linear3 = nn.Linear(input_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x, sample_num, debug_info=None):
        bottom_normalized_entropy2, bottom_log_prob2, bottom_span2 = self._forward_packed(x, sample_num)
        return bottom_normalized_entropy2, bottom_log_prob2, bottom_span2

    def _forward_packed(self, x, sample_num):
        x_embedding = self.embd_parser(x)
        # mask = torch.ones((x_embedding.shape[0], x_embedding.shape[1]), dtype=torch.float32)
        # if USE_CUDA:
        #     mask = mask.cuda()
        # hidden, cell = self._transform_leafs(x_embedding, mask) 

        span_1 = x_embedding
        span_2 = (x_embedding[:,1:,:] + x_embedding[:,:-1,:]) / 2
        span_3 = (x_embedding[:,2:,:] + x_embedding[:,1:-1,:] +x_embedding[:,:-2,:]) / 3
        # print("span", span_1.shape, span_2.shape, span_3.shape)  # [1, n, 128] [1, n-1, 128] [1, n-2, 128]

        score_1 = self.span_linear1(span_1.expand(sample_num, -1, -1))
        score_2 = self.span_linear2(span_2.expand(sample_num, -1, -1))
        score_3 = self.span_linear3(span_3.expand(sample_num, -1, -1))

        mask_1 = torch.ones_like(score_1)
        mask_2 = torch.ones_like(score_2)
        mask_3 = torch.ones_like(score_3)

        cat_distr_1 = Categorical(score_1, mask_1)
        cat_distr_2 = Categorical(score_2, mask_2)
        cat_distr_3 = Categorical(score_3, mask_3)

        bottom_action_1, _ = self._sample_action(cat_distr_1, mask_1, False, None, False, None)
        bottom_action_2, _ = self._sample_action(cat_distr_2, mask_2, False, None, False, None)
        bottom_action_3, _ = self._sample_action(cat_distr_3, mask_3, False, None, False, None)
        # print("bottom_action", bottom_action_1.shape, bottom_action_2.shape, bottom_action_3.shape)  # [10, n, 2] [10, n-1, 2] [10, n-1, 2]

        bottom_action1_bool = bottom_action_1[:, :, 0].bool()
        bottom_action2_bool = bottom_action_2[:, :, 0].bool()
        bottom_action3_bool = bottom_action_3[:, :, 0].bool()

        # if span tokens not in alignments dict, do not abstract them
        x_length = len(x[0])
        for pos in range(x_length-2):
            encode_idx = [x[0][i].item() for i in range(pos, pos+3)]
            encode_idx = list(map(str, encode_idx))
            remv_idx = None
            if USE_CUDA:
                remv_idx = torch.LongTensor([pos]).cuda()
            else:
                remv_idx = torch.LongTensor([pos])
            if encode_idx[0] not in self.alignments_idx:
                bottom_action1_bool.index_fill_(1, remv_idx, False)
            if " ".join(encode_idx[:2]) not in self.alignments_idx:
                bottom_action2_bool.index_fill_(1, remv_idx, False)
            if " ".join(encode_idx) not in self.alignments_idx:
                bottom_action3_bool.index_fill_(1, remv_idx, False)

        if str(x[0][-1].item()) not in self.alignments_idx:
            bottom_action1_bool.index_fill_(1, torch.LongTensor([x_length-1]).cuda(), False)
        if str(x[0][-2].item()) not in self.alignments_idx:
            bottom_action1_bool.index_fill_(1, torch.LongTensor([x_length-2]).cuda(), False)
        if str(x[0][-2].item())+" "+str(x[0][-1].item()) not in self.alignments_idx:
            bottom_action2_bool.index_fill_(1, torch.LongTensor([x_length-2]).cuda(), False)   


        def reduce_intersec_conflit(bottom_action1, bottom_action2, flag=False): 
            span1_len, span2_len = bottom_action1.size(1), bottom_action2.size(1)
            diff =  span2_len - span1_len
            assert diff > 0
            intersec = torch.zeros_like(bottom_action2, dtype=bool)
            for i in range(0, diff+1):
                intersec[:, i:span1_len+i] = intersec[:, i:span1_len+i] | (bottom_action2[:, i:span1_len+i] & bottom_action1)
            if flag:  # Special treatment of the case when span=2 and 3
                intersec[:, :-2] = intersec[:, :-2] | (bottom_action2[:, :-2] & bottom_action1[:, 1:])
                intersec[:, 2:] = intersec[:, 2:] | (bottom_action2[:, 2:] & bottom_action1[:, :-1])
            bottom_action2[intersec] = False
            return bottom_action2

        def reduce_self_conflit(bottom_action, span):
            token_len = bottom_action.size(1)
            for i in range(1, token_len):
                bottom_action[:, i] = ~bottom_action[:, i - 1] & bottom_action[:, i]
            if span==3:
                for i in range(2, token_len):
                    bottom_action[:, i] = ~bottom_action[:, i - 2] & bottom_action[:, i]
            return bottom_action

        bottom_action3_bool = reduce_self_conflit(bottom_action3_bool, 3)
        bottom_action2_bool = reduce_intersec_conflit(bottom_action3_bool, bottom_action2_bool, flag=True)
        bottom_action1_bool = reduce_intersec_conflit(bottom_action3_bool, bottom_action1_bool)
        bottom_action2_bool = reduce_self_conflit(bottom_action2_bool, 2)
        bottom_action1_bool = reduce_intersec_conflit(bottom_action2_bool, bottom_action1_bool)

        reduce_idx1 = torch.nonzero(bottom_action1_bool)
        reduce_idx2 = torch.nonzero(bottom_action2_bool)
        reduce_idx3 = torch.nonzero(bottom_action3_bool)

        bottom_span = [[] for _ in range(sample_num)]
        [bottom_span[x[0]].append([x[1].item(), x[1].item()]) for x in reduce_idx1]
        [bottom_span[x[0]].append([x[1].item(), x[1].item() + 1]) for x in reduce_idx2]
        [bottom_span[x[0]].append([x[1].item(), x[1].item() + 2]) for x in reduce_idx3]

        reduce = torch.tensor([1., 0.])
        not_reduce = torch.tensor([0., 1.])
        if USE_CUDA:
            reduce = reduce.cuda()
            not_reduce = not_reduce.cuda()
        bottom_action_2[bottom_action2_bool] = reduce
        bottom_action_2[~bottom_action2_bool] = not_reduce
        bottom_action_3[bottom_action3_bool] = reduce
        bottom_action_3[~bottom_action3_bool] = not_reduce
        # print("after conflit resolves", bottom_action2_bool, bottom_action_2)

        normalized_entropy1, log_prob1 = cat_distr_1.normalized_entropy, -cat_distr_1.log_prob(bottom_action_1)
        normalized_entropy2, log_prob2 = cat_distr_2.normalized_entropy, -cat_distr_2.log_prob(bottom_action_2)
        normalized_entropy3, log_prob3 = cat_distr_3.normalized_entropy, -cat_distr_3.log_prob(bottom_action_3)

        normalized_entropy1, log_prob1 = normalized_entropy1.mean(dim=1), log_prob1.sum(dim=1)
        normalized_entropy2, log_prob2 = normalized_entropy2.mean(dim=1), log_prob2.sum(dim=1)
        normalized_entropy3, log_prob3 = normalized_entropy3.mean(dim=1), log_prob3.sum(dim=1)

        bottom_normalized_entropy = (normalized_entropy1 + normalized_entropy2 + normalized_entropy3) / 3.

        bottom_log_prob = log_prob1 + log_prob2 + log_prob3

        # pdb.set_trace()
        # try:
        #     assert len(bottom_normalized_entropy/action_cnt) == len(bottom_log_prob) == len(bottom_span) == sample_num
        # except:
        #     pdb.set_trace()

        return bottom_normalized_entropy, bottom_log_prob, bottom_span

    def _forward(self, x, relaxed=False, tau_weights=None, straight_through=False):
        x_embedding = self.embd_parser(x)
        # print("x.shape", x.shape)  # 1,16
        # print("embedding.shape", x_embedding.shape)  # 1,16,128

        bottom_probs = []
        bottom_gumbel_noise = []
        bottom_actions = []
        bottom_entropy = []
        bottom_normalized_entropy = []
        bottom_log_prob = []

        reduce_span = []

        all_span = []

        debug_reduce_probs = []

        mask = torch.ones((x_embedding.shape[0], x_embedding.shape[1]), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()
        hidden, cell = self._transform_leafs(x_embedding, mask)
        # print("hidden", hidden.shape)  # 1,16,128

        for i in range(x_embedding.shape[1]):
            noise_i = None
            eval_swr_actions_i = None

            bottom_cat_distr, bottom_gumbel_noise_i, bottom_actions_i = self._swr_make_step(hidden, i, relaxed,
                                                                                            tau_weights,
                                                                                            straight_through, noise_i,
                                                                                            eval_swr_actions_i)
            # m0~m9 are manually reduced
            if self.input_lang.index2word[x[0][i].item()] in all_entities:
                bottom_actions_i = torch.tensor([[1., 0.]])
                if USE_CUDA:
                    bottom_actions_i = bottom_actions_i.cuda()

            debug_reduce_probs.append(float(bottom_cat_distr.probs[0][0]))
            bottom_probs.append(bottom_cat_distr.probs)
            bottom_gumbel_noise.append(bottom_gumbel_noise_i)
            bottom_actions.append(bottom_actions_i)
            bottom_entropy.append(bottom_cat_distr.entropy)
            bottom_normalized_entropy.append(bottom_cat_distr.normalized_entropy)
            bottom_log_prob.append(-bottom_cat_distr.log_prob(bottom_actions_i))

            span = [i, i]

            all_span.append(span)

            if bottom_actions_i[0, 0] == 1:
                reduce_span.append(span)

        for i in range(1, x_embedding.shape[1]):
            # Max span length is 3.
            if i >= 3:
                break
            hidden, cell = self._make_step_tree(hidden, cell)
            candidate_p_idx = self.get_candidate_p_idx(i, x.shape[1], reduce_span)  # get candidate parent node idx
            last_reduce_span = [-1, -1]
            for p_idx in candidate_p_idx:
                span = [p_idx, p_idx + i]
                all_span.append(span)

                noise_i = None
                eval_sr_actions_i = None
                bottom_cat_distr, bottom_gumbel_noise_i, bottom_actions_i = self._sr_make_step(hidden, p_idx, relaxed,
                                                                                               tau_weights,
                                                                                               straight_through,
                                                                                               noise_i,
                                                                                               eval_sr_actions_i)
                # Make sure no overlap between reduced spans
                if span[0] <= last_reduce_span[1]:
                    bottom_actions_i = torch.tensor([[0., 1.]])
                    if USE_CUDA:
                        bottom_actions_i = bottom_actions_i.cuda()

                debug_reduce_probs.append(float(bottom_cat_distr.probs[0][0]))
                bottom_probs.append(bottom_cat_distr.probs)
                bottom_gumbel_noise.append(bottom_gumbel_noise_i)
                bottom_actions.append(bottom_actions_i)
                bottom_entropy.append(bottom_cat_distr.entropy)
                bottom_normalized_entropy.append(bottom_cat_distr.normalized_entropy)
                bottom_log_prob.append(-bottom_cat_distr.log_prob(bottom_actions_i))

                if bottom_actions_i[0, 0] == 1:
                    reduce_span.append(span)
                    last_reduce_span = span
            # pdb.set_trace()

        # print("bottom_normalized_entropy", len(bottom_normalized_entropy))
        # print("bottom_log_prob", len(bottom_log_prob))
        # print("reduce_span", len(reduce_span))

        entropy = sum(bottom_entropy)
        normalized_entropy = sum(bottom_normalized_entropy) / len(bottom_normalized_entropy)
        bottom_log_prob = sum(bottom_log_prob)

        assert relaxed is False

        bottom_rl_infos = [entropy, normalized_entropy, bottom_actions, bottom_log_prob, all_span, reduce_span]
        return bottom_rl_infos

    def get_candidate_p_idx(self, depth, leaf_length, reduce_span):
        # If one of the child node has been reduced, the parent node cannot be reduced, otherwise will have overlap
        candidate_p_idx = []
        for p_idx in range(leaf_length - depth):
            p_span = [p_idx, p_idx + depth]
            IS_CANDIDATE = True
            for span in reduce_span:
                if (p_span[0] <= span[0] <= p_span[1]) or (p_span[0] <= span[1] <= p_span[1]):
                    IS_CANDIDATE = False
                    break
            if IS_CANDIDATE is True:
                candidate_p_idx.append(p_idx)

        return candidate_p_idx

    def _swr_make_step(self, hidden, i, relaxed, tau_weights, straight_through, gumbel_noise, ev_swr_actions):
        # make step on single token reduce or not
        word_index = i
        h_word = hidden[:, word_index]
        sr_score = self.sr_linear(h_word)
        sr_mask = torch.ones_like(sr_score)

        sr_cat_distr = Categorical(sr_score, sr_mask)
        if ev_swr_actions is None:
            sr_actions, gumbel_noise = self._sample_action(sr_cat_distr, sr_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            sr_actions = ev_swr_actions

        return sr_cat_distr, gumbel_noise, sr_actions

    def _sr_make_step(self, hidden, p_idx, relaxed, tau_weights, straight_through, gumbel_noise, ev_sr_actions):
        # make step on span reduce or not
        actions_index = p_idx
        h_act = hidden[:, actions_index]
        sr_score = self.sr_linear(h_act)
        sr_mask = torch.ones_like(sr_score)

        sr_cat_distr = Categorical(sr_score, sr_mask)
        if ev_sr_actions is None:
            sr_actions, gumbel_noise = self._sample_action(sr_cat_distr, sr_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            sr_actions = ev_sr_actions

        return sr_cat_distr, gumbel_noise, sr_actions

    def _make_step_tree(self, hidden, cell):
        # ==== calculate the prob distribution over the merge actions and sample one ====

        h_l, c_l = hidden[:, :-1], cell[:, :-1]
        h_r, c_r = hidden[:, 1:], cell[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)

        return h_p, c_p

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class BottomUpTreeComposer(BinaryTreeBasedModule):
    # To generate a binary tree structure based on bottom abstractions
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim, input_lang, output_lang,
                self_attention_in_tree=False, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.sr_linear = nn.Linear(in_features=hidden_dim, out_features=2)
        self.use_self_attention = self_attention_in_tree
        if self.use_self_attention:
            self.q = nn.Parameter(torch.empty(size=(hidden_dim * 2,), dtype=torch.float32))
        else:
            self.q = nn.Parameter(torch.empty(size=(hidden_dim,), dtype=torch.float32))
        # if use self attention, we should employ these parameters
        self.bilinear_w = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.var_linear = nn.Linear(in_features=hidden_dim, out_features=3)
        self.hidden_dim = hidden_dim
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)
        nn.init.uniform_(self.bilinear_w.weight, -0.1, 0.1)

    def forward(self, pair, x, bottom_span_batch, span2output_token_batch,
                relaxed=False, tau_weights=None, straight_through=False, debug_info=None):

        batch_size = len(bottom_span_batch)
        length_ori = len(x[0])
        length_batch = [length_ori for _ in range(batch_size)]

        e_tokens = [self.output_lang.index2word[i] for i in range(7, 17)]
        r_tokens = [self.output_lang.index2word[i] for i in range(17, 55)]
        a_tokens = [self.output_lang.index2word[i] for i in range(55, 83)]

        span2output_token_dict_batch = []
        for span2output_token in span2output_token_batch:
            span2output_token_dict = {}
            for span_token in span2output_token:
                span2output_token_dict[str(span_token[0])] = span_token[1]
            span2output_token_dict_batch.append(span2output_token_dict)

        if USE_CUDA:
            single_mask = torch.tensor([[1.]]).cuda()
            h_x1, c_x1 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x1']]]).cuda()), mask=single_mask)
            h_x2, c_x2 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x2']]]).cuda()), mask=single_mask)
            h_x3, c_x3 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x3']]]).cuda()), mask=single_mask)
            h_pad, c_pad = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['PAD']]]).cuda()), mask=single_mask)
        else:
            single_mask = torch.tensor([[1.]])
            h_x1, c_x1 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x1']]])), mask=single_mask)
            h_x2, c_x2 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x2']]])), mask=single_mask)
            h_x3, c_x3 = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['x3']]])), mask=single_mask)
            h_pad, c_pad = self._transform_leafs(self.embd_parser(torch.tensor([[special_tokens['PAD']]])), mask=single_mask)

        span_start_end = [[i, i] for i in range(length_ori)]
        span_start_end_batch = [span_start_end for _ in range(batch_size)]

        for in_batch_idx in range(batch_size):
            bottom_span_batch[in_batch_idx].sort(key=lambda span: span[0], reverse=True)

        var_normalized_entropy = []
        var_log_prob = []

        x_embedding = self.embd_parser(x)
        x_embedding = x_embedding.expand(batch_size, x_embedding.shape[1], x_embedding.shape[2])
        mask = torch.ones((x_embedding.shape[0], x_embedding.shape[1]), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()
        hidden_1, cell_1 = self._transform_leafs(x_embedding, mask)

        for in_batch_idx in range(batch_size):
            bottom_span = bottom_span_batch[in_batch_idx]
            bottom_span.sort(key=lambda span: span[0], reverse=True)
            span_start_end = span_start_end_batch[in_batch_idx]
            for span in bottom_span:
                span_start_end = span_start_end[:span[0]] + [span] + span_start_end[span[1] + 1:]
                span_start_end_batch[in_batch_idx] = span_start_end
                action_idx = span[0]
                if span[1] - span[0] == 0:
                    h_pad_list = []
                    c_pad_list = []
                elif span[1] - span[0] == 1:
                    h_pad_list = [h_pad]
                    c_pad_list = [c_pad]
                    length_batch[in_batch_idx] -= 1
                else:
                    assert span[1] - span[0] == 2
                    h_pad_list = [h_pad, h_pad]
                    c_pad_list = [c_pad, c_pad]
                    length_batch[in_batch_idx] -= 2

                span2output_token_dict = span2output_token_dict_batch[in_batch_idx]
                token = span2output_token_dict[str(span)]
                if token in e_tokens:
                    h_x, c_x = h_x1, c_x1
                elif token in r_tokens:
                    h_x, c_x = h_x2, c_x2
                else:
                    assert token in a_tokens
                    h_x, c_x = h_x3, c_x3
                # pdb.set_trace()
                hidden_1_one_batch = torch.cat(
                    [hidden_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     h_x,
                     hidden_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]] + h_pad_list, dim=1)
                cell_1_one_batch = torch.cat(
                    [cell_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     c_x,
                     cell_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]] + c_pad_list, dim=1)

                hidden_1 = torch.cat([hidden_1[:in_batch_idx], hidden_1_one_batch, hidden_1[in_batch_idx+1:]], dim=0)
                cell_1 = torch.cat([cell_1[:in_batch_idx], cell_1_one_batch, cell_1[in_batch_idx+1:]], dim=0)

        hidden, cell = hidden_1, cell_1

        reduce_span_in_all_span_batch = [[] for _ in range(batch_size)]
        for in_batch_idx in range(batch_size):
            bottom_span = bottom_span_batch[in_batch_idx]
            span_start_end = span_start_end_batch[in_batch_idx]
            for span in span_start_end:
                if span in bottom_span:
                    reduce_span_in_all_span_batch[in_batch_idx].append([span])
                else:
                    reduce_span_in_all_span_batch[in_batch_idx].append([])

        parent_child_spans_batch = [[] for _ in range(batch_size)]
        span2repre_batch = [{} for _ in range(batch_size)]

        normalized_entropy = []
        log_prob = []

        mask = torch.zeros((batch_size, length_ori), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()
        for in_batch_idx in range(batch_size):
            for idx in range(length_ori):
                if idx < length_batch[in_batch_idx]:
                    mask[in_batch_idx, idx] = 1.

        for i in range(1, x_embedding.shape[1]):
            # pdb.set_trace()
            noise = None
            ev_actions = None

            cat_distr, _, actions, hidden, cell, nt_list = self._make_step(hidden, cell, mask[:, i:],
                                                                                 relaxed, tau_weights,
                                                                                 straight_through, noise,
                                                                                 ev_actions)

            actions_idx = actions.argmax(dim=1)
            hidden_parent = hidden[torch.arange(hidden.shape[0]), actions_idx]  # batch_size * hidden_size
            var_cat_distr, _, var_actions = self._var_make_step(hidden_parent, relaxed,
                                                               tau_weights,
                                                               straight_through, noise,
                                                               ev_actions)

            reduce_list = []
            for in_batch_idx in range(batch_size):
                if in_batch_idx not in nt_list:
                    continue

                action_idx = actions[in_batch_idx].argmax().item()
                span_start_end = span_start_end_batch[in_batch_idx]
                merged_span = [span_start_end[action_idx][0], span_start_end[action_idx+1][1]]
                # update original span_start_end_batch
                span_start_end_batch[in_batch_idx] = \
                    span_start_end[:action_idx] + [merged_span] + span_start_end[action_idx+2:]

                reduce_span_in_all_span = reduce_span_in_all_span_batch[in_batch_idx]
                reduce_span = reduce_span_in_all_span[action_idx] + reduce_span_in_all_span[action_idx+1]
                #update original reduce_span_in_all_span_batch
                reduce_span_in_all_span_batch[in_batch_idx] = \
                    reduce_span_in_all_span[:action_idx] + [reduce_span] + reduce_span_in_all_span[action_idx+2:]

                # If a span contains 2 reduced spans, this span will be reduced
                if len(reduce_span) >= 2:
                    assert len(reduce_span) == 2
                    reduce_list.append(in_batch_idx)
                    reduce_span_in_all_span_batch[in_batch_idx][action_idx] = [merged_span]
                    span2repre_batch[in_batch_idx][str(merged_span)] = \
                        hidden[in_batch_idx:in_batch_idx+1, action_idx]

                    if var_actions[in_batch_idx, 0] == 1:
                        h_x, c_x = h_x1, c_x1
                    elif var_actions[in_batch_idx, 1] == 1:
                        h_x, c_x = h_x2, c_x2
                    else:
                        h_x, c_x = h_x3, c_x3
                    # pdb.set_trace()
                    hidden, cell = self.abst_embed(hidden, cell, h_x, c_x, in_batch_idx, action_idx)

                    parent_child_span = [merged_span, reduce_span]
                    parent_child_spans_batch[in_batch_idx].append(parent_child_span)

            nt_mask = [1 if i in nt_list else 0 for i in range(batch_size)]
            nt_mask = torch.tensor(nt_mask, dtype=torch.float32)
            reduce_mask = [1 if i in reduce_list else 0 for i in range(batch_size)]
            reduce_mask = torch.tensor(reduce_mask, dtype=torch.float32)
            if USE_CUDA:
                nt_mask = nt_mask.cuda()
                reduce_mask = reduce_mask.cuda()

            normalized_entropy.append(cat_distr.normalized_entropy * nt_mask)
            log_prob.append(-cat_distr.log_prob(actions) * nt_mask)

            var_normalized_entropy.append(var_cat_distr.normalized_entropy * reduce_mask)
            var_log_prob.append(-var_cat_distr.log_prob(var_actions) * reduce_mask)
            # pdb.set_trace()

            # hidden.sum().backward(retain_graph=True)

        # pdb.set_trace()

        log_prob = sum(log_prob) + sum(var_log_prob)
        normalized_entropy = (sum(normalized_entropy) + sum(var_normalized_entropy)) / (len(normalized_entropy) + len(var_normalized_entropy))

        assert relaxed is False

        tree_rl_infos = [normalized_entropy, log_prob, parent_child_spans_batch, span2repre_batch]
        return tree_rl_infos

    def abst_embed(self, hidden, cell, h_x, c_x, in_batch_idx, action_idx):
        # replace the abstrat with certain variable (x1 or x2)
        h_p_new = torch.cat([hidden[in_batch_idx:in_batch_idx+1, :action_idx],
                             h_x,
                             hidden[in_batch_idx:in_batch_idx+1, action_idx+1:]], dim=1)
        c_p_new = torch.cat([cell[in_batch_idx:in_batch_idx+1, :action_idx],
                             c_x,
                             cell[in_batch_idx:in_batch_idx+1, action_idx+1:]], dim=1)
        h_batch_new = torch.cat([hidden[:in_batch_idx],
                                 h_p_new,
                                 hidden[in_batch_idx+1:]], dim=0)
        c_batch_new = torch.cat([cell[:in_batch_idx],
                                 c_p_new,
                                 cell[in_batch_idx+1:]], dim=0)

        return h_batch_new, c_batch_new

    def _var_make_step(self, span_repr, relaxed, tau_weights, straight_through, gumbel_noise, ev_sr_actions):
        # make step on choice of x1 or x2
        var_score = self.var_linear(span_repr)
        var_mask = torch.ones_like(var_score)

        var_cat_distr = Categorical(var_score, var_mask)
        if ev_sr_actions is None:
            var_actions, gumbel_noise = self._sample_action(var_cat_distr, var_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            var_actions = ev_sr_actions

        return var_cat_distr, gumbel_noise, var_actions

    def _make_step(self, hidden, cell, mask_ori, relaxed, tau_weights, straight_through, gumbel_noise, ev_actions):
        # make step on generating binary tree
        mask = copy.deepcopy(mask_ori)
        batch_size = len(mask)
        nt_list = [in_batch_idx for in_batch_idx in range(batch_size)]
        for in_batch_idx in range(batch_size):
            if mask[in_batch_idx].sum() == 0.:
                nt_list.remove(in_batch_idx)
                mask[in_batch_idx, 0] = 1.

        h_l, c_l = hidden[:, :-1], cell[:, :-1]
        h_r, c_r = hidden[:, 1:], cell[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)

        if self.use_self_attention:
            cand_size = h_p.shape[1]
            query_vector = h_p.unsqueeze(dim=2).repeat(1, 1, cand_size, 1). \
                view(-1, cand_size * cand_size, self.hidden_dim)
            value_vector = h_p.unsqueeze(dim=1).repeat(1, cand_size, 1, 1). \
                view(-1, cand_size * cand_size, self.hidden_dim)
            attn_score = torch.tanh(self.bilinear_w(query_vector, value_vector))
            attn_weights = F.softmax(attn_score.view(-1, cand_size, cand_size), dim=2).view(-1, cand_size * cand_size,
                                                                                            1)
            value_vector_flatten = value_vector * attn_weights
            attn_vector = value_vector_flatten.view(-1, cand_size, cand_size, self.hidden_dim).sum(dim=2)
            q_mul_vector = torch.cat([h_p, attn_vector], dim=-1)
        else:
            q_mul_vector = h_p

        score = torch.matmul(q_mul_vector, self.q)  # (N x L x d, d) -> (N x L)
        cat_distr = Categorical(score, mask)
        if ev_actions is None:
            actions, gumbel_noise = self._sample_action(cat_distr, mask, relaxed, tau_weights, straight_through,
                                                        gumbel_noise)
        else:
            
            actions = ev_actions
        # ==== incorporate sampled action into the agent's representation of the environment state ====
        h_p, c_p = BinaryTreeBasedModule._merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask)

        return cat_distr, gumbel_noise, actions, h_p, c_p, nt_list

    def _make_step_tree(self, hidden, cell):
        # ==== calculate the prob distribution over the merge actions and sample one ====

        h_l, c_l = hidden[:, :-1], cell[:, :-1]
        h_r, c_r = hidden[:, 1:], cell[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)

        return h_p, c_p

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class BottomClassifier(BinaryTreeBasedModule):
    # To classify bottom abstractions
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim,
                 input_lang, output_lang, alignments_idx, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.hidden_dim = hidden_dim
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.class_num = self.output_lang.n_words - 7
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=self.class_num)
        self.alignments_idx = alignments_idx

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x, bottom_span_batch,
                relaxed=False, tau_weights=None, straight_through=False):

        batch_size = len(bottom_span_batch)
        x_embedding = self.embd_parser(x)
        x_embedding = x_embedding.expand(batch_size, x_embedding.shape[1], x_embedding.shape[2])
        span2output_token_batch = [[] for _ in range(batch_size)]
        mask = torch.ones((x_embedding.shape[0], x_embedding.shape[1]), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()

        hidden_1, cell_1 = self._transform_leafs(x_embedding, mask)
        hidden_2, cell_2 = self._make_step_tree(hidden_1, cell_1)
        hidden_3, cell_3 = self._make_step_tree(hidden_2, cell_2)

        noise = None
        eval_swr_actions = None

        mask_1, mask_2, mask_3 = self.generate_mask(x)
        cat_distr_1, _, actions_1 = self._classifier_make_step(hidden_1, mask_1,
                                                                relaxed, tau_weights,
                                                                straight_through, noise,
                                                                eval_swr_actions)
        cat_distr_2, _, actions_2 = self._classifier_make_step(hidden_2, mask_2,
                                                               relaxed, tau_weights,
                                                               straight_through, noise,
                                                               eval_swr_actions)
        cat_distr_3, _, actions_3 = self._classifier_make_step(hidden_3, mask_3,
                                                               relaxed, tau_weights,
                                                               straight_through, noise,
                                                               eval_swr_actions)

        bottom_span_1_mask = torch.zeros_like(cat_distr_1.normalized_entropy)
        bottom_span_2_mask = torch.zeros_like(cat_distr_2.normalized_entropy)
        bottom_span_3_mask = torch.zeros_like(cat_distr_3.normalized_entropy)

        if USE_CUDA:
            bottom_span_1_mask = bottom_span_1_mask.cuda()
            bottom_span_2_mask = bottom_span_2_mask.cuda()
            bottom_span_3_mask = bottom_span_3_mask.cuda()

        for in_batch_idx in range(batch_size):
            bottom_span = bottom_span_batch[in_batch_idx]
            for span in bottom_span:
                pos = span[0]
                if span[1] - span[0] == 0:
                    bottom_span_1_mask[in_batch_idx, span[0]] = 1.
                    actions = actions_1[in_batch_idx]
                elif span[1] - span[0] == 1:
                    bottom_span_2_mask[in_batch_idx, span[0]] = 1.
                    actions = actions_2[in_batch_idx]
                else:
                    bottom_span_3_mask[in_batch_idx, span[0]] = 1.
                    actions = actions_3[in_batch_idx]

                bottom_class_index = actions[pos].argmax().item() + 7
                bottom_class_output_token = self.output_lang.index2word[bottom_class_index]
                span2output_token_batch[in_batch_idx].append([span, bottom_class_output_token])

        # pdb.set_trace()
        normalized_entropy = ((cat_distr_1.normalized_entropy * bottom_span_1_mask).mean(dim=1) +
                              (cat_distr_2.normalized_entropy * bottom_span_2_mask).mean(dim=1) +
                              (cat_distr_3.normalized_entropy * bottom_span_3_mask).mean(dim=1)) / 3
        class_log_prob = (-cat_distr_1.log_prob(actions_1) * bottom_span_1_mask).sum(dim=1) + \
                         (-cat_distr_2.log_prob(actions_2) * bottom_span_2_mask).sum(dim=1) + \
                         (-cat_distr_3.log_prob(actions_3) * bottom_span_3_mask).sum(dim=1)

        bottom_class_infos = [normalized_entropy, class_log_prob, span2output_token_batch]

        return bottom_class_infos

    def generate_mask(self, x):
        x_length = len(x[0])
        class_mask_1 = torch.zeros((1, x_length, self.class_num), dtype=torch.float32)
        class_mask_2 = torch.zeros((1, x_length-1, self.class_num), dtype=torch.float32)
        class_mask_3 = torch.zeros((1, x_length-2, self.class_num), dtype=torch.float32)

        candidates_idx_1 = [[] for _ in range(x_length)]
        candidates_idx_2 = [[] for _ in range(x_length-1)]
        candidates_idx_3 = [[] for _ in range(x_length-2)]
        for pos in range(x_length-2):
            # pdb.set_trace()
            encode_idx = [x[0][i].item() for i in range(pos, pos+3)]
            encode_idx = list(map(str, encode_idx))
            span1_tok, span2_tok, span3_tok = encode_idx[0], " ".join(encode_idx[:2]), " ".join(encode_idx)
            if span1_tok in self.alignments_idx:
                candidates_idx_1[pos] = self.alignments_idx[span1_tok]
            if span2_tok in self.alignments_idx:
                candidates_idx_2[pos] = self.alignments_idx[span2_tok]
            if span3_tok in self.alignments_idx:
                candidates_idx_3[pos] = self.alignments_idx[span3_tok]   

        span1_tok = str(x[0][-1].item())
        if span1_tok in self.alignments_idx:
            candidates_idx_1[-1] = self.alignments_idx[span1_tok]        
        span1_tok, span2_tok = str(x[0][-2].item()), str(x[0][-2].item())+" "+str(x[0][-1].item())
        if span1_tok in self.alignments_idx:
            candidates_idx_1[-2] = self.alignments_idx[span1_tok] 
        if span2_tok in self.alignments_idx:
            candidates_idx_2[-1] = self.alignments_idx[span2_tok] 

        for pos, candidates_idx in enumerate(candidates_idx_1):
            for candidate_idx in candidates_idx:
                class_mask_1[0, pos, candidate_idx-7] = 1.
        for pos, candidates_idx in enumerate(candidates_idx_2):
            for candidate_idx in candidates_idx:
                class_mask_2[0, pos, candidate_idx-7] = 1.
        for pos, candidates_idx in enumerate(candidates_idx_3):
            for candidate_idx in candidates_idx:
                class_mask_3[0, pos, candidate_idx-7] = 1.

        return class_mask_1, class_mask_2, class_mask_3

    def _classifier_make_step(self, hidden, class_mask, relaxed, tau_weights, straight_through, gumbel_noise, ev_swr_actions):
        # make step on choice of decode token for a reduced span
        class_score = self.classifier(hidden)

        if USE_CUDA:
            class_score = class_score.cuda()
            class_mask = class_mask.cuda()

        class_cat_distr = Categorical(class_score, class_mask)

        if ev_swr_actions is None:
            class_actions, gumbel_noise = self._sample_action(class_cat_distr, class_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            class_actions = ev_swr_actions

        return class_cat_distr, gumbel_noise, class_actions

    def _make_step_tree(self, hidden, cell):
        # ==== calculate the prob distribution over the merge actions and sample one ====
        h_l, c_l = hidden[:, :-1], cell[:, :-1]
        h_r, c_r = hidden[:, 1:], cell[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
        return h_p, c_p

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class Solver(BinaryTreeBasedModule):
    # To compose bottom abstractions with rules
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim, output_lang, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.semantic_linear = nn.Linear(in_features=hidden_dim, out_features=5)
        self.hidden_dim = hidden_dim
        self.output_lang = output_lang

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, pair, span2output_token, parent_child_spans, span2repre,
                relaxed=False, tau_weights=None, straight_through=False, debug_info=None):

        # TODO: maybe have bugs when reduce_span is empty
        e_tokens = [self.output_lang.index2word[i] for i in range(7, 17)]
        r_tokens = [self.output_lang.index2word[i] for i in range(17, 55)]
        a_tokens = [self.output_lang.index2word[i] for i in range(55, 83)]
        # pdb.set_trace()
        span2semantic = self.init_semantic_class(span2output_token, e_tokens, r_tokens, a_tokens)
        # pdb.set_trace()

        semantic_probs = []
        semantic_gumbel_noise = []
        semantic_actions = []
        semantic_entropy = []
        semantic_normalized_entropy = []
        semantic_log_prob = []
        debug_reduce_probs = []

        noise_i = None
        eval_swr_actions_i = None

        for parent_child_span in parent_child_spans:
            parent_span = parent_child_span[0]
            child0_span = parent_child_span[1][0]
            child1_span = parent_child_span[1][1]
            child0_semantic = span2semantic[str(child0_span)]
            child1_semantic = span2semantic[str(child1_span)]
            # pdb.set_trace()
            cat_distr, gumbel_noise_i, actions_i, parent_semantic = self.semantic_merge(child0_semantic, child1_semantic,
                                                                                        span2repre[str(parent_span)],
                                                                                        relaxed, tau_weights,
                                                                                        straight_through, noise_i,
                                                                                        eval_swr_actions_i)
            span2semantic[str(parent_span)] = parent_semantic

            debug_reduce_probs.append(float(cat_distr.probs[0][0]))
            semantic_probs.append(cat_distr.probs)
            semantic_gumbel_noise.append(gumbel_noise_i)
            semantic_actions.append(actions_i)
            semantic_entropy.append(cat_distr.entropy)
            semantic_normalized_entropy.append(cat_distr.normalized_entropy)
            semantic_log_prob.append(-cat_distr.log_prob(actions_i))

            # If occur T, break
            if actions_i[0, 4] == 1:
                break

            # pdb.set_trace()

        entropy = sum(semantic_entropy)
        normalized_entropy = sum(semantic_normalized_entropy) / len(semantic_normalized_entropy)
        semantic_log_prob = sum(semantic_log_prob)

        assert relaxed is False

        semantic_rl_infos = [entropy, normalized_entropy, semantic_actions, semantic_log_prob, span2semantic, parent_span]
        return semantic_rl_infos

    def init_semantic_class(self, span2output_token, e_tokens, r_tokens, a_tokens):
        span2semantic = {}
        for span in span2output_token:
            span_position = span[0]
            output_token = span[1]

            assert output_token in e_tokens + r_tokens + a_tokens

            if output_token in e_tokens:
                span_semantic = E([output_token])
            elif output_token in r_tokens:
                span_semantic = R([output_token.split()[1]])
            else:
                span_semantic = A([[output_token.split()[1]], [output_token.split()[2]]])

            span2semantic[str(span_position)] = span_semantic

        return span2semantic

    def semantic_merge(self, child0_semantic, child1_semantic, parent_repre, relaxed, tau_weights, straight_through, gumbel_noise, ev_swr_actions):
        # Only the mask of R + A has two candidates, others only have one candidate.
        semantic_score = self.semantic_linear(parent_repre)
        semantic_mask = torch.zeros_like(semantic_score)

        cat_distr = None

        # pdb.set_trace()

        if isinstance(child0_semantic, E):
            if isinstance(child1_semantic, E):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E(child1_semantic)
                semantic_mask[0, 0] = 1
            elif isinstance(child1_semantic, R):
                parent_semantic = P()
                parent_semantic.add_R_E(child1_semantic, child0_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, A):
                parent_semantic = T()
                parent_semantic.add_E_A(child0_semantic, child1_semantic)
                semantic_mask[0, 4] = 1
            elif isinstance(child1_semantic, P):
                parent_semantic = T()
                parent_semantic.add_E_P(child0_semantic, child1_semantic)
                semantic_mask[0, 4] = 1
            elif isinstance(child1_semantic, T):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_T(child1_semantic)
                semantic_mask[0, 0] = 1

        elif isinstance(child0_semantic, R):
            if isinstance(child1_semantic, E):
                parent_semantic = P()
                parent_semantic.add_R_E(child0_semantic, child1_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, R):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_R(child1_semantic)
                semantic_mask[0, 1] = 1
            elif isinstance(child1_semantic, A):
                semantic_mask[0, 1] = 1
                semantic_mask[0, 3] = 1
                cat_distr = Categorical(semantic_score, semantic_mask)
                actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                            straight_through,
                                                            gumbel_noise)
                if actions[0, 1] == 1:
                    parent_semantic = copy.deepcopy(child0_semantic)
                    parent_semantic.add_A(child1_semantic)
                else:
                    assert actions[0, 3] == 1
                    parent_semantic = P()
                    parent_semantic.add_R_A(child0_semantic, child1_semantic)
            elif isinstance(child1_semantic, P):
                parent_semantic = P()
                parent_semantic.add_R_P(child0_semantic, child1_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, T):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_T(child1_semantic)
                semantic_mask[0, 1] = 1

        elif isinstance(child0_semantic, A):
            if isinstance(child1_semantic, E):
                parent_semantic = T()
                parent_semantic.add_E_A(child1_semantic, child0_semantic)
                semantic_mask[0, 4] = 1
            elif isinstance(child1_semantic, R):
                semantic_mask[0, 1] = 1
                semantic_mask[0, 3] = 1
                cat_distr = Categorical(semantic_score, semantic_mask)
                actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                            straight_through,
                                                            gumbel_noise)
                if actions[0, 1] == 1:
                    parent_semantic = copy.deepcopy(child1_semantic)
                    parent_semantic.add_A(child0_semantic)
                else:
                    assert actions[0, 3] == 1
                    parent_semantic = P()
                    parent_semantic.add_R_A(child1_semantic, child0_semantic)
            elif isinstance(child1_semantic, A):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_A(child1_semantic)
                semantic_mask[0, 2] = 1
            elif isinstance(child1_semantic, P):
                parent_semantic = P()
                parent_semantic.add_A_P(child0_semantic, child1_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, T):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_T(child1_semantic)
                semantic_mask[0, 2] = 1

        elif isinstance(child0_semantic, P):
            if isinstance(child1_semantic, E):
                parent_semantic = T()
                parent_semantic.add_E_P(child1_semantic, child0_semantic)
                semantic_mask[0, 4] = 1
            elif isinstance(child1_semantic, R):
                parent_semantic = P()
                parent_semantic.add_R_P(child1_semantic, child0_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, A):
                parent_semantic = P()
                parent_semantic.add_A_P(child1_semantic, child0_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, P):
                # pdb.set_trace()
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P(child1_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, T):
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_T(child1_semantic)
                semantic_mask[0, 3] = 1

        elif isinstance(child0_semantic, T):
            if isinstance(child1_semantic, E):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_T(child0_semantic)
                semantic_mask[0, 0] = 1
            elif isinstance(child1_semantic, R):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_T(child0_semantic)
                semantic_mask[0, 1] = 1
            elif isinstance(child1_semantic, A):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_T(child0_semantic)
                semantic_mask[0, 2] = 1
            elif isinstance(child1_semantic, P):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_T(child0_semantic)
                semantic_mask[0, 3] = 1
            elif isinstance(child1_semantic, T):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_T(child0_semantic)
                semantic_mask[0, 4] = 1

        if cat_distr is None:
            assert semantic_mask.sum() == 1
            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                           straight_through,
                                                           gumbel_noise)
        else:
            assert semantic_mask.sum() == 2

        return cat_distr, gumbel_noise, actions, parent_semantic

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class HRLModel(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, label_dim,
                 decay_r, encode_mode, x_ratio_rate, logger, is_test, lesson,
                 composer_leaf=BinaryTreeBasedModule.no_transformation, 
                 composer_trans_hidden=None,
                 var_normalization=False,
                 input_lang=None, output_lang=None,
                 alignments_idx={}):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.label_dim = label_dim
        self.alignments_idx = alignments_idx
        self.abstractor = BottomAbstrator(word_dim, hidden_dim, vocab_size, composer_leaf,
                                             composer_trans_hidden, input_lang, alignments_idx)
        self.composer = BottomUpTreeComposer(word_dim, hidden_dim, vocab_size, "no_transformation",
                                             composer_trans_hidden, input_lang, output_lang)
        self.classifier = BottomClassifier(word_dim, hidden_dim, vocab_size, composer_leaf,
                                             composer_trans_hidden, input_lang, output_lang, alignments_idx)
        self.rule_based_solver = Solver(word_dim, hidden_dim, vocab_size, composer_leaf,
                                        composer_trans_hidden, output_lang)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.decay_r = decay_r
        self.case = None
        self.logger = logger
        self.is_test = is_test
        self.lesson = lesson


    def case_study(self, reduced_output, token_list):
        token_list = token_list[0].split()
        sentence_len = len(token_list)
        sentences_num = len(reduced_output)
        res = []
        for i in range(sentences_num):
            sentence = reduced_output[i]
            dic = {}
            min_num, max_num = sentence_len+1, 0
            for x in sentence:
                parent = x[0]
                children = x[1]
                dic[tuple(parent)] = children
                max_num = max(max_num, parent[1])
                min_num = min(min_num, parent[0])

            def dfs(parent):
                if not parent in dic:
                    return "(" + " ".join([token_list[i] for i in range(parent[0], parent[1] + 1)]) + ")"
                children = dic[parent]
                child0 = dfs(tuple(children[0]))
                child1 = dfs(tuple(children[1]))

                def fill_gap(begin, end):
                    string = ""
                    for i in range(begin, end):
                        string += token_list[i] + " "
                    return string[:-1]

                case_string = "(" + fill_gap(parent[0], children[0][0]) + child0 \
                              + fill_gap(children[0][1] + 1, children[1][0]) + child1 \
                              + fill_gap(children[1][1] + 1, parent[1] + 1) + ")"
                return case_string

            res.append(dfs((min_num, max_num)))
        return res

    def forward(self, pair, x, sample_num, epoch=None, debug_info=None):
        # print(pair[0])
        batch_forward_info = self._forward(
            pair, x, sample_num, epoch, debug_info=debug_info)
        return batch_forward_info

    def _forward(self, pair, x, sample_num, epoch, debug_info):
        start_t = time.time()
        bottom_rl_infos = self.abstractor(x, sample_num, debug_info=debug_info)
        # print("\nabstractor time:", time.time()-start_t)
        bottom_normalized_entropy, bottom_log_prob, bottom_span_batch = bottom_rl_infos

        try:
            assert len(bottom_normalized_entropy) == len(bottom_log_prob) == len(bottom_span_batch) == sample_num
        except:
            pdb.set_trace()

        batch_forward_info = []
        continue_in_batch_idx = [i for i in range(sample_num)]

        for in_batch_idx, bottom_span in enumerate(bottom_span_batch):
            # print(bottom_span)
            if len(bottom_span) < 2:
                batch_forward_info.append([bottom_normalized_entropy[in_batch_idx].unsqueeze(0),
                                           bottom_log_prob[in_batch_idx].unsqueeze(0), 0., 0])
                continue_in_batch_idx.remove(in_batch_idx)

        if continue_in_batch_idx == []:
            return batch_forward_info

        new_sample_num = len(continue_in_batch_idx)

        bottom_normalized_entropy = [bottom_normalized_entropy[idx] for idx in continue_in_batch_idx]
        bottom_log_prob = [bottom_log_prob[idx] for idx in continue_in_batch_idx]
        bottom_span_batch = [bottom_span_batch[idx] for idx in continue_in_batch_idx]

        start_t = time.time()
        bottom_class_infos = self.classifier(x, bottom_span_batch)
        # print("classifier time:", time.time()-start_t)
        class_normalized_entropy, class_log_prob, span2output_token_batch = bottom_class_infos

        start_t = time.time()
        tree_rl_info = self.composer(pair, x, bottom_span_batch, span2output_token_batch, debug_info=debug_info)
        # print("composer time:", time.time()-start_t)
        tree_normalized_entropy, tree_log_prob, parent_child_spans_batch, span2repre_batch = tree_rl_info
        self.case = self.case_study(parent_child_spans_batch, pair)
        token_true = 0

        for in_batch_idx in range(new_sample_num):
            parent_child_spans = parent_child_spans_batch[in_batch_idx]
            span2repre = span2repre_batch[in_batch_idx]
            span2output_token = span2output_token_batch[in_batch_idx]

            try:
                semantic_rl_infos = self.rule_based_solver(pair, span2output_token, parent_child_spans, span2repre)
            except:
                pdb.set_trace()

            _, semantic_normalized_entropy, _, semantic_log_prob, span2semantic, end_span = semantic_rl_infos

            try:
                pred_chains = self.translate(span2semantic[str(end_span)])
                label_chains = self.transform([pair[1]])

                flat_func = lambda x: [y for l in x for y in flat_func(l)] if type(x) is list else [x]
                tokens_p = set(flat_func(pred_chains))
                tokens_l = set(flat_func(label_chains))
                if tokens_p&tokens_l==tokens_p:
                    token_true=1

                pred_chains = self.norm_chains(pred_chains)
                label_chains = self.norm_chains(label_chains)
                reward = self.get_reward(pred_chains, label_chains)
            except:
                reward = 0.

            normalized_entropy = bottom_normalized_entropy[in_batch_idx] + \
                                 tree_normalized_entropy[in_batch_idx] + \
                                 class_normalized_entropy[in_batch_idx] + \
                                 semantic_normalized_entropy
            # pdb.set_trace()
            log_prob = bottom_log_prob[in_batch_idx] + \
                       tree_log_prob[in_batch_idx] + \
                       class_log_prob[in_batch_idx] + \
                       semantic_log_prob

            batch_forward_info.append([normalized_entropy, log_prob, reward, token_true])

        return batch_forward_info

    def get_reward(self, pred_chains, label_chains):
        # overlap single chains / all single chains
        pred_single_chains = []
        for R_P in pred_chains[1]:
            chains = self.get_single_chains(R_P)
            for chain in chains:
                pred_single_chains.append(pred_chains[0] + chain)
        pred_single_chains_str = [" ".join(chain) for chain in pred_single_chains]

        label_single_chains = []
        for R_P in label_chains[1]:
            chains = self.get_single_chains(R_P)
            for chain in chains:
                label_single_chains.append(label_chains[0] + chain)
        label_single_chains_str = [" ".join(chain) for chain in label_single_chains]

        def post_process(chains):
            synonymy_dic = {'film.film.edited_by': '?x0 a film.film', 
                            'film.film.sequel': '?x0 a film.film', 
                            'film.film.directed_by': '?x0 a film.film', 
                            'film.film.written_by': '?x0 a film.film', 
                            'film.film_art_director.films_art_directed': '?x0 a film.film_art_director', 
                            'film.director.film': '?x0 a film.director', 
                            'film.writer.film': '?x0 a film.writer', 
                            'film.film.film_art_direction_by': '?x0 a film.film', 
                            'film.producer.films_executive_produced': '?x0 a film.producer', 
                            'film.editor.film': '?x0 a film.editor', 
                            'people.person.nationality': '?x0 a people.person', 
                            'people.person.gender': '?x0 a people.person', 
                            'film.film.costume_design_by': '?x0 a film.film', 
                            'film.film_costumer_designer.costume_design_for_film': '?x0 a film.film_costumer_designer', 
                            'film.film.cinematography': '?x0 a film.film', 
                            'film.film.prequel': '?x0 a film.film', 
                            'film.film.executive_produced_by': '?x0 a film.film', 
                            'film.cinematographer.film': '?x0 a film.cinematographer', 
                            'film.actor.film/ns:film.performance.film': '?x0 a film.actor', 
                            'film.film.starring/ns:film.performance.actor': '?x0 a film.film', 
                            'film.actor.film/ns:film.performance.character': '?x0 a film.actor', 
                            'business.employer.employees/ns:business.employment_tenure.person': '?x0 a business.employer', 
                            'people.person.employment_history/ns:business.employment_tenure.company': '?x0 a people.person', 
                            'film.film.produced_by|ns:film.film.production_companies': '?x0 a film.film', 
                            '^ns:people.person.gender': '?x0 a people.person', 
                            '^ns:people.person.nationality': '?x0 a people.person', 
                            'film.film.distributors/ns:film.film_film_distributor_relationship.distributor': '?x0 a film.film', 
                            'film.film_distributor.films_distributed/ns:film.film_film_distributor_relationship.film': '?x0 a film.film_distributor'}

            for chain in chains:
                key = chain.split()[1]
                if key in synonymy_dic:
                    attribute = synonymy_dic[key]
                    while attribute in chains:
                        chains.remove(attribute)

            return chains

        label_single_chains_str = post_process(label_single_chains_str)
        pred_single_chains_str = post_process(pred_single_chains_str)

        l_tokens = [x.split()[1:] for x in label_single_chains_str]
        l_tokens = sum(l_tokens, [])
        p_tokens = [x.split()[1:] for x in pred_single_chains_str]
        p_tokens = sum(p_tokens, [])
        reward1 = len(set(p_tokens) & set(l_tokens)) / len(set(p_tokens) | set(l_tokens))
        reward2 = len(set(pred_single_chains_str) & set(label_single_chains_str)) / len(set(pred_single_chains_str) | set(label_single_chains_str))
        reward = (reward1 + reward2) / 2

        if self.is_test:
            self.logger.info(f"{self.case} \n")
            self.logger.info(f"{label_chains} \n {pred_chains} \n")
            self.logger.info(f"{label_single_chains_str} \n {pred_single_chains_str} \n {reward}\n")

        return reward

    def get_single_chains(self, R_P):
        single_chains = []
        relations = R_P[0]
        ps = R_P[1]
        instance_strs = [isinstance(p, str) for p in ps]
        instance_lists = [isinstance(p, list) for p in ps]
        if instance_strs[0] is True:
            for instance_str in instance_strs:
                assert instance_str is True
            for instance_list in instance_lists:
                assert instance_list is False
        else:
            for instance_str in instance_strs:
                assert instance_str is False
            for instance_list in instance_lists:
                assert instance_list is True
        if instance_strs[0] is True:
            for relation in relations:
                for p in ps:
                    single_chain = [relation, p]
                    single_chains.append(single_chain)
            return single_chains
        else:
            for relation in relations:
                for p in ps:
                    p_chains = self.get_single_chains(p)
                    for p_chain in p_chains:
                        single_chains.append([relation] + p_chain)
            return single_chains

    def norm_chains(self, all_chains):
        if all_chains[0] != ['?x0']:
            for entity in all_chains[0]:
                all_chains[1].append([['is'], [entity]])
            all_chains[0] = ['?x0']
        return all_chains

    def transform(self, decode_sentences):
        decode_triplets = []
        for sentence in decode_sentences:
            str_triplets = sentence.split(" . ")
            triplets = []
            for str_triplet in str_triplets:
                if 'FILTER' not in str_triplet and 'filter' not in str_triplet:
                    triplets.append(str_triplet.split())
            decode_triplets.append(triplets)

        decode_initial_tokens = []
        decode_left_tokens = []
        decode_right_tokens = []

        for example_triplets in decode_triplets:
            left_set = set([triplet[0] for triplet in example_triplets])
            right_set = set([triplet[2] for triplet in example_triplets])
            initial_set = set()
            for left_token in left_set:
                if left_token not in right_set:
                    initial_set.add(left_token)
            if '?x0' in initial_set:
                assert len(initial_set) == 1
            decode_initial_tokens.append(initial_set)
            decode_left_tokens.append(left_set)
            decode_right_tokens.append(right_set)

        decode_chains = []

        for example_idx, example_triplets in enumerate(decode_triplets):
            initial_set = decode_initial_tokens[example_idx]
            left_set = decode_left_tokens[example_idx]
            right_set = decode_right_tokens[example_idx]

            left_token2tripltes = {}
            right_token2tripltes = {}
            for triplet in example_triplets:
                if triplet[0] not in left_token2tripltes:
                    left_token2tripltes[triplet[0]] = [triplet]
                else:
                    left_token2tripltes[triplet[0]].append(triplet)
                if triplet[2] not in right_token2tripltes:
                    right_token2tripltes[triplet[2]] = [triplet]
                else:
                    right_token2tripltes[triplet[2]].append(triplet)

            if len(initial_set) >= 2:
                assert '?x0' not in initial_set
                assert sum([len(left_token2tripltes[left_token]) for left_token in initial_set]) == len(
                    initial_set) * len(left_token2tripltes[list(initial_set)[0]])

            all_chains = self.get_chains(list(initial_set)[0], left_token2tripltes)
            all_chains = [list(initial_set), all_chains]
            decode_chains.append(all_chains)

        return decode_chains[0]

    def get_chains(self, initial_token, left_token2tripltes):
        initial_triplets = left_token2tripltes[initial_token]
        all_chains = []
        right_token2triplets = {}
        right_token2relations = {}
        for triplet in initial_triplets:
            if triplet[2] not in right_token2triplets:
                right_token2triplets[triplet[2]] = [triplet]
                right_token2relations[triplet[2]] = [triplet[1]]
            else:
                right_token2triplets[triplet[2]].append(triplet)
                right_token2relations[triplet[2]].append(triplet[1])
        while True:
            UPDATE = False
            right_token_list = [right_token for right_token in right_token2triplets]
            right_token_num = len(right_token_list)
            if right_token_num >= 2:
                for token_idx in range(right_token_num - 1):
                    token_1 = right_token_list[token_idx]
                    if token_1[0] not in ['M', 'm']:
                        continue
                    for token_idy in range(token_idx + 1, right_token_num):
                        token_2 = right_token_list[token_idy]
                        if token_2[0] not in ['M', 'm']:
                            continue
                        if set(right_token2relations[token_1]) == set(right_token2relations[token_2]):
                            new_token = token_1 + " | " + token_2
                            right_token2triplets[new_token] = right_token2triplets[token_1]
                            del right_token2triplets[token_1]
                            del right_token2triplets[token_2]
                            right_token2relations[new_token] = right_token2relations[token_1]
                            del right_token2relations[token_1]
                            del right_token2relations[token_2]
                            UPDATE = True
                            break
                    if UPDATE is True:
                        break
            else:
                break

            if UPDATE is True:
                continue
            else:
                break

        for right_token in right_token2triplets:
            same_triplets = right_token2triplets[right_token]
            same_relations = [triplet[1] for triplet in same_triplets]
            sub_chain = [same_relations]
            if right_token[0:2] != '?x':
                entities = right_token.split(" | ")
            else:
                entities = self.get_chains(right_token, left_token2tripltes)
            sub_chain.append(entities)
            all_chains.append(sub_chain)
        return all_chains

    # Translate a nest class into a list
    def translate(self, semantic):
        attributes = []
        nest_attributes = []
        nest_Ps = []
        triplets = []
        if isinstance(semantic, E):
            triplets = semantic.triplets
        elif isinstance(semantic, R):
            attributes = semantic.attributes
            triplets = semantic.triplets
        elif isinstance(semantic, A):
            attributes = semantic.attributes
            triplets = semantic.triplets
        elif isinstance(semantic, P):
            attributes = semantic.attributes
            nest_attributes = semantic.nest_attributes
            nest_Ps = semantic.nest_Ps
            triplets = semantic.triplets
        else:
            triplets = semantic.triplets

        all_chains = []
        for attribute in attributes:
            all_chains.append(attribute)
        for nest_attribute in nest_attributes:
            all_chains.append(nest_attribute)
        for nest_P in nest_Ps:
            all_chains.append(self.flat_semantic(nest_P))
        for triplet in triplets:
            all_chains.append(self.flat_semantic(triplet[1:]))

        if triplets:
            assert not attributes and not nest_attributes and not nest_Ps
            all_chains = [triplets[0][0], all_chains]
        else:
            all_chains = [['?x0'], all_chains]

        return all_chains

    # To flat the structure [relations, P]
    # Recursion
    def flat_semantic(self, R_P):
        relations = R_P[0]
        ps = R_P[1]
        if not isinstance(ps, P):
            return [relations, ps]
        else:
            flat_P = []
            # pdb.set_trace()
            if ps.attributes:
                flat_P = flat_P + ps.attributes
            if ps.nest_attributes:
                flat_P = flat_P + ps.nest_attributes
            if ps.nest_Ps:
                flat_nest_Ps = []
                R_Ps_in_ps = ps.nest_Ps
                for R_P_in_ps in R_Ps_in_ps:
                    flat_nest_Ps.append(self.flat_semantic(R_P_in_ps))
                flat_P = flat_P + flat_nest_Ps
                return [relations, flat_P]
            else:
                return [relations, flat_P]


