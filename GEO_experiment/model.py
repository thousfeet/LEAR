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
import re
import pdb
import os

USE_CUDA = torch.cuda.is_available()

PAD_token = 0
SOS_token = 1
EOS_token = 2
x1_token = 3
x2_token = 4
x3_token = 5
x4_token = 6

all_entities = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
available_src_vars = ['x1', 'x2', 'x3', 'x4']


class E:  # Entities (M0~M9)
    def __init__(self, entity):
        # entity = [predicate, [entity_1, entity_2, ...]]
        self.entity = [entity]

    def add_E_and(self, E0):
        right_E = copy.deepcopy(E0)
        self.entity = ['('] + self.entity + [','] + right_E.entity + [')']

    def add_E_ex(self, E0):
        right_E = copy.deepcopy(E0)
        self.entity = ['exclude', '('] + self.entity + [','] + right_E.entity + [')']

    def add_E_in(self, E0):
        right_E = copy.deepcopy(E0)
        self.entity = ['intersection', '('] + self.entity + [','] + right_E.entity + [')']

    def add_P(self, P0):
        left_P = copy.deepcopy(P0)
        self.entity = left_P.predicate + self.entity + left_P.postfix

class P:
    def __init__(self, predicate):
        # entity = [predicate, [entity_1, entity_2, ...]]
        self.predicate = [predicate, '(']
        self.postfix = [')']

    def add_P_order(self, P0):
        right_p = copy.deepcopy(P0)
        self.predicate = self.predicate + right_p.predicate
        self.postfix = self.postfix + right_p.postfix

    def add_P_reorder(self, P0):
        right_p = copy.deepcopy(P0)
        self.predicate = right_p.predicate + self.predicate
        self.postfix = self.postfix + right_p.postfix


class BottomAbstrator(nn.Module):
    # To make bottom abstractions such as 'M0' and 'executive produce'
    def __init__(self, input_lang, alignment):
        super().__init__()
        self.input_lang = input_lang
        self.alignment = alignment

    def forward(self, x, pair):
        bottom_span = []
        bottom_span2cand_funql = {}
        span_length_prior = [3, 2, 1]
        for span_length in span_length_prior:
            position = 0
            while position < (x.size(1) - (span_length-1)):
                # if pair[0] == 'what is the population density of var0':
                #     pdb.set_trace()
                end_position = position + (span_length-1)

                overlap = False
                for span in bottom_span:
                    if (position >= span[0] and position <= span[1]) or (end_position >= span[0] and end_position <= span[1]):
                        overlap = True
                        break
                if overlap:
                    position += 1
                    continue

                token_idx = x[0][position: end_position+1]
                token = " ".join([self.input_lang.index2word[index.item()] for index in token_idx])
                if token in self.alignment:
                    bottom_span.append([position, end_position])
                    bottom_span2cand_funql[str([position, end_position])] = self.alignment[token]
                position += 1

        return bottom_span, bottom_span2cand_funql

class BottomClassifier(BinaryTreeBasedModule):
    # To classify bottom abstractions
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim,
                 input_lang, output_lang, alignments,
                 dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)

        self.output_lang = output_lang
        self.alignments = alignments

        self.hidden_dim = hidden_dim
        self.input_lang = input_lang
        self.output_lang = output_lang

        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.class_num = self.output_lang.n_words
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=self.class_num)

    def forward(self, x, pair, bottom_span, bottom_span2cand_funql, sample_num,
                relaxed=False, tau_weights=None, straight_through=False, noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None
                ):
        x_embedding = self.embd_parser(x)
        x_embedding = x_embedding.expand(sample_num, -1, -1)
        mask = torch.ones((x_embedding.shape[0], x_embedding.shape[1]), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()

        batch_normalized_entropy = [[] for _ in range(sample_num)]
        batch_log_prob = [[] for _ in range(sample_num)]
        batch_span2output_token = [[] for _ in range(sample_num)]
        batch_bottom_span_new = [[] for _ in range(sample_num)]

        hidden_1, cell_1 = self._transform_leafs(x_embedding, mask)
        hidden_2, cell_2 = self._make_step_tree(hidden_1, cell_1)
        hidden_3, cell_3 = self._make_step_tree(hidden_2, cell_2)

        for span in bottom_span:
            if span[1] - span[0] == 0:
                hidden = hidden_1[:, span[0]]
            elif span[1] - span[0] == 1:
                hidden = hidden_2[:, span[0]]
            else:
                assert span[1] - span[0] == 2
                hidden = hidden_3[:, span[0]]

            mask = torch.zeros((1, self.class_num), dtype=torch.float32)
            cand_funql = bottom_span2cand_funql[str(span)]
            for funql in cand_funql:
                funql_idx = self.output_lang.word2index[funql]
                mask[0, funql_idx] = 1.
            mask = mask.expand(sample_num, -1)

            cat_distr, _, actions = self._classifier_make_step(hidden, mask,
                                                               relaxed, tau_weights,
                                                               straight_through, noise,
                                                               eval_swr_actions)

            normalized_entropy = cat_distr.normalized_entropy
            log_prob = -cat_distr.log_prob(actions)
            for batch_idx in range(sample_num):
                action_idx = actions[batch_idx].argmax().item()
                output_token = self.output_lang.index2word[action_idx]
                batch_span2output_token[batch_idx].append([span, output_token])
                if output_token != '':
                    batch_bottom_span_new[batch_idx].append(span)

                if len(cand_funql) >= 2:
                    batch_normalized_entropy[batch_idx].append(normalized_entropy[batch_idx])
                    batch_log_prob[batch_idx].append(log_prob[batch_idx])

        # pdb.set_trace()
        batch_normalized_entropy = [sum(normalized_entropy) / len(normalized_entropy)
                                    if len(normalized_entropy) > 0 else 0. for normalized_entropy in batch_normalized_entropy]

        batch_log_prob = [sum(log_prob) for log_prob in batch_log_prob]

        return [batch_normalized_entropy, batch_log_prob, batch_bottom_span_new, batch_span2output_token]

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


class BottomUpTreeComposer(BinaryTreeBasedModule):
    # To generate a binary tree structure based on bottom abstractions
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim, input_lang,
                 output_lang,
                 alignments={}, entity_list=[], predicate_list=[], funql_type={}, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.sr_linear = nn.Linear(in_features=hidden_dim, out_features=2)

        self.q = nn.Parameter(torch.empty(size=(hidden_dim,), dtype=torch.float32))
        # if use self attention, we should employ these parameters
        self.var_linear = nn.Linear(in_features=hidden_dim, out_features=3)
        self.hidden_dim = hidden_dim

        self.input_lang = input_lang
        self.output_lang = output_lang

        self.alignments = alignments
        self.entity_list = entity_list
        self.predicate_list = predicate_list

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)

    def forward(self, pair, x, bottom_span_batch, span2output_token_batch,
                relaxed=False, tau_weights=None, straight_through=False, noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None):

        batch_size = len(bottom_span_batch)
        length_ori = len(x[0])
        length_batch = [length_ori for _ in range(batch_size)]

        e_tokens = self.entity_list
        r_tokens = self.predicate_list

        span2output_token_dict_batch = []
        for span2output_token in span2output_token_batch:
            span2output_token_dict = {}
            for span_token in span2output_token:
                span2output_token_dict[str(span_token[0])] = span_token[1]
            span2output_token_dict_batch.append(span2output_token_dict)

        if USE_CUDA:
            single_mask = torch.tensor([[1.]]).cuda()
            h_x1, c_x1 = self._transform_leafs(self.embd_parser(torch.tensor([[x1_token]]).cuda()), mask=single_mask)
            h_x2, c_x2 = self._transform_leafs(self.embd_parser(torch.tensor([[x2_token]]).cuda()), mask=single_mask)
            h_x3, c_x3 = self._transform_leafs(self.embd_parser(torch.tensor([[x3_token]]).cuda()), mask=single_mask)
            h_pad, c_pad = self._transform_leafs(self.embd_parser(torch.tensor([[PAD_token]]).cuda()), mask=single_mask)
        else:
            single_mask = torch.tensor([[1.]])
            h_x1, c_x1 = self._transform_leafs(self.embd_parser(torch.tensor([[x1_token]])), mask=single_mask)
            h_x2, c_x2 = self._transform_leafs(self.embd_parser(torch.tensor([[x2_token]])), mask=single_mask)
            h_x3, c_x3 = self._transform_leafs(self.embd_parser(torch.tensor([[x3_token]])), mask=single_mask)
            h_pad, c_pad = self._transform_leafs(self.embd_parser(torch.tensor([[PAD_token]])), mask=single_mask)

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
        # hidden_2, cell_2 = self._make_step_tree(hidden_1, cell_1)
        # hidden_3, cell_3 = self._make_step_tree(hidden_2, cell_2)
        #
        # noise = None
        # ev_actions = None
        # var_cat_distr_1, _, var_actions_1 = self._var_make_step(hidden_1, relaxed,
        #                                                        tau_weights,
        #                                                        straight_through, noise,
        #                                                        ev_actions)
        # var_cat_distr_2, _, var_actions_2 = self._var_make_step(hidden_2, relaxed,
        #                                                         tau_weights,
        #                                                         straight_through, noise,
        #                                                         ev_actions)
        # var_cat_distr_3, _, var_actions_3 = self._var_make_step(hidden_3, relaxed,
        #                                                         tau_weights,
        #                                                         straight_through, noise,
        #                                                         ev_actions)
        #
        # bottom_span_1_mask = torch.zeros_like(var_cat_distr_1.normalized_entropy)
        # bottom_span_2_mask = torch.zeros_like(var_cat_distr_2.normalized_entropy)
        # bottom_span_3_mask = torch.zeros_like(var_cat_distr_3.normalized_entropy)
        #
        # if USE_CUDA:
        #     bottom_span_1_mask = bottom_span_1_mask.cuda()
        #     bottom_span_2_mask = bottom_span_2_mask.cuda()
        #     bottom_span_3_mask = bottom_span_3_mask.cuda()

        # pdb.set_trace()

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
                    # bottom_span_1_mask[in_batch_idx, action_idx] = 1.
                    # if var_actions_1[in_batch_idx, action_idx, 0] == 1:
                    #     h_x, c_x = h_x1, c_x1
                    # elif var_actions_1[in_batch_idx, action_idx, 1] == 1:
                    #     h_x, c_x = h_x2, c_x2
                    # else:
                    #     h_x, c_x = h_x3, c_x3
                elif span[1] - span[0] == 1:
                    h_pad_list = [h_pad]
                    c_pad_list = [c_pad]
                    length_batch[in_batch_idx] -= 1
                    # bottom_span_2_mask[in_batch_idx, action_idx] = 1.
                    # if var_actions_2[in_batch_idx, action_idx, 0] == 1:
                    #     h_x, c_x = h_x1, c_x1
                    # elif var_actions_2[in_batch_idx, action_idx, 1] == 1:
                    #     h_x, c_x = h_x2, c_x2
                    # else:
                    #     h_x, c_x = h_x3, c_x3
                else:
                    assert span[1] - span[0] == 2
                    h_pad_list = [h_pad, h_pad]
                    c_pad_list = [c_pad, c_pad]
                    length_batch[in_batch_idx] -= 2
                    # bottom_span_3_mask[in_batch_idx, action_idx] = 1.
                    # if var_actions_3[in_batch_idx, action_idx, 0] == 1:
                    #     h_x, c_x = h_x1, c_x1
                    # elif var_actions_3[in_batch_idx, action_idx, 1] == 1:
                    #     h_x, c_x = h_x2, c_x2
                    # else:
                    #     h_x, c_x = h_x3, c_x3
                span2output_token_dict = span2output_token_dict_batch[in_batch_idx]
                token = span2output_token_dict[str(span)]
                if token in e_tokens:
                    h_x, c_x = h_x1, c_x1
                else:
                    assert token in r_tokens
                    h_x, c_x = h_x2, c_x2
                # pdb.set_trace()
                hidden_1_one_batch = torch.cat(
                    [hidden_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     h_x,
                     hidden_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]] + h_pad_list, dim=1)
                cell_1_one_batch = torch.cat(
                    [cell_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     c_x,
                     cell_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]] + c_pad_list, dim=1)

                hidden_1 = torch.cat([hidden_1[:in_batch_idx], hidden_1_one_batch, hidden_1[in_batch_idx + 1:]], dim=0)
                cell_1 = torch.cat([cell_1[:in_batch_idx], cell_1_one_batch, cell_1[in_batch_idx + 1:]], dim=0)

        hidden, cell = hidden_1, cell_1
        # pdb.set_trace()

        # var_normalized_entropy.append(((var_cat_distr_1.normalized_entropy * bottom_span_1_mask).mean(dim=1) +
        #                                (var_cat_distr_2.normalized_entropy * bottom_span_2_mask).mean(dim=1) +
        #                                (var_cat_distr_3.normalized_entropy * bottom_span_3_mask).mean(dim=1)) / 3.)
        # var_log_prob.append((-var_cat_distr_1.log_prob(var_actions_1) * bottom_span_1_mask).sum(dim=1) +
        #                     (-var_cat_distr_2.log_prob(var_actions_2) * bottom_span_2_mask).sum(dim=1) +
        #                     (-var_cat_distr_3.log_prob(var_actions_3) * bottom_span_3_mask).sum(dim=1))
        # pdb.set_trace()

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
                merged_span = [span_start_end[action_idx][0], span_start_end[action_idx + 1][1]]
                # update original span_start_end_batch
                span_start_end_batch[in_batch_idx] = \
                    span_start_end[:action_idx] + [merged_span] + span_start_end[action_idx + 2:]

                reduce_span_in_all_span = reduce_span_in_all_span_batch[in_batch_idx]
                reduce_span = reduce_span_in_all_span[action_idx] + reduce_span_in_all_span[action_idx + 1]
                # update original reduce_span_in_all_span_batch
                reduce_span_in_all_span_batch[in_batch_idx] = \
                    reduce_span_in_all_span[:action_idx] + [reduce_span] + reduce_span_in_all_span[action_idx + 2:]

                # If a span contains 2 reduced spans, this span will be reduced
                if len(reduce_span) >= 2:
                    assert len(reduce_span) == 2
                    reduce_list.append(in_batch_idx)
                    reduce_span_in_all_span_batch[in_batch_idx][action_idx] = [merged_span]
                    span2repre_batch[in_batch_idx][str(merged_span)] = \
                        hidden[in_batch_idx:in_batch_idx + 1, action_idx]

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
        # pdb.set_trace()
        normalized_entropy = (sum(normalized_entropy) + sum(var_normalized_entropy)) / (
                    len(normalized_entropy) + len(var_normalized_entropy))

        assert relaxed is False

        tree_rl_infos = [normalized_entropy, log_prob, parent_child_spans_batch, span2repre_batch]

        return tree_rl_infos

    def abst_embed(self, hidden, cell, h_x, c_x, in_batch_idx, action_idx):
        # replace the abstrat with certain variable (x1 or x2)
        h_p_new = torch.cat([hidden[in_batch_idx:in_batch_idx + 1, :action_idx],
                             h_x,
                             hidden[in_batch_idx:in_batch_idx + 1, action_idx + 1:]], dim=1)
        c_p_new = torch.cat([cell[in_batch_idx:in_batch_idx + 1, :action_idx],
                             c_x,
                             cell[in_batch_idx:in_batch_idx + 1, action_idx + 1:]], dim=1)
        h_batch_new = torch.cat([hidden[:in_batch_idx],
                                 h_p_new,
                                 hidden[in_batch_idx + 1:]], dim=0)
        c_batch_new = torch.cat([cell[:in_batch_idx],
                                 c_p_new,
                                 cell[in_batch_idx + 1:]], dim=0)

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


class Solver(nn.Module):
    # To compose bottom abstractions with rules
    def __init__(self, hidden_dim, output_lang,
                 entity_list=[], predicate_list=[]):
        super().__init__()

        # 0: ( E1 , E2 )
        # 1: exclude ( E1 , E2 )
        # 2: intersection ( E1 , E2 )
        self.semantic_E_E = nn.Linear(in_features=hidden_dim, out_features=3)

        # 0: ( P1 , P2 )
        # 1: ( P2 , P1 )
        self.semantic_P_P = nn.Linear(in_features=hidden_dim, out_features=2)

        self.semantic_E_P = nn.Linear(in_features=hidden_dim, out_features=1)

        self.hidden_dim = hidden_dim
        self.output_lang = output_lang

        self.entity_list = entity_list
        self.predicate_list = predicate_list

    def forward(self, pair, span2output_token, parent_child_spans, span2repre,
                relaxed=False, tau_weights=None, straight_through=False, noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None):

        # TODO: maybe have bugs when reduce_span is empty
        # pdb.set_trace()
        span2semantic = self.init_semantic_class(span2output_token)
        # pdb.set_trace()

        semantic_normalized_entropy = []
        semantic_log_prob = []

        noise_i = None
        eval_swr_actions_i = None

        for parent_child_span in parent_child_spans:
            parent_span = parent_child_span[0]
            child0_span = parent_child_span[1][0]
            child1_span = parent_child_span[1][1]
            assert child0_span[1] < child1_span[0]
            child0_semantic = span2semantic[str(child0_span)]
            child1_semantic = span2semantic[str(child1_span)]
            # pdb.set_trace()
            cat_distr, _, actions_i, parent_semantic = self.semantic_merge(child0_semantic, child1_semantic,
                                                                           span2repre[str(parent_span)],
                                                                           relaxed, tau_weights,
                                                                           straight_through, noise_i,
                                                                           eval_swr_actions_i)
            span2semantic[str(parent_span)] = parent_semantic

            if cat_distr is not None:
                semantic_normalized_entropy.append(cat_distr.normalized_entropy)
                semantic_log_prob.append(-cat_distr.log_prob(actions_i))

            # pdb.set_trace()

        # pdb.set_trace()

        if len(semantic_normalized_entropy) > 0:
            normalized_entropy = sum(semantic_normalized_entropy) / len(semantic_normalized_entropy)
        else:
            normalized_entropy = 0.
        semantic_log_prob = sum(semantic_log_prob)

        if parent_child_spans == []:
            assert len(span2output_token) == 1
            parent_span = span2output_token[0][0]

        semantic_rl_infos = [normalized_entropy, semantic_log_prob, span2semantic, parent_span]


        return semantic_rl_infos

    def init_semantic_class(self, span2output_token):
        span2semantic = {}
        for span in span2output_token:
            span_position = span[0]
            output_token = span[1]

            assert output_token in self.entity_list + self.predicate_list

            if output_token in self.entity_list:
                span_semantic = E(output_token)
            else:
                span_semantic = P(output_token)

            span2semantic[str(span_position)] = span_semantic

        return span2semantic

    def semantic_merge(self, child0_semantic, child1_semantic, parent_repre,
                       relaxed, tau_weights, straight_through, gumbel_noise, ev_swr_actions):
        if isinstance(child0_semantic, E) and isinstance(child1_semantic, E):
            semantic_score = self.semantic_E_E(parent_repre)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                        straight_through,
                                                        gumbel_noise)
            if actions[0, 0] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_and(child1_semantic)
            elif actions[0, 1] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_ex(child1_semantic)
            else:
                assert actions[0, 2] == 1
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_in(child1_semantic)

        elif isinstance(child0_semantic, P) and isinstance(child1_semantic, P):
            semantic_score = self.semantic_P_P(parent_repre)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                        straight_through,
                                                        gumbel_noise)

            if actions[0, 0] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P_order(child1_semantic)
            else:
                assert actions[0, 1] == 1
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P_reorder(child1_semantic)

        else:
            semantic_score = self.semantic_E_P(parent_repre)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                        straight_through,
                                                        gumbel_noise)
            assert actions[0, 0] == 1
            if isinstance(child0_semantic, P) and isinstance(child1_semantic, E):
                parent_semantic = copy.deepcopy(child1_semantic)
                parent_semantic.add_P(child0_semantic)
            else:
                assert isinstance(child1_semantic, P) and isinstance(child0_semantic, E)
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P(child1_semantic)

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
                 composer_trans_hidden=None,
                 var_normalization=False,
                 input_lang=None, output_lang=None,
                 alignments={}, entity_list=[], predicate_list=[], funql_type={}):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.label_dim = label_dim
        self.alignments = alignments
        self.entity_list = entity_list
        self.predicate_list = predicate_list
        self.funql_type = funql_type
        self.abstractor = BottomAbstrator(input_lang, alignments)
        self.classifier = BottomClassifier(word_dim, hidden_dim, vocab_size, "bi_lstm_transformation",
                                           composer_trans_hidden,
                                           input_lang, output_lang, alignments)
        self.composer = BottomUpTreeComposer(word_dim, hidden_dim, vocab_size, "no_transformation",
                                             composer_trans_hidden, input_lang, output_lang,
                                             alignments=alignments,
                                             entity_list=entity_list,
                                             predicate_list=predicate_list,
                                             funql_type=funql_type)
        self.solver = Solver(hidden_dim, output_lang,
                             entity_list=entity_list,
                             predicate_list=predicate_list)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.is_test = False

        self.case = None

    # TODO: change the paremeters
    def get_high_parameters(self):
        return list(chain(self.classifier.parameters(), ))

    def get_low_parameters(self):
        return list(chain(self.composer.parameters(), self.solver.parameters()))

    def forward(self, pair, x, sample_num, epoch=0):

        batch_forward_info, pred_chain, label_chain = self._forward(
            pair, x, sample_num, epoch)

        return batch_forward_info, pred_chain, label_chain

    def _forward(self, pair, x, sample_num, epoch):
        assert x.size(1) > 1

        bottom_span, bottom_span2cand_funql = self.abstractor(x, pair)

        classify_info = self.classifier(x, pair, bottom_span, bottom_span2cand_funql, sample_num)
        class_normalized_entropy_batch, class_log_prob_batch, bottom_span_batch, span2output_token_batch = classify_info

        tree_rl_info = self.composer(pair, x, bottom_span_batch, span2output_token_batch)
        tree_normalized_entropy_batch, tree_log_prob_batch, parent_child_spans_batch, span2repre_batch = tree_rl_info

        batch_forward_info = []

        for in_batch_idx in range(sample_num):
            parent_child_spans = parent_child_spans_batch[in_batch_idx]
            span2repre = span2repre_batch[in_batch_idx]
            span2output_token = []
            for span_info in span2output_token_batch[in_batch_idx]:
                if span_info[1] != '':
                    span2output_token.append(span_info)

            semantic_rl_infos = self.solver(pair, span2output_token, parent_child_spans, span2repre)

            semantic_normalized_entropy, semantic_log_prob, span2semantic, end_span = semantic_rl_infos

            pred_chain = self.translate(span2semantic[str(end_span)])
            # pdb.set_trace()
            use_fix_most = True
            if use_fix_most is True:
                pred_chain = self.fix_most(pred_chain)
            # pdb.set_trace()
            label_chain = pair[1].split()
            reward, bp_reward = self.get_reward(pred_chain, label_chain)

            # pdb.set_trace()
            if USE_CUDA:
                normalized_entropy = torch.tensor([0.]).cuda() + \
                                     class_normalized_entropy_batch[in_batch_idx] + \
                                     tree_normalized_entropy_batch[in_batch_idx] + \
                                     semantic_normalized_entropy

                log_prob = torch.tensor([0.]).cuda() + \
                           class_log_prob_batch[in_batch_idx] + \
                           tree_log_prob_batch[in_batch_idx] + \
                           semantic_log_prob
            else:
                normalized_entropy = torch.tensor([0.]) + \
                                     class_normalized_entropy_batch[in_batch_idx] + \
                                     tree_normalized_entropy_batch[in_batch_idx] + \
                                     semantic_normalized_entropy

                log_prob = torch.tensor([0.]) + \
                           class_log_prob_batch[in_batch_idx] + \
                           tree_log_prob_batch[in_batch_idx] + \
                           semantic_log_prob

            batch_forward_info.append([normalized_entropy, log_prob, reward, bp_reward])

        # pdb.set_trace()

        # if pair[0] == 'what is the population density of var0':
        #     pdb.set_trace()

        return batch_forward_info, pred_chain, label_chain

    def get_reward(self, pred_chain_ori, label_chain_ori):
        pred_chain = [token for token in pred_chain_ori if token not in ['(', ')', ',']]
        label_chain = [token for token in label_chain_ori if token not in ['(', ')', ',']]
        pred_len = len(pred_chain)
        label_len = len(label_chain)
        max_com_len = 0
        for pred_idx in range(pred_len):
            for label_idx in range(label_len):
                com_len = 0
                if pred_chain[pred_idx] != label_chain[label_idx]:
                    continue
                else:
                    com_len += 1
                    right_hand_length = min(pred_len - pred_idx - 1, label_len - label_idx - 1)
                    for right_hand_idx in range(right_hand_length):
                        if pred_chain[pred_idx + right_hand_idx + 1] == label_chain[label_idx + right_hand_idx + 1]:
                            com_len += 1
                            continue
                        else:
                            break
                    if com_len > max_com_len:
                        max_com_len = com_len

        reward = max_com_len / (pred_len + label_len - max_com_len)

        assert reward <= 1.
        # if reward == 1.:
        #     try:
        #         assert pred_chain_ori == label_chain_ori
        #     except:
        #         pdb.set_trace()

        bottom_reward = len(set(pred_chain+label_chain)) / (len(set(pred_chain))+len(set(label_chain)))

        # if set(pred_chain) == set(label_chain):
        #     reward += 1.

        return reward, reward + 0.1 * bottom_reward

    # Translate a nest class into a list
    def translate(self, semantic):
        # pdb.set_trace()
        if isinstance(semantic, P):
            flat_chain = semantic.predicate + ['all'] + semantic.postfix
        else:
            assert isinstance(semantic, E)
            flat_chain = semantic.entity

        return flat_chain

    def fix_most(self, pred_chain):
        if pred_chain.count('most') == 1:
            most_idx = pred_chain.index('most')
            pred_chain = pred_chain[0:most_idx-4] + pred_chain[most_idx:most_idx+2] + \
                         pred_chain[most_idx-4:most_idx] + pred_chain[most_idx+2:]
        return pred_chain

