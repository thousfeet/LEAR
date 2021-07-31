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
    # 0: in
    # 1: on
    # 2: beside
    # 3: recipient theme (E E)
    # 4: theme recipient (E to E)
    # 5: theme agent (E by E)
    # 6: recipient agent (to E by E)
    def __init__(self, entity):
        # [in_word/on_word/beside_word, agent_word/theme_word/recipient_word, entity]
        self.entity_chain = [[None, None, entity]]
        # entity_class
        self.entity_para = None

    def add_E_in(self, E0):
        left_E = copy.deepcopy(E0)
        left_E.entity_chain[0][0] = 'in'
        self.entity_chain = self.entity_chain + left_E.entity_chain
        self.entity_para = None

    def add_E_on(self, E0):
        left_E = copy.deepcopy(E0)
        left_E.entity_chain[0][0] = 'on'
        self.entity_chain = self.entity_chain + left_E.entity_chain
        self.entity_para = None

    def add_E_beside(self, E0):
        left_E = copy.deepcopy(E0)
        left_E.entity_chain[0][0] = 'beside'
        self.entity_chain = self.entity_chain + left_E.entity_chain
        self.entity_para = None

    def add_E_recipient_theme(self, E0):
        self.entity_chain[0][1] = 'recipient'
        para_E = copy.deepcopy(E0)
        para_E.entity_chain[0][1] = 'theme'
        self.entity_para = para_E

    def add_E_theme_recipient(self, E0):
        self.entity_chain[0][1] = 'theme'
        para_E = copy.deepcopy(E0)
        para_E.entity_chain[0][1] = 'recipient'
        self.entity_para = para_E

    def add_E_theme_agent(self, E0):
        self.entity_chain[0][1] = 'theme'
        para_E = copy.deepcopy(E0)
        para_E.entity_chain[0][1] = 'agent'
        self.entity_para = para_E

    def add_E_recipient_agent(self, E0):
        self.entity_chain[0][1] = 'recipient'
        para_E = copy.deepcopy(E0)
        para_E.entity_chain[0][1] = 'agent'
        self.entity_para = para_E


class P:
    def __init__(self, action):
        # [ccomp_word/xcomp_word, agent_entity_class, theme_entity_class, recipient_entity_class, action_word]
        self.action_chain = [[None, None, None, None, action]]

        self.has_agent = False
        self.has_theme = False
        self.has_recipient = False

        self.is_full = False

    def add_E_agent(self, E0):
        assert E0.entity_para is None

        E_add = copy.deepcopy(E0)
        E_add.entity_chain[0][1] = 'agent'
        self.action_chain[0][1] = E_add

    def add_E_theme(self, E0):
        assert E0.entity_para is None

        E_add = copy.deepcopy(E0)
        E_add.entity_chain[0][1] = 'theme'
        self.action_chain[0][2] = E_add

    def add_E_recipient(self, E0):
        assert E0.entity_para is None

        E_add = copy.deepcopy(E0)
        E_add.entity_chain[0][1] = 'recipient'
        self.action_chain[0][3] = E_add

    def add_E_recipient_theme(self, E0):
        assert E0.entity_chain[0][1] == 'recipient'
        assert E0.entity_para.entity_chain[0][1] == 'theme'

        E_add_0 = copy.deepcopy(E0)
        E_add_1 = copy.deepcopy(E0.entity_para)

        E_add_0.entity_para = None
        E_add_1.entity_para = None

        self.action_chain[0][3] = E_add_0
        self.action_chain[0][2] = E_add_1

    def add_E_theme_recipient(self, E0):
        assert E0.entity_chain[0][1] == 'theme'
        assert E0.entity_para.entity_chain[0][1] == 'recipient'

        E_add_0 = copy.deepcopy(E0)
        E_add_1 = copy.deepcopy(E0.entity_para)

        E_add_0.entity_para = None
        E_add_1.entity_para = None

        self.action_chain[0][2] = E_add_0
        self.action_chain[0][3] = E_add_1

    def add_E_theme_agent(self, E0):
        assert E0.entity_chain[0][1] == 'theme'
        assert E0.entity_para.entity_chain[0][1] == 'agent'

        E_add_0 = copy.deepcopy(E0)
        E_add_1 = copy.deepcopy(E0.entity_para)

        E_add_0.entity_para = None
        E_add_1.entity_para = None

        self.action_chain[0][2] = E_add_0
        self.action_chain[0][1] = E_add_1

    def add_E_recipient_agent(self, E0):
        assert E0.entity_chain[0][1] == 'recipient'
        assert E0.entity_para.entity_chain[0][1] == 'agent'

        E_add_0 = copy.deepcopy(E0)
        E_add_1 = copy.deepcopy(E0.entity_para)

        E_add_0.entity_para = None
        E_add_1.entity_para = None

        self.action_chain[0][3] = E_add_0
        self.action_chain[0][1] = E_add_1

    def add_P_ccomp(self, P0):
        P_add = copy.deepcopy(P0)
        P_add.action_chain[0][0] = 'ccomp'

        self.action_chain = self.action_chain + P_add.action_chain

    def add_P_xcomp(self, P0):
        P_add = copy.deepcopy(P0)
        P_add.action_chain[0][0] = 'xcomp'

        self.action_chain = self.action_chain + P_add.action_chain


class BottomAbstrator(nn.Module):
    # To make bottom abstractions such as 'M0' and 'executive produce'
    def __init__(self, alignment_idx):
        super().__init__()
        self.alignment_idx = alignment_idx

    def forward(self, x):
        bottom_span = []
        for position, token in enumerate(x[0]):
            if token.item() in self.alignment_idx:
                bottom_span.append([position, position])
            else:
                continue

        return bottom_span


class BottomClassifier(nn.Module):
    # To classify bottom abstractions
    def __init__(self, output_lang, alignments_idx):
        super().__init__()
        self.output_lang = output_lang
        self.alignments_idx = alignments_idx

    def forward(self, x, bottom_span):
        span2output_token = []
        for span in bottom_span:
            position = span[0]
            input_idx = x[0, position].item()
            output_idx = self.alignments_idx[input_idx]
            assert len(output_idx) == 1
            output_idx = output_idx[0]
            output_token = self.output_lang.index2word[output_idx]
            span2output_token.append([span, output_token])

        return span2output_token


class BottomUpTreeComposer(BinaryTreeBasedModule):
    # To generate a binary tree structure based on bottom abstractions
    def __init__(self, input_dim, hidden_dim, vocab_size, leaf_transformation, trans_hidden_dim, input_lang,
                 output_lang,
                 alignments_idx={}, entity_list=[], caus_predicate_list=[], unac_predicate_list=[],
                 dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.q = nn.Parameter(torch.empty(size=(hidden_dim,), dtype=torch.float32))
        self.var_linear = nn.Linear(in_features=hidden_dim, out_features=2)
        self.hidden_dim = hidden_dim
        self.input_lang = input_lang
        self.output_lang = output_lang

        self.alignments_idx = alignments_idx
        self.entity_list = entity_list
        self.predicate_list = caus_predicate_list + unac_predicate_list

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)

    def forward(self, pair, x, bottom_span_batch, span2output_token_batch,
                relaxed=False, tau_weights=None, straight_through=False, noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None):

        batch_size = len(bottom_span_batch)

        span2variable_batch = [{} for _ in range(batch_size)]

        length_ori = len(x[0])

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
        else:
            single_mask = torch.tensor([[1.]])
            h_x1, c_x1 = self._transform_leafs(self.embd_parser(torch.tensor([[x1_token]])), mask=single_mask)
            h_x2, c_x2 = self._transform_leafs(self.embd_parser(torch.tensor([[x2_token]])), mask=single_mask)

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
                assert span[1] - span[0] == 0

                span2output_token_dict = span2output_token_dict_batch[in_batch_idx]
                token = span2output_token_dict[str(span)]
                assert token in self.entity_list + self.predicate_list

                if token in self.entity_list:
                    h_x, c_x = h_x1, c_x1
                else:
                    h_x, c_x = h_x2, c_x2

                hidden_1_one_batch = torch.cat(
                    [hidden_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     h_x,
                     hidden_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]], dim=1)
                cell_1_one_batch = torch.cat(
                    [cell_1[in_batch_idx:in_batch_idx + 1, :span[0], :],
                     c_x,
                     cell_1[in_batch_idx:in_batch_idx + 1, span[1] + 1:, :]], dim=1)

                hidden_1 = torch.cat([hidden_1[:in_batch_idx], hidden_1_one_batch, hidden_1[in_batch_idx + 1:]], dim=0)
                cell_1 = torch.cat([cell_1[:in_batch_idx], cell_1_one_batch, cell_1[in_batch_idx + 1:]], dim=0)

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

        mask = torch.ones((batch_size, length_ori), dtype=torch.float32)
        if USE_CUDA:
            mask = mask.cuda()

        for i in range(1, x_embedding.shape[1]):
            # pdb.set_trace()
            noise = None
            ev_actions = None

            cat_distr, _, actions, hidden, cell = self._make_step(hidden, cell, mask[:, i:],
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
                        span2variable_batch[in_batch_idx][str(merged_span)] = 'entity'
                    else:
                        assert var_actions[in_batch_idx, 1] == 1
                        h_x, c_x = h_x2, c_x2
                        span2variable_batch[in_batch_idx][str(merged_span)] = 'predicate'
                    # pdb.set_trace()
                    hidden, cell = self.abst_embed(hidden, cell, h_x, c_x, in_batch_idx, action_idx)

                    parent_child_span = [merged_span, reduce_span]
                    parent_child_spans_batch[in_batch_idx].append(parent_child_span)

            reduce_mask = [1 if i in reduce_list else 0 for i in range(batch_size)]
            reduce_mask = torch.tensor(reduce_mask, dtype=torch.float32)
            if USE_CUDA:
                reduce_mask = reduce_mask.cuda()

            normalized_entropy.append(cat_distr.normalized_entropy)
            log_prob.append(-cat_distr.log_prob(actions))

            var_normalized_entropy.append(var_cat_distr.normalized_entropy * reduce_mask)
            var_log_prob.append(-var_cat_distr.log_prob(var_actions) * reduce_mask)

        log_prob = sum(log_prob) + sum(var_log_prob)

        normalized_entropy = (sum(normalized_entropy) + sum(var_normalized_entropy)) / (
                    len(normalized_entropy) + len(var_normalized_entropy))

        assert relaxed is False

        tree_rl_infos = [normalized_entropy, log_prob, parent_child_spans_batch, span2repre_batch]

        return tree_rl_infos, span2variable_batch

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

        return cat_distr, gumbel_noise, actions, h_p, c_p

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
                 entity_list=[], caus_predicate_list=[], unac_predicate_list=[]):
        super().__init__()

        # 0: in
        # 1: on
        # 2: beside
        # 3: recipient theme (E E)
        # 4: theme recipient (E to E)
        # 5: theme agent (E by E)
        # 6: recipient agent (to E by E)
        self.semantic_E_E = nn.Linear(in_features=hidden_dim, out_features=7)

        # 0: agent
        # 1: theme
        # 2: recipient
        self.semantic_P_E = nn.Linear(in_features=hidden_dim, out_features=3)

        # 0: ccomp
        # 1: xcomp
        self.semantic_P_P = nn.Linear(in_features=hidden_dim, out_features=2)

        self.hidden_dim = hidden_dim
        self.output_lang = output_lang

        self.entity_list = entity_list
        self.caus_predicate_list = caus_predicate_list
        self.unac_predicate_list = unac_predicate_list
        self.predicate_list = caus_predicate_list + unac_predicate_list

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

        normalized_entropy = sum(semantic_normalized_entropy) / len(semantic_normalized_entropy)
        semantic_log_prob = sum(semantic_log_prob)

        assert relaxed is False

        final_semantic = copy.deepcopy(span2semantic[str(parent_span)])
        for action_idx in range(len(final_semantic.action_chain)):
            action_info = final_semantic.action_chain[action_idx]
            if action_info[1] is not None and action_info[2] is None and action_info[3] is None:
                agent_class = copy.deepcopy(action_info[1])
                assert agent_class.entity_chain[0][1] == 'agent'
                action = action_info[4]

                # in gen
                # A donut on the bed rolled .
                # * bed ( x _ 4 ) ; donut ( x _ 1 ) AND donut . nmod . on ( x _ 1 , x _ 4 ) AND roll . theme ( x _ 5 , x _ 1 )
                # The girl beside a table rolled .
                # * girl ( x _ 1 ) ; girl . nmod . beside ( x _ 1 , x _ 4 ) AND table ( x _ 4 ) AND roll . agent ( x _ 5 , x _ 1 )

                # in train
                # Victoria hoped that a girl rolled .
                # hope . agent ( x _ 1 , Victoria ) AND hope . ccomp ( x _ 1 , x _ 5 ) AND girl ( x _ 4 ) AND roll . theme ( x _ 5 , x _ 4 )

                if action in self.unac_predicate_list and len(agent_class.entity_chain) == 1:
                    # if action in self.unac_predicate_list:
                    agent_class.entity_chain[0][1] = 'theme'
                    final_semantic.action_chain[action_idx][1] = None
                    final_semantic.action_chain[action_idx][2] = agent_class
                    final_semantic.has_agent = False
                    final_semantic.has_theme = True

        span2semantic[str(parent_span)] = final_semantic

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
                parent_semantic.add_E_in(child1_semantic)
            elif actions[0, 1] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_on(child1_semantic)
            elif actions[0, 2] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_beside(child1_semantic)
            elif actions[0, 3] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_recipient_theme(child1_semantic)
            elif actions[0, 4] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_theme_recipient(child1_semantic)
            elif actions[0, 5] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_theme_agent(child1_semantic)
            else:
                assert actions[0, 6] == 1
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_E_recipient_agent(child1_semantic)

        elif isinstance(child0_semantic, P) and isinstance(child1_semantic, P):
            semantic_score = self.semantic_P_P(parent_repre)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                        straight_through,
                                                        gumbel_noise)

            if actions[0, 0] == 1:
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P_ccomp(child1_semantic)
            else:
                assert actions[0, 1] == 1
                parent_semantic = copy.deepcopy(child0_semantic)
                parent_semantic.add_P_xcomp(child1_semantic)

        else:
            semantic_score = self.semantic_P_E(parent_repre)
            semantic_mask = torch.ones_like(semantic_score)

            if isinstance(child0_semantic, P) and isinstance(child1_semantic, E):
                parent_semantic = copy.deepcopy(child0_semantic)
                E_semantic = copy.deepcopy(child1_semantic)
            else:
                assert isinstance(child1_semantic, P) and isinstance(child0_semantic, E)
                parent_semantic = copy.deepcopy(child1_semantic)
                E_semantic = copy.deepcopy(child0_semantic)

            if E_semantic.entity_para is not None:
                if E_semantic.entity_chain[0][1] == 'recipient' and \
                        E_semantic.entity_para.entity_chain[0][1] == 'theme':
                    parent_semantic.add_E_recipient_theme(E_semantic)
                    parent_semantic.has_recipient = True
                    parent_semantic.has_theme = True
                elif E_semantic.entity_chain[0][1] == 'theme' and \
                        E_semantic.entity_para.entity_chain[0][1] == 'recipient':
                    parent_semantic.add_E_theme_recipient(E_semantic)
                    parent_semantic.has_theme = True
                    parent_semantic.has_recipient = True
                elif E_semantic.entity_chain[0][1] == 'theme' and \
                        E_semantic.entity_para.entity_chain[0][1] == 'agent':
                    parent_semantic.add_E_theme_agent(E_semantic)
                    parent_semantic.has_theme = True
                    parent_semantic.has_agent = True
                else:
                    assert E_semantic.entity_chain[0][1] == 'recipient' and \
                           E_semantic.entity_para.entity_chain[0][1] == 'agent'
                    parent_semantic.add_E_recipient_agent(E_semantic)
                    parent_semantic.has_recipient = True
                    parent_semantic.has_agent = True

                cat_distr, actions, gumbel_noise = None, None, None

            else:
                if not parent_semantic.is_full:
                    assert parent_semantic.has_agent is False or \
                           parent_semantic.has_theme is False or \
                           parent_semantic.has_recipient is False
                    if parent_semantic.has_agent is True:
                        semantic_mask[0, 0] = 0
                    if parent_semantic.has_theme is True:
                        semantic_mask[0, 1] = 0
                    if parent_semantic.has_recipient is True:
                        semantic_mask[0, 2] = 0

                cat_distr = Categorical(semantic_score, semantic_mask)
                actions, gumbel_noise = self._sample_action(cat_distr, semantic_mask, relaxed, tau_weights,
                                                            straight_through,
                                                            gumbel_noise)
                if actions[0, 0] == 1:
                    parent_semantic.add_E_agent(E_semantic)
                    parent_semantic.has_agent = True
                elif actions[0, 1] == 1:
                    parent_semantic.add_E_theme(E_semantic)
                    parent_semantic.has_theme = True
                else:
                    assert actions[0, 2] == 1
                    parent_semantic.add_E_recipient(E_semantic)
                    parent_semantic.has_recipient = True

            if parent_semantic.has_agent is True and \
                    parent_semantic.has_theme is True and \
                    parent_semantic.has_recipient is True:
                parent_semantic.is_full = True

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
                 alignments_idx={}, entity_list=[], caus_predicate_list=[], unac_predicate_list=[]):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.label_dim = label_dim
        self.alignments_idx = alignments_idx
        self.entity_list = entity_list
        self.caus_predicate_list = caus_predicate_list
        self.unac_predicate_list = unac_predicate_list
        self.abstractor = BottomAbstrator(alignments_idx)
        self.classifier = BottomClassifier(output_lang, alignments_idx)
        self.composer = BottomUpTreeComposer(word_dim, hidden_dim, vocab_size, "no_transformation",
                                             composer_trans_hidden, input_lang, output_lang,
                                             alignments_idx=alignments_idx,
                                             entity_list=entity_list,
                                             caus_predicate_list=caus_predicate_list,
                                             unac_predicate_list=unac_predicate_list)
        self.solver = Solver(hidden_dim, output_lang,
                             entity_list=entity_list,
                             caus_predicate_list=caus_predicate_list,
                             unac_predicate_list=unac_predicate_list)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.is_test = False

        self.case = None

    # TODO: change the paremeters
    def get_high_parameters(self):
        return list(chain(self.composer.parameters()))

    def get_low_parameters(self):
        return list(chain(self.solver.parameters()))

    def case_study(self, reduced_output, token_list):
        token_list = token_list[0].split()
        sentence_len = len(token_list)
        sentences_num = len(reduced_output)
        res = []
        for i in range(sentences_num):
            sentence = reduced_output[i]
            dic = {}
            min_num, max_num = sentence_len + 1, 0
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

    def forward(self, pair, x, sample_num, is_test=False, epoch=None, debug_info=None):
        self.is_test = is_test

        batch_forward_info, pred_chain, label_chain = self._forward(
            pair, x, sample_num, epoch, is_test, debug_info=debug_info)
        return batch_forward_info, pred_chain, label_chain

    def _forward(self, pair, x, sample_num, epoch, is_test, debug_info):
        assert x.size(1) > 1

        bottom_span = self.abstractor(x)

        span2output_token = self.classifier(x, bottom_span)

        bottom_span_batch = [bottom_span for _ in range(sample_num)]
        span2output_token_batch = [span2output_token for _ in range(sample_num)]

        tree_rl_info, span2variable_batch = self.composer(pair, x, bottom_span_batch, span2output_token_batch)
        tree_normalized_entropy, tree_log_prob, parent_child_spans_batch, span2repre_batch = tree_rl_info

        batch_forward_info = []

        for in_batch_idx in range(sample_num):
            parent_child_spans = parent_child_spans_batch[in_batch_idx]
            span2repre = span2repre_batch[in_batch_idx]
            span2output_token = span2output_token_batch[in_batch_idx]

            semantic_rl_infos = self.solver(pair, span2output_token, parent_child_spans, span2repre)

            semantic_normalized_entropy, semantic_log_prob, span2semantic, end_span = semantic_rl_infos

            pred_chain = self.translate(span2semantic[str(end_span)])
            # pdb.set_trace()
            label_chain = self.process_output(pair[1])
            reward = self.get_reward(pred_chain, label_chain)

            # pdb.set_trace()

            normalized_entropy = tree_normalized_entropy[in_batch_idx] + \
                                 semantic_normalized_entropy

            log_prob = tree_log_prob[in_batch_idx] + \
                       semantic_log_prob

            batch_forward_info.append([normalized_entropy, log_prob, reward])

        # pdb.set_trace()

        return batch_forward_info, pred_chain, label_chain

    def get_reward(self, pred_chain, label_chain):
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
        if reward == 1.:
            assert pred_chain == label_chain

        return reward

    def get_entity_chain(self, semantic):
        if semantic is None:
            return ['None']

        assert isinstance(semantic, E)
        flat_chain = []
        # [in_word / on_word / beside_word, agent_word / theme_word / recipient_word, entity]
        for entity_idx, entity_info in enumerate(semantic.entity_chain):
            if entity_idx == 0:
                flat_entity_chain = [entity_info[2]]
            else:
                assert entity_info[0] is not None
                flat_entity_chain = [entity_info[0], entity_info[2]]
            flat_chain = flat_chain + flat_entity_chain

        return flat_chain

    # Translate a nest class into a list
    def translate(self, semantic):
        assert isinstance(semantic, P)
        flat_chain = []

        for action_idx, action_info in enumerate(semantic.action_chain):
            # [ccomp_word / xcomp_word, agent_entity_class, theme_entity_class, recipient_entity_class, action_word]
            if action_idx == 0:
                flat_action_chain = [action_info[4]]
            else:
                assert action_info[0] is not None
                flat_action_chain = [action_info[0], action_info[4]]

            if action_info[0] == 'xcomp':
                agent_chain = reserve_agent_chain
            else:
                agent_chain = self.get_entity_chain(action_info[1])
            theme_chain = self.get_entity_chain(action_info[2])
            recipient_chain = self.get_entity_chain(action_info[3])
            flat_action_chain = flat_action_chain + agent_chain + theme_chain + recipient_chain
            reserve_agent_chain = copy.deepcopy(agent_chain)

            flat_chain = flat_chain + flat_action_chain

        return flat_chain

    def process_output(self, output):
        output_process = output.replace(" and ", " AND ")
        output_process = output_process.replace(" ccomp ", " CCOMP ")
        output_process = output_process.replace(" xcomp ", " XCOMP ")
        output_process = output_process.replace(" nmod ", " NMOD ")
        output_process = output_process.replace(" agent ", " AGENT ")
        output_process = output_process.replace(" theme ", " THEME ")
        output_process = output_process.replace(" recipient ", " RECIPIENT ")
        output_process = output_process.replace(" x ", " X ")
        output_process = output_process.replace(" ", "")
        output_process = output_process.replace("*", "")
        output_process = re.split(";|AND", output_process)

        object_stack = {}
        output_process_no_obj = copy.deepcopy(output_process)
        for function in output_process:
            if '.' not in function:  # entity
                function_para = re.split("\(|\)", function)[:-1]
                assert len(function_para) == 2
                object, object_index = function_para
                assert object_index not in object_stack
                object_stack[object_index] = object
                output_process_no_obj.remove(function)
            else:
                continue

        output_process_no_obj_no_pr = copy.deepcopy(output_process_no_obj)
        output_process_no_obj.reverse()
        for function in output_process_no_obj:
            if 'NMOD' in function:  # pr
                function_para = re.split("\(|\)|\.|,", function)[:-1]
                assert len(function_para) == 5
                pr_word = function_para[2]
                pr_subject_index = function_para[3]
                pr_subject = object_stack[pr_subject_index]
                # assert for 2 things:
                # first, re-ensure subject
                # second, ensure subject in object_stack has no other pr
                # the second means that the composition process is from right to left
                # (implemented by output_process_no_obj.reverse())
                # this is to ensure the left subject has all corresponding objects
                assert pr_subject == function_para[0]
                pr_object_index = function_para[4]
                # ensure pr_object_index is an index but not like Emma
                assert 'X' in pr_object_index
                # this can also ensure the left subject has all corresponding objects
                # (as the object is popped, it cannot occur as subject again)
                pr_object = object_stack.pop(pr_object_index)
                new_subject = pr_subject + " " + pr_word + " " + pr_object
                object_stack[pr_subject_index] = new_subject
                output_process_no_obj_no_pr.remove(function)
            else:
                continue

        output_process_only_comp = copy.deepcopy(output_process_no_obj_no_pr)
        predicate_stack = {}
        predicate_object_stack = {}
        for function in output_process_no_obj_no_pr:
            if 'COMP' not in function:
                function_para = re.split("\(|\)|\.|,", function)[:-1]
                assert len(function_para) == 4
                predicate, object_type, predicate_index, object_index = function_para
                if 'X' not in object_index:
                    object = object_index
                else:
                    object = object_stack[object_index]
                if predicate_index not in predicate_stack:
                    predicate_stack[predicate_index] = predicate
                    predicate_object_stack[predicate_index] = {"AGENT": "None",
                                                               "THEME": "None",
                                                               "RECIPIENT": "None",
                                                               }
                predicate_object_stack[predicate_index][object_type] = object
                output_process_only_comp.remove(function)
            else:
                continue

        for predicate_key in predicate_object_stack:
            if predicate_object_stack[predicate_key]["RECIPIENT"] != "None" \
                    and predicate_object_stack[predicate_key]["THEME"] == "None":
                pdb.set_trace()

        final_chain = []
        if output_process_only_comp:
            output_process_only_comp.reverse()
            for function in output_process_only_comp:
                function_para = re.split("\(|\)|\.|,", function)[:-1]
                assert len(function_para) == 4
                compose_type = function_para[1]
                assert compose_type in ["CCOMP", "XCOMP"]
                subject_index = function_para[2]
                object_index = function_para[3]
                subject = predicate_stack[subject_index]
                subject_contain = predicate_object_stack[subject_index]
                try:
                    assert subject == function_para[0]
                except:
                    pdb.set_trace()
                # ensure object will not occur as subject
                object = predicate_stack.pop(object_index)
                object_contain = predicate_object_stack.pop(object_index)
                if compose_type == 'XCOMP':
                    assert subject_contain["AGENT"] == object_contain["AGENT"]
                if final_chain == []:
                    final_chain = [[subject, subject_contain], compose_type, [object, object_contain]]
                else:
                    final_chain = [[subject, subject_contain], compose_type] + final_chain

        else:
            assert len(predicate_stack) == 1
            for predicate_index in predicate_stack:
                subject = predicate_stack[predicate_index]
                subject_contain = predicate_object_stack[predicate_index]
                final_chain = [[subject, subject_contain]]

        flat_chain = []

        for pred_info in final_chain:
            if isinstance(pred_info, list):
                action = pred_info[0]
                flat_chain.append(action)

                action_info = pred_info[1]
                agent = action_info['AGENT']
                agent = agent.split()
                flat_chain = flat_chain + agent

                theme = action_info['THEME']
                theme = theme.split()
                flat_chain = flat_chain + theme

                recipient = action_info['RECIPIENT']
                recipient = recipient.split()
                flat_chain = flat_chain + recipient

            else:
                assert pred_info in ["CCOMP", "XCOMP"]
                flat_chain.append(pred_info.lower())

        return flat_chain
