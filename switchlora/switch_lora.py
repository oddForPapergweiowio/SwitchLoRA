import loralib as _lora
import torch.nn.functional as _F
from . import lora_utils
import torch as _torch
import torch.nn as _nn
import math as _math
import numpy.random as _random
import os as _os
import json as _json
from typing import Mapping as _Mapping, Any as _Any
from functools import partial as _partial
from torch.optim.lr_scheduler import MultiplicativeLR as _MultiplicativeLR
from torch.optim.lr_scheduler import ChainedScheduler as _ChainedScheduler
import numpy as _np
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

# Min step adam optimizer needed to warm-up
_ADAM_WARM_STEP = 5
# Type to descend switch_lora interval
_SWITCH_DESCEND_TYPE = "exponential"
# _SWITCH_DESCEND_TYPE = "Z"
# whether to drop candidates
_DROP_CANDIDATES = False
_FIX_SWITCH_LORA_INTERVAL = False
_ZERO_INIT_B = False
_CANDIDATES_DROP_RATE = 0.
_SWITCH_LORA_INTERVAL = 40
_ADJUST_LORA_SCHEDULE = False
_ZERO_SWITCH_STATE = False
_ZERO_SWITCH_STEP_STATE = False
_ZERO_ALL_STATE = False
_ADD_WEIGHTED_RANK = False


def add_parse_switch_lora_args(parser):
    """
    Recommended arguments for switch_lora
    @param parser: parser = argparse.ArgumentParser()
    """
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=128)

    parser.add_argument("--switch_lora_drop", type=float, default=0,
                        help="Rate of candidates to drop.")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--switch_lora_interval", type=int, default=40)
    parser.add_argument("--adjust_lora_schedule", action='store_true')
    parser.add_argument("--zero_switch_state", action='store_true')
    parser.add_argument("--zero_switch_step_state", action='store_true')
    parser.add_argument("--zero_all_state", action='store_true')
    parser.add_argument("--add_weighted_rank", action='store_true')
    parser.add_argument("--fix_switch_lora_interval", action='store_true',
                        help="Whether to fix switch lora interval.")
    parser.add_argument("--switch_lora_descent_rate", type=float, default=0.995)
    parser.add_argument("--adam_warm_step", type=int, default=5,
                        help="Min step adam optimizer needed to warm-up. Switched LoRA will be fixed in this step. Set to -1 means no warm-up(Do not use 0)")
    parser.add_argument("--switch_descend_type", type=str, default="exponential",
                        help="Type of descend rate. Z or exponential.")
    parser.add_argument("--drop_switch_lora_candidates", action='store_true',
                        help="Whether to drop candidates with steps going.")
    parser.add_argument("--zero_init_B", action='store_true')
    parser.add_argument("--init_lora_type", type=str, default=None,
                        help="Set to origin_lora to use origin LoRA initialization method.")
    parser.add_argument("--switch_lora", action='store_true',
                        help="Use switched LoRA which will overlap --lora option.")
    parser.add_argument("--cal_delta_norm", action='store_true')
    parser.add_argument("--lora_scheduler", action='store_true')
    parser.add_argument("--change_lora_lr", action='store_true')
    parser.add_argument("--quantize", default=None, type=str, choices=[None, "4bit", "8bit"])
    parser.add_argument("--use_double_quant", action='store_true')


def set_hyper_args(args):
    global _ADAM_WARM_STEP
    _ADAM_WARM_STEP = args.adam_warm_step
    global _SWITCH_DESCEND_TYPE
    _SWITCH_DESCEND_TYPE = args.switch_descend_type
    global _DROP_CANDIDATES
    _DROP_CANDIDATES = args.drop_switch_lora_candidates
    global _FIX_SWITCH_LORA_INTERVAL
    _FIX_SWITCH_LORA_INTERVAL = args.fix_switch_lora_interval
    global _CANDIDATES_DROP_RATE
    _CANDIDATES_DROP_RATE = args.switch_lora_drop
    global _SWITCH_LORA_INTERVAL
    _SWITCH_LORA_INTERVAL = args.switch_lora_interval
    global _ADJUST_LORA_SCHEDULE
    _ADJUST_LORA_SCHEDULE = args.adjust_lora_schedule
    global _ZERO_SWITCH_STATE
    _ZERO_SWITCH_STATE = args.zero_switch_state
    global _ZERO_SWITCH_STEP_STATE
    _ZERO_SWITCH_STEP_STATE = args.zero_switch_step_state
    global _ZERO_ALL_STATE
    _ZERO_ALL_STATE = args.zero_all_state
    global _ADD_WEIGHTED_RANK
    _ADD_WEIGHTED_RANK = args.add_weighted_rank
    global _ZERO_INIT_B
    _ZERO_INIT_B = args.zero_init_B
    lora_utils.CAL_DELTA_NORM = args.cal_delta_norm
    lora_utils.set_init_lora_method(args.init_lora_type)


@_torch.no_grad()
def _correct_switched_lora(model):
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        for i, step in enumerate(layer.fixed_A_steps):
            if step > 0:
                # lora_A is split by line index
                layer.lora_A.grad[i, :] = 0
                layer.fixed_A_steps[i] -= 1
            elif step < 0:
                layer.fixed_A_steps[i] = 0

        for i, step in enumerate(layer.fixed_B_steps):
            if step > 0:
                # lora_B is split by column index
                layer.lora_B.grad[:, i] = 0
                layer.fixed_B_steps[i] -= 1
            elif step < 0:
                layer.fixed_B_steps[i] = 0


def _init_orthogonal_lora(mat_size: int, init_size: float, rank: int, src_mat, candidate_num: int):
    """

    :param mat_size: max of matrix row and column size
    :param init_size: the average value of elements in the matrix
    :return: orthogonal vectors list whose size is mat_size and the length of vector is mat_size too
    """
    # obtain candidates list
    # matrix = _torch.randn(mat_size, mat_size, device=src_mat.device, dtype=src_mat.dtype)
    # q, r = _torch.linalg.qr(matrix)
    # q *= init_size

    # TODO: check orthogonal or random is better
    # candidates_list = list(_torch.chunk(q, mat_size, dim=0))
    # candidates_list = [candidates_list[i] for i in len(candidates_list) if i < candidate_num]
    candidates_list = [src_mat.new_zeros(mat_size) for _ in range(candidate_num)]
    for candidate in candidates_list:
        candidate.uniform_(-init_size, init_size)

    # init candidates list
    candidates_weight = [1. / rank for _ in range(candidate_num)]
    selected_indices = _random.choice(list(range(candidate_num)), size=rank, replace=False)
    return candidates_list, candidates_weight, selected_indices


def _zero_lora_states(model, optimizer):
    if not _ZERO_SWITCH_STATE:
        return
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        for i, step in enumerate(layer.fixed_A_steps):
            if step == _ADAM_WARM_STEP or step < 0:
                state = optimizer.state[layer.lora_A]
                if "exp_avg" not in state:
                    continue
                if _ZERO_ALL_STATE:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                    if _ZERO_SWITCH_STEP_STATE:
                        state["step"].zero_()
                else:
                    state["exp_avg"][i, :] = 0
                    state["exp_avg_sq"][i, :] = 0
                    if _ZERO_SWITCH_STEP_STATE:
                        state["step"][i, :] = 0

        for i, step in enumerate(layer.fixed_B_steps):
            if step == _ADAM_WARM_STEP or step < 0:
                state = optimizer.state[layer.lora_B]
                if "exp_avg" not in state:
                    continue
                if _ZERO_ALL_STATE:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                    if _ZERO_SWITCH_STEP_STATE:
                        state["step"].zero_()
                else:
                    state["exp_avg"][:, i] = 0
                    state["exp_avg_sq"][:, i] = 0
                    if _ZERO_SWITCH_STEP_STATE:
                        state["step"][:, i] = 0


def _get_lora_schedule(global_step,
                       base_interval,
                       expect_switch_descend_step):
    ratio = 1.
    if _FIX_SWITCH_LORA_INTERVAL:
        interval = base_interval
    else:
        interval = base_interval / _get_switch_rate(global_step, expect_switch_descend_step)
    # in interval steps, fixed steps is _ADAM_WARM_STEP.
    if _ADAM_WARM_STEP > 0 and _ADJUST_LORA_SCHEDULE:
        ratio = ratio * (interval / (interval - _ADAM_WARM_STEP))
    return ratio


def _get_other_schedule(global_step):
    return 1.


def obtain_lora_scheduler(
        optimizer,
        base_interval,
        expect_switch_descend_step,
        optim_beta,
        origin_scheduler,
        last_epoch=-1,
):
    lr_lambda = []
    for elem in optim_beta:
        if "lr_ratio" in elem:
            schedule = _partial(
                _get_lora_schedule,
                base_interval=base_interval,
                expect_switch_descend_step=expect_switch_descend_step
            )
        else:
            schedule = _partial(
                _get_other_schedule
            )
        lr_lambda.append(schedule)
    switch_lora_scheduler = _MultiplicativeLR(optimizer, lr_lambda, last_epoch)
    return _ChainedScheduler([origin_scheduler, switch_lora_scheduler])


def _get_switch_rate(global_step: int, expect_switch_descend_step):
    if _SWITCH_DESCEND_TYPE == "Z":
        # Slowly decrease when step is little.
        # Fast decrease when step is close to expect_switch_descend_step.
        k = 10. / expect_switch_descend_step
        value = 1 - 1 / (1 + _math.exp(-k * (global_step - expect_switch_descend_step)))
    elif _SWITCH_DESCEND_TYPE == "exponential":
        # decrease exponentially
        # value is 0.3 when reaching expect_switch_descend_step
        x = 0.3 ** (1 / expect_switch_descend_step)
        value = x ** (global_step + 1)
    else:
        raise ValueError("Unsupported descend type.")
    # return max(value, 0.0001)
    return max(value, 1e-8)


def _get_switch_replace_num(select_num, global_step, base_switch_interval, expect_switch_descend_step):
    if _FIX_SWITCH_LORA_INTERVAL:
        interval = base_switch_interval
    else:
        interval = base_switch_interval / _get_switch_rate(global_step, expect_switch_descend_step)
    if interval < _ADAM_WARM_STEP * 2:  # *2 since only one of lora_A and lora_B can be fixed
        raise RuntimeError("Switch interval can not be less than adam warm up steps.")
    replace_num = select_num / interval
    replace_num_decimal = replace_num - int(replace_num)
    replace_num = int(replace_num) + (1 if _random.random() < replace_num_decimal else 0)
    return replace_num


@_torch.no_grad()
def switch_lora(model, optimizer, global_step, expect_switch_descend_step):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w

    def drop(candidates, candidates_len, candidates_weight: list[float], selected_indices):
        """
        Drop one candidate if the condition meets
        """
        if not _DROP_CANDIDATES:
            return candidates, candidates_len
        available_indices = set(range(candidates_len)) - set(selected_indices)
        available_indices = list(available_indices)
        if len(available_indices) == 0:
            return candidates, candidates_len

        # search drop_index
        drop_index = available_indices[0]
        for i in available_indices:
            if candidates_weight[i] < candidates_weight[drop_index]:
                drop_index = i
        to_drop = candidates[drop_index]

        candidates[drop_index] = candidates[candidates_len - 1]

        candidates_weight[drop_index] = candidates_weight[candidates_len - 1]
        del candidates_weight[candidates_len - 1]

        # Add the dropped candidate to remained candidates
        # to_drop = to_drop / (candidates_len - 1)
        # for candidate in new_candidates:
        #     old_norm = _torch.norm(candidate)
        #     candidate += to_drop
        #     candidate *= old_norm / _torch.norm(candidate)

        for i, s in enumerate(selected_indices):
            if s == candidates_len - 1:
                selected_indices[i] = drop_index
            elif s == drop_index:
                raise RuntimeError("Wrong drop index")
        return candidates, candidates_len - 1

    gather_estimated_rank(model, global_step)
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        layer._candidates2cpu()
        origin_available_candidate_num = min(layer.in_features, layer.out_features) - layer.r
        if origin_available_candidate_num <= 0:
            continue

        replace_num = _get_switch_replace_num(layer.r, global_step, _SWITCH_LORA_INTERVAL, expect_switch_descend_step)
        fixed_A_num = sum([1 for s in layer.fixed_A_steps if s > 0])
        fixed_B_num = sum([1 for s in layer.fixed_B_steps if s > 0])
        available_num = min((layer.candidate_A_index - fixed_A_num - layer.candidate_B_index) % layer.r,
                            (layer.candidate_B_index - fixed_B_num - layer.candidate_A_index) % layer.r)
        if replace_num > available_num:
            replace_num = available_num

        if replace_num == 0:
            continue

        to_replace_A = [i % layer.r for i in range(layer.candidate_A_index, layer.candidate_A_index + replace_num)]
        to_replace_B = [i % layer.r for i in range(layer.candidate_B_index, layer.candidate_B_index + replace_num)]
        layer.candidate_A_index = (layer.candidate_A_index + replace_num) % layer.r
        layer.candidate_B_index = (layer.candidate_B_index + replace_num) % layer.r
        replace_A_map = _select_indices(layer.candidates_A_len, to_replace_A, layer.selected_A_indices,
                                        layer.fixed_A_steps, layer.candidates_A_weight)
        _switch_chosen_indices(layer, replace_A_map, "A", _CANDIDATES_DROP_RATE)
        replace_B_map = _select_indices(layer.candidates_B_len, to_replace_B, layer.selected_B_indices,
                                        layer.fixed_B_steps, layer.candidates_B_weight)
        if _ZERO_INIT_B:
            replace_B_map.clear()
        _switch_chosen_indices(layer, replace_B_map, "B", _CANDIDATES_DROP_RATE)

        switch_rate = _get_switch_rate(global_step, expect_switch_descend_step)
        if (layer.candidates_A_len - layer.r) / origin_available_candidate_num > switch_rate:
            layer.candidates_A, layer.candidates_A_len = drop(layer.candidates_A, layer.candidates_A_len,
                                                              layer.candidates_A_weight,
                                                              layer.selected_A_indices)
        if (layer.candidates_B_len - layer.r) / origin_available_candidate_num > switch_rate:
            layer.candidates_B, layer.candidates_B_len = drop(layer.candidates_B, layer.candidates_B_len,
                                                              layer.candidates_B_weight,
                                                              layer.selected_B_indices)

    _zero_lora_states(model, optimizer)
    _correct_switched_lora(model)


@_torch.no_grad()
def gather_estimated_rank(model, step):
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        if not hasattr(layer, "gathered_ranks"):
            continue
        if not hasattr(layer, "ranks"):
            layer.ranks = {}
        layer.ranks[step] = [0] * layer.r
        for i in range(layer.r):
            gathered_rank = layer.gathered_ranks[i]
            if len(gathered_rank) == 0:
                continue
            std = _np.std(gathered_rank)
            avg = _np.average(gathered_rank)
            layer.ranks[step][i] = avg
            gathered_rank.clear()

        n = layer.lora_A.shape[1]
        dest_rank = (2 / (n * _torch.pi)) ** 0.5

        avg_rank = _np.average(layer.ranks[step])
        for i in range(layer.r):
            rank = layer.ranks[step][i]
            layer.candidates_A_weight[layer.selected_A_indices[i]].fill_(1. / n / (dest_rank / rank) ** 3)


@_torch.no_grad()
def _switch_chosen_indices(layer, replace_map: dict[int, int], replace_type: str, switch_drop: float):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w

    def to_candidate(mat, candidate):
        # set new_mat as candidate
        # new_mat = (1-switch_drop)*candidate+(mat-candidate)
        new_mat = mat - switch_drop * candidate
        # normalize(new_mat) so that its norm is the same as math
        new_mat *= _torch.norm(mat) / _torch.norm(new_mat)
        candidate.copy_(new_mat)
        return candidate

    if replace_type == "A":
        mat = layer.lora_A

        # update weight
        for mat_index, dest_index in replace_map.items():
            if layer.quantize is None:
                layer.weight += T(
                    layer.lora_B[:, None, mat_index] @ (mat[mat_index, None, :] - layer.candidates_A[dest_index]),
                    layer) * layer.scaling
            else:
                odd_mat = layer.odd_mat
                bnbF.dequantize_4bit(layer.weight.data, layer.weight.quant_state, out=odd_mat)
                odd_mat += T(
                    layer.lora_B[:, None, mat_index] @ (mat[mat_index, None, :] - layer.candidates_A[dest_index]),
                    layer) * layer.scaling
                layer.weight.data, layer.weight.quant_state = bnbF.quantize_4bit(
                    odd_mat,
                    quant_type=layer.weight.quant_type,
                    compress_statistics=layer.weight.compress_statistics,
                )

        # save lora_A value to candidates
        if switch_drop != 0:
            for mat_index, dest_index in replace_map.items():
                origin_candidate_index = layer.selected_A_indices[mat_index]
                # lora_A is split by line index
                to_candidate(mat[mat_index, :], layer.candidates_A[origin_candidate_index])

        # change lora_A value to candidates
        for mat_index, dest_index in replace_map.items():
            # lora_A is split by line index
            mat[mat_index, :] = layer.candidates_A[dest_index]
            layer.selected_A_indices[mat_index] = dest_index

            layer.fixed_B_steps[mat_index] = _ADAM_WARM_STEP

    else:
        mat = layer.lora_B

        # update weight
        for mat_index, dest_index in replace_map.items():
            if layer.quantize is None:
                layer.weight += T(
                    (mat[:, mat_index] - layer.candidates_B[dest_index]).unsqueeze(1) @ layer.lora_A[mat_index, None, :],
                    layer) * layer.scaling
            else:
                odd_mat = layer.odd_mat
                bnbF.dequantize_4bit(layer.weight.data, layer.weight.quant_state, out=odd_mat)
                odd_mat += T(
                    (mat[:, mat_index] - layer.candidates_B[dest_index]).unsqueeze(1) @ layer.lora_A[mat_index, None, :],
                    layer) * layer.scaling
                layer.weight.data, layer.weight.quant_state = bnbF.quantize_4bit(
                    odd_mat,
                    quant_type=layer.weight.quant_type,
                    compress_statistics=layer.weight.compress_statistics,
                )

        # save lora_B value to candidates
        if switch_drop != 0:
            for mat_index, dest_index in replace_map.items():
                origin_candidate_index = layer.selected_B_indices[mat_index]
                # lora_B is split by column index
                to_candidate(mat[:, mat_index], layer.candidates_B[origin_candidate_index])

        # change lora_B value to candidates
        for mat_index, dest_index in replace_map.items():
            # lora_B is split by column index
            mat[:, mat_index] = layer.candidates_B[dest_index]
            layer.selected_B_indices[mat_index] = dest_index

            layer.fixed_A_steps[mat_index] = _ADAM_WARM_STEP


def _select_indices(candidate_num: int,
                    indices_to_replace: list[int],
                    current_indices,
                    fixed_steps,
                    weights) -> dict[int, int]:
    """
    :param Total candidate_num: number of candidates (including unavailable ones)
    :param indices_to_replace: indices in lora_A/lora_B to be replaced.
    :param current_indices: current selected candidate indices. i.e. selected_A_indices or selected_B_indices
    :param fixed_steps: remained fixed steps for selected elements.
    :param weights: weights of possibility to select candidate elements
    :return:
    """
    all_indices = set(range(candidate_num))
    current_indices_set = set(current_indices)
    candidates_indices_to_replace = set([current_indices[i] for i in indices_to_replace])
    fixed_indices = set([current_indices[i] for i in range(len(current_indices)) if fixed_steps[i] > 0])
    available_indices = all_indices - (current_indices_set - candidates_indices_to_replace) - fixed_indices

    if len(fixed_indices.intersection(candidates_indices_to_replace)) > 0:
        raise RuntimeError("Fixed indices are chosen to switch")

    available_indices = list(available_indices)
    p = [weights[i] for i in available_indices]
    sum_p = sum(p)
    p = [p_i / sum_p for p_i in p]
    selected_indices = _random.choice(available_indices, size=len(indices_to_replace),
                                      replace=False, p=p)
    replace_map = dict(zip(indices_to_replace, selected_indices))
    return replace_map


def _get_int_para(value):
    return _nn.Parameter(_torch.tensor(value, dtype=_torch.int32), requires_grad=False)


def _get_float_para(value):
    return _nn.Parameter(_torch.tensor(value, dtype=_torch.float32), requires_grad=False)


class SwitchLoRAModel(_torch.nn.Module):
    def __init__(
            self,
            origin_model,
            to_lora_layer_name,
            r: int = 128,
            lora_alpha: float = 1,
            lora_dropout: float = 0.1,
            quantize=None,
            use_double_quant: bool = False
    ):
        super().__init__()
        self.origin_model = origin_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.to_lora_layer_name = to_lora_layer_name
        self.quantize = quantize
        self.use_double_quant = use_double_quant

        self.forward = self.origin_model.forward

        lora_utils.set_use_lora(layer_replace_dict)
        lora_utils.replace_with_lora_auto(
            self.origin_model, to_lora_layer_name,
            lora_rank=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize=quantize,
            merge_weights=quantize is None,
            bnb_4bit_use_double_quant=use_double_quant
        )

    def _pre_save_candidates_list(self):
        def set_value(param, value):
            param.fill_(value)

        def set_list_value(param_list, values):
            for i, param in enumerate(param_list):
                param.fill_(values[i])

        lora_layers = lora_utils.iter_lora_layers(self)
        for layer in lora_layers:
            if not hasattr(layer, "candidates_A_len"):
                continue
            set_value(layer.candidates_A_len_param, layer.candidates_A_len)
            set_value(layer.candidates_B_len_param, layer.candidates_B_len)
            set_value(layer.candidate_A_index_param, layer.candidate_A_index)
            set_value(layer.candidate_B_index_param, layer.candidate_B_index)
            set_list_value(layer.candidates_A_weight_param, layer.candidates_A_weight)
            set_list_value(layer.candidates_B_weight_param, layer.candidates_B_weight)
            set_list_value(layer.selected_A_indices_param, layer.selected_A_indices)
            set_list_value(layer.selected_B_indices_param, layer.selected_B_indices)
            set_list_value(layer.fixed_A_steps_param, layer.fixed_A_steps)
            set_list_value(layer.fixed_B_steps_param, layer.fixed_B_steps)

    def save_pretrained(self, path, **kwargs):
        self._pre_save_candidates_list()
        self.origin_model.save_pretrained(path, **kwargs)
        with open(_os.path.join(path, "switch_lora_config.json"), "w") as f:
            _json.dump({
                "r": self.r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "to_lora_layer_name": self.to_lora_layer_name,
                "quantize": self.quantize,
                "use_double_quant": self.use_double_quant,
                # "switch_lora_descent_rate": self.switch_lora_descent_rate,
                # "switch_lora_interval": self.switch_lora_interval,
                # "switch_lora_drop": self.switch_lora_drop,
            }, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        from transformers import AutoModelForCausalLM, AutoConfig
        with open(_os.path.join(path, "switch_lora_config.json"), "r") as f:
            relora_config = _json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)

        model = cls(base_model, **relora_config)

        with open(_os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = _torch.load(f, map_location="cpu")

        model.origin_model.load_state_dict(state_dict, strict=True)
        return model

    def load_state_dict(self, state_dict: _Mapping[str, _Any], strict: bool = True, assign: bool = False):
        result = self.origin_model.load_state_dict(state_dict, strict, assign)
        return result


class SwitchLoraLayer():
    def __init__(
            self,
            sync_lora_beta: float = 0.5,
    ):
        if not hasattr(self, 'lora_A'):
            raise RuntimeError("init of SwitchLoraLayer should be used before loralib LoRALayer init.")
        self.sync_lora_beta = sync_lora_beta
        in_features = self.lora_A.shape[1]
        out_features = self.lora_B.shape[0]
        r = self.lora_A.shape[0]

        self._init_candidates()

        # flag to judge whether to estimate at next forward propagation
        self.to_estimate_rank = False

        if lora_utils.CAL_DELTA_NORM:
            # For test
            # Used to calculate the norm of gradients
            self.Wx = _nn.Parameter(_torch.empty(50, out_features), requires_grad=False)  # 50 is from transformer
            self.Ax = _nn.Parameter(_torch.empty(50, r), requires_grad=False)  # 50 is from transformer
            self.BAx = _nn.Parameter(_torch.empty(50, out_features), requires_grad=False)  # 50 is from transformer
            self.BA = _nn.Parameter(_torch.empty(out_features, in_features), requires_grad=False)
            self.deltaA = _nn.Parameter(_torch.empty(r, in_features), requires_grad=False)
            self.deltaB = _nn.Parameter(_torch.empty(out_features, r), requires_grad=False)
            self.deltaW = _nn.Parameter(_torch.empty(out_features, in_features), requires_grad=False)

    def forward(self, x: _torch.Tensor):
        if self.training and self.to_estimate_rank:
            with _torch.no_grad():
                rank = lora_utils.estimate_rank(self.lora_A.transpose(0, 1), x)
                if not hasattr(self, "gathered_ranks"):
                    self.gathered_ranks = []
                for i, r in enumerate(rank):
                    if len(self.gathered_ranks) <= i:
                        self.gathered_ranks.append([])
                    self.gathered_ranks[i].append(r.item())

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            bound_A, bound_B = lora_utils.get_init_lora_bound(self.in_features, self.out_features, self.r)
            with _torch.no_grad():
                self.lora_A.uniform_(-bound_A, bound_A)
                self.lora_B.uniform_(-bound_B, bound_B)

    @_torch.no_grad()
    def _init_candidates(self):
        """
        Initialize the candidates for lora_A and lora_B.
        """
        if hasattr(self, "candidates_A"):
            return
        # Follow Xavier and Kaiming initialization to keep std of output of lora_B*lora_A uniform.
        # But make A*deltaB and B*deltaA the same size
        bound_A, bound_B = lora_utils.get_init_lora_bound(self.in_features, self.out_features, self.r)
        if _ZERO_INIT_B:
            bound_B = 0

        candidate_num = min(self.in_features, self.out_features)

        self.candidates_A, self.candidates_A_weight, self.selected_A_indices \
            = _init_orthogonal_lora(self.in_features, bound_A, self.r, self.lora_A, candidate_num)
        self.candidates_B, self.candidates_B_weight, self.selected_B_indices \
            = _init_orthogonal_lora(self.out_features, bound_B, self.r, self.lora_B, candidate_num)
        self.fixed_A_steps = [0] * len(self.selected_A_indices)
        self.fixed_B_steps = [0] * len(self.selected_B_indices)

        # register candidates as model parameters
        self.candidates_A = _nn.ParameterList([_nn.Parameter(p, requires_grad=False) for p in self.candidates_A])
        self.candidates_B = _nn.ParameterList([_nn.Parameter(p, requires_grad=False) for p in self.candidates_B])

        # set some variables for candidates as model parameters
        self.candidates_A_len_param = _get_int_para(len(self.candidates_A))
        self.candidates_B_len_param = _get_int_para(len(self.candidates_B))
        self.candidates_A_weight_param = _nn.ParameterList([_get_float_para(v) for v in self.candidates_A_weight])
        self.candidates_B_weight_param = _nn.ParameterList([_get_float_para(v) for v in self.candidates_B_weight])
        self.selected_A_indices_param = _nn.ParameterList([_get_int_para(v) for v in self.selected_A_indices])
        self.selected_B_indices_param = _nn.ParameterList([_get_int_para(v) for v in self.selected_B_indices])

        self.fixed_A_steps_param = _nn.ParameterList(
            [_get_int_para(0) for _ in range(len(self.selected_A_indices))])
        self.fixed_B_steps_param = _nn.ParameterList(
            [_get_int_para(0) for _ in range(len(self.selected_B_indices))])

        self.candidate_A_index_param = _get_int_para(0)
        # set to self.r / 2 to make sure selected A and selected B are different
        # since select strategy is incremental selection
        self.candidate_B_index_param = _get_int_para(self.r // 2)

        # self.candidates_A_len = len(self.candidates_A)
        # self.candidates_B_len = len(self.candidates_B)

        # self.fixed_A_steps = [0 for _ in range(len(self.selected_A_indices))]
        # self.fixed_B_steps = [0 for _ in range(len(self.selected_B_indices))]

        # self.candidate_A_index = 0
        # set to self.r / 2 to make sure selected A and selected B are different
        # since select strategy is incremental selection
        # self.candidate_B_index = self.r // 2

        for i, index in enumerate(self.selected_A_indices):
            self.lora_A[i, :] = self.candidates_A[index]
        for i, index in enumerate(self.selected_B_indices):
            self.lora_B[:, i] = self.candidates_B[index]

    def _candidates2cpu(self):
        def get_value(param):
            return param.item()

        def set_list_value(param_list, value_list):
            for i, param in enumerate(param_list):
                value_list[i] = param.item()

        if hasattr(self, "candidates_A_len"):
            return

        self.candidates_A_len = get_value(self.candidates_A_len_param)
        self.candidates_B_len = get_value(self.candidates_B_len_param)
        self.candidate_A_index = get_value(self.candidate_A_index_param)
        self.candidate_B_index = get_value(self.candidate_B_index_param)
        set_list_value(self.candidates_A_weight_param, self.candidates_A_weight)
        set_list_value(self.candidates_B_weight_param, self.candidates_B_weight)
        set_list_value(self.selected_A_indices_param, self.selected_A_indices)
        set_list_value(self.selected_B_indices_param, self.selected_B_indices)
        set_list_value(self.fixed_A_steps_param, self.fixed_A_steps)
        set_list_value(self.fixed_B_steps_param, self.fixed_B_steps)


class Linear(_lora.Linear, SwitchLoraLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            sync_lora_beta: float = 0.5,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            quantize=None,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            **kwargs):
        _lora.Linear.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out,
                              merge_weights, **kwargs)

        SwitchLoraLayer.__init__(self, sync_lora_beta)
        self.weight.requires_grad = False

        if merge_weights and quantize is not None:
            raise NotImplementedError("Merging is not yet supported when quantization is enabled.")

        self.quantize = quantize
        if quantize is None:
            pass
        elif quantize == "4bit":
            self.weight = bnb.nn.Params4bit(
                self.weight.data,
                requires_grad=False,
                compress_statistics=bnb_4bit_use_double_quant,
                quant_type=bnb_4bit_quant_type,
            )
        elif quantize == "8bit":
            # logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
            raise NotImplementedError(
                "merge_and_reinit_functional for quantized models is not implemented yet. Use non-functional implementation")
            self.weight = bnb.nn.Int8Params(
                self.weight.data,
                requires_grad=False,
            )
        else:
            raise ValueError(f"Unknown quantize type: {quantize}")
        if quantize is not None:
            # Used for convenience. Memory overhead here can be diminished.
            # Memory usage of this parameter is not included in our paper.
            self.odd_mat = _nn.Parameter(_torch.empty(
                self.weight.data.shape,
                dtype=self.lora_B.dtype,
                device=self.weight.data.device), requires_grad=False)


    def forward(self, x: _torch.Tensor):
        # Replace _lora.Linear for better efficiency
        # result = _lora.Linear.forward(self, x)
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            if self.quantize == "4bit":
                result = bnb.matmul_4bit(x, T(self.weight.t()), bias=self.bias, quant_state=self.weight.quant_state)
            elif self.quantize == "8bit":
                result = bnb.matmul(x, T(self.weight.t()), bias=self.bias, quant_state=self.weight.quant_state)
            else:
                result = _F.linear(x, T(self.weight), bias=self.bias)
            # result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += _F.linear(_F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        else:
            if self.quantize:
                raise NotImplementedError("Merging is not yet supported when quantization is enabled.")
            result = _F.linear(x, T(self.weight), bias=self.bias)

        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        _lora.Linear.reset_parameters(self)
        SwitchLoraLayer.reset_parameters(self)


class SwitchLoraConv(_lora.ConvLoRA, SwitchLoraLayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1,
                 sync_lora_beta: float = 0.5, lora_dropout=0., merge_weights=True, **kwargs):
        _lora.ConvLoRA.__init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1,
                                lora_dropout=0., merge_weights=True, **kwargs)
        SwitchLoraLayer.__init__(self, sync_lora_beta)

    def forward(self, x: _torch.Tensor):
        result = _lora.ConvLoRA.forward(self, x)
        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        _lora.ConvLoRA.reset_parameters(self)
        SwitchLoraLayer.reset_parameters(self)


class Conv1d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(_nn.Conv1d, *args, **kwargs)


class Conv2d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(_nn.Conv2d, *args, **kwargs)


class Conv3d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(_nn.Conv3d, *args, **kwargs)


layer_replace_dict = {
    _nn.Linear: Linear,
    _nn.Conv1d: Conv1d,
    _nn.Conv2d: Conv2d,
    _nn.Conv3d: Conv3d
}
