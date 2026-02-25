# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def expand_chunk_mask(chunk_mask, chunk_id):
    """
    chunk_mask: (B, Kmax)   binary mask for each chunk
    chunk_id:   (B, N)      chunk indices 0..Kmax

    Returns:
        original_mask: (B, N) binary mask
    """
    B, N = chunk_id.shape

    # pad chunk_mask with a zero column at front for chunk_id = 0
    padded_mask = torch.cat([
        torch.zeros(B, 1, dtype=chunk_mask.dtype, device=chunk_mask.device),
        chunk_mask
    ], dim=1)    # shape (B, Kmax+1)

    # gather back to original positions
    original_mask = padded_mask.gather(1, chunk_id)

    return original_mask

def chunk_sums_vectorized(mask, values, return_chunk_id = True):
    """
    mask:   (B, N) bool or {0,1}
    values: (B, N) float/real, requires_grad ok

    Returns:
      sums_padded: (B, Kmax) tensor of per-row chunk sums (padded with 0)
      lengths:     (B,) number of chunks per row
    """
    B, N = mask.shape
    mask = mask.bool()

    # Mark starts of each contiguous 1-run in every row
    prev = F.pad(mask[:, :-1], (1, 0), value=False)        # (B, N)
    starts = mask & ~prev                                   # (B, N)

    # Per-row chunk ids: 0 outside mask, 1..K inside chunks
    chunk_id = starts.cumsum(dim=1).to(torch.long) * mask.to(torch.long)  # (B, N)

    # Number of chunks per row and global max
    lengths = chunk_id.max(dim=1).values                    # (B,)
    Kmax = int(lengths.max().item())

    # Handle case with no chunks at all
    if Kmax == 0:
        return values.new_zeros((B, 0)), lengths, (chunk_id if return_chunk_id else None)

    # Scatter-add values into (B, Kmax+1); column 0 is "non-chunk" bin
    out = values.new_zeros((B, Kmax + 1))
    out.scatter_add_(1, chunk_id.clamp_max(Kmax), values * mask)  # (B, Kmax+1)

    # Drop the non-chunk column → per-row chunk sums, padded to Kmax
    sums_padded = out[:, 1:]                                      # (B, Kmax)
    return sums_padded, lengths, (chunk_id if return_chunk_id else None)


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        valid_search_stats = float(
            np.array(data.meta_info["valid_search_stats"], dtype=np.int16).mean()
        )

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages','token_level_rewards']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch_keys = set(data.batch.keys()) if hasattr(data, 'batch') else set()
        if 'answer_mask' in batch_keys:
            select_keys.append('answer_mask')

        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                if 'answer_mask' in data.keys() and self.config.mask_answer:
                    answer_mask = data['answer_mask']
                    if answer_mask.shape[-1] != response_length:
                        answer_mask = answer_mask[:, -response_length:]
                    answer_mask = answer_mask.to(response_mask.dtype)
                else:
                    answer_mask = torch.ones_like(response_mask, dtype=response_mask.dtype)

                old_log_prob = data['old_log_probs']
                ref_log_prob = data['ref_log_prob']
                advantages = data['advantages']
                token_level_rewards = data['token_level_rewards']
                response_length = token_level_rewards.shape[-1]
                non_zero_mask = (token_level_rewards != 0)
                scores = (token_level_rewards * non_zero_mask).sum(dim=-1,keepdim=True)

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff
                positive_advantage_mask = (advantages >= 0).float()
                negative_advantage_mask = (advantages < 0).float()
                penalty_term = old_log_prob.detach() - log_prob

                # _,_, chunk_ids = chunk_sums_vectorized(response_mask,penalty_term)
                # max_vals = torch.max(chunk_ids, dim=-1, keepdim=True)[0]
                # last_chunk_mask = (chunk_ids != max_vals).float()  ## mask last chunk
                # metrics["actor/mean_reduce_chunk_last"] = (
                #     torch.mean(torch.sum(1 - last_chunk_mask, dim=-1)).detach().item()
                # )

                # answer_mask = last_chunk_mask
                lambda_answer_part = getattr(self.config, 'answer_no_reduce_lambda', 0.0)

                if self.config.mask_adaptive:
                    answer_weight = max(valid_search_stats-1,0) ## set the base search number to 2, can be changed if need more search
                    mask_weight = lambda_answer_part*answer_weight
                else:
                    mask_weight = lambda_answer_part
                metrics["actor/answer_weight"] = mask_weight
                coeff_map = torch.where(
                    answer_mask.bool(),
                    self.config.no_reduce_lambda,
                    min(mask_weight, self.config.no_reduce_lambda),
                )

                diff_log_prob = old_log_prob.detach() - log_prob
                base_penalty_term = torch.maximum(torch.zeros_like(log_prob), diff_log_prob)

                if self.config.chunk_noreduce:
                    adv_sum = advantages.sum(-1)
                    masked_response_mask = response_mask * positive_advantage_mask
                    sum_val,length, chunk_ids = chunk_sums_vectorized(masked_response_mask,penalty_term)
                    chunk_mask = (sum_val > 0).to(torch.long)  # (B, Kmax)
                    original_token_mask = expand_chunk_mask(chunk_mask, chunk_ids)
                    valid_penalty_tokens = original_token_mask.detach() * base_penalty_term
                    metrics["actor/mean_reduce_chunk"] = torch.mean(chunk_mask[adv_sum>=0].sum(-1).float()).detach().item()
                else:
                    masked_response_mask = response_mask * positive_advantage_mask
                    current_penalty_term = diff_log_prob * positive_advantage_mask
                    penalty_sentence = verl_F.masked_sum(
                        current_penalty_term, masked_response_mask, axis=-1
                    )
                    penalty_sentence = torch.maximum(torch.zeros_like(penalty_sentence), penalty_sentence)
                    penalty_sentence[penalty_sentence > 0] = 1.0
                    valid_penalty_tokens = penalty_sentence.unsqueeze(-1) * base_penalty_term * positive_advantage_mask


                weighted_penalty = valid_penalty_tokens * coeff_map
                loss_noreduce = verl_F.masked_mean(weighted_penalty, response_mask)

                mask_per_response = (answer_mask == 0).float().sum(dim=-1)
                avg_mask_per_response = mask_per_response.mean().detach()
                has_zero_ratio = (mask_per_response > 0).float().mean().detach()
                metrics["actor/num_answer_mask"] = avg_mask_per_response.item()
                metrics["actor/is_answer_mask"] = has_zero_ratio.item()
                # metrics["actor/num_answer_mask_activate"] = num_answer_mask.detach().item()
                if torch.any((1-answer_mask) > 0):
                    lowest_prob_mean = verl_F.masked_mean(torch.exp(log_prob), 1-answer_mask)
                    lowest_prob_min = verl_F.masked_min(torch.exp(log_prob), 1-answer_mask)
                    metrics["actor/lowest_prob_avg_mean"] = (
                        lowest_prob_mean.detach().item()
                    )
                    metrics["actor/lowest_prob_avg_min"] = (
                        lowest_prob_min.detach().item()
                    )
                else:
                    metrics["actor/lowest_prob_avg_mean"] = 0.0
                    metrics["actor/lowest_prob_avg_min"] = 0.0


                pos_avg_old_log_prob = verl_F.masked_mean(old_log_prob* positive_advantage_mask, response_mask)
                metrics.update({"actor/pos_avg_old_log_prob": pos_avg_old_log_prob.detach().item()})
                neg_avg_old_log_prob = verl_F.masked_mean(old_log_prob* negative_advantage_mask, response_mask)
                metrics.update({"actor/neg_avg_old_log_prob": neg_avg_old_log_prob.detach().item()})

                pos_avg_ref_log_prob = verl_F.masked_mean(ref_log_prob* positive_advantage_mask, response_mask)
                metrics.update({"actor/pos_avg_ref_log_prob": pos_avg_ref_log_prob.detach().item()})
                neg_avg_ref_log_prob = verl_F.masked_mean(ref_log_prob* negative_advantage_mask, response_mask)
                metrics.update({"actor/neg_avg_ref_log_prob": neg_avg_ref_log_prob.detach().item()})


                policy_loss = policy_loss +  loss_noreduce
                metrics["actor/loss_noreduce"] = loss_noreduce.detach().item()
                metrics["actor/loss_noreduce_coef"] = self.config.no_reduce_lambda

                if self.config.use_kl_loss:
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)
            if self.config.grad_nan_clip:
                torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), float("inf"))
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
