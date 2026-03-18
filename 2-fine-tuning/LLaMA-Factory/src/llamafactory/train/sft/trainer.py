# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        在父类（已支持 SFT+UL 混合损失）的基础上，分别计算 CE 与 UL 子集的 token 平均熵，
        并在 sequence-parallel 时做组内规约。结果写入：
            - outputs["ce_data_entropy"]
            - outputs["ul_data_entropy"]
        """
        # —— 先把会被父类 compute_loss 弹走的字段留一份引用（用于计算熵） ——
        labels_ref = inputs.get("labels", None)
        loss_type_ref = inputs.get("loss_type", None)
        attn_ref = inputs.get("attention_mask", None)

        if return_outputs:
            # 父类：已计算好 SFT+UL 的混合 loss，并返回 logits
            loss_and_outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            # 兼容父类两种返回：有的实现直接返回 (loss, outputs)，有的可能只返回 loss
            if isinstance(loss_and_outputs, tuple) and len(loss_and_outputs) == 2:
                loss, outputs = loss_and_outputs
            else:
                # 如果父类只返回了 loss，这里没有 logits 就无法计算熵
                return loss_and_outputs

            # === 分别计算 CE 与 UL 子集的 token 熵 ===
            with torch.no_grad():
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]  # [B, L, V]
                if labels_ref is None or loss_type_ref is None:
                    # 没有必要字段就不计算熵
                    ce_entropy = ul_entropy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                else:
                    B, L, V = logits.shape
                    device = logits.device
                    labels = labels_ref.to(device)
                    loss_type = loss_type_ref.to(device)  # [B]
                    if attn_ref is not None:
                        labels = torch.where(attn_ref.to(device).bool(), labels, torch.full_like(labels, -100))

                    # token 熵（逐位置）
                    log_probs = F.log_softmax(logits.view(-1, V), dim=-1)  # [B*L, V]
                    probs = log_probs.exp()
                    token_entropy = -(probs * log_probs).sum(dim=-1).view(B, L)  # [B, L]

                    # valid = (labels != -100)  # [B, L]
                    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()  # [B, L]
                    valid = (shift_labels != -100)
                    lt2d = loss_type.view(B, 1).expand(B, L)  # [B, L]
                    ce_mask = (lt2d == 0) & valid
                    ul_mask = (lt2d == 1) & valid

                    ce_cnt = ce_mask.sum()
                    ul_cnt = ul_mask.sum()
                    ce_entropy = (token_entropy[ce_mask].sum() / ce_cnt.clamp(min=1)) if ce_cnt > 0 else torch.tensor(0.0, device=device)
                    ul_entropy = (token_entropy[ul_mask].sum() / ul_cnt.clamp(min=1)) if ul_cnt > 0 else torch.tensor(0.0, device=device)

                # 写回
                outputs["ce_data_entropy"] = ce_entropy
                outputs["ul_data_entropy"] = ul_entropy

            # 返回与 HF 期望一致的形状
            result = (loss, outputs)

        else:
            # 若调用方没要 outputs，就走父类逻辑（此时不计算熵）
            result = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        if "logits" in outputs:
            logits = outputs.logits
            if logits is not None and logits.requires_grad:
                logits = logits.detach()
            outputs.logits = None
            del logits

        return result
    
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
