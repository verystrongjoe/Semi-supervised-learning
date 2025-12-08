# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from copy import deepcopy
from collections import Counter

from semilearn.algorithms.hooks import MaskingHook


class FlexMatchThresholdingHook(MaskingHook):
    """
    Adaptive Thresholding in FlexMatch
    """
    def __init__(self, ulb_dest_len, num_classes, thresh_warmup=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ulb_dest_len = ulb_dest_len
        self.num_classes = num_classes
        self.thresh_warmup = thresh_warmup
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs):
        pseudo_counter = Counter(self.selected_label.tolist())
        if pseudo_counter.get(-1, 0) < self.ulb_dest_len:  # not all unlabeled data are -1 (at least one sample has been pseudo-labeled)
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, idx_ulb, softmax_x_ulb=True, *args, **kwargs):
        probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        classwise_acc = self.classwise_acc.to(max_idx.device)
        mask = max_probs.ge(algorithm.p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))  # convex
        select = max_probs.ge(algorithm.p_cutoff)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1].cpu()] = max_idx[select == 1].cpu()
        self.update()

        return mask
        
        

