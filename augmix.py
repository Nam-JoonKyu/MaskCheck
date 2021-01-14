import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentations import *
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

class AugMix(nn.Module):
    def __init__(self, k=3, alpha=1, severity=3):
        super(AugMix, self).__init__()
        self.k = k
        self.alpha = alpha
        self.severity = severity
        self.dirichlet = Dirichlet(torch.full(torch.Size([k]), alpha, dtype=torch.float32))
        self.beta = Beta(alpha, alpha)
        self.augs = augmentations
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def augment_and_mix(self, images, preprocess):
        '''
        Args:
            images: PIL Image
            preprocess: transform[ToTensor, Normalize]

        Returns: AugmentAndMix Tensor
        '''
        mix = torch.zeros_like(preprocess(images))
        w = self.dirichlet.sample()
        for i in range(self.k):
            aug = images.copy()
            depth = np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augs)
                aug = op(aug, 3)
            mix = mix + w[i] * preprocess(aug)

        m = self.beta.sample()

        augmix = m * preprocess(images) + (1 - m) * mix

        return augmix

    def jensen_shannon(self, logits_o, logits_1, logits_2):
        p_o = F.softmax(logits_o, dim=1)
        p_1 = F.softmax(logits_1, dim=1)
        p_2 = F.softmax(logits_2, dim=1)

        # kl(q.log(), p) -> KL(p, q)
        M = torch.clamp((p_o + p_1 + p_2) / 3, 1e-7, 1)  # to avoid exploding
        js = (self.kl(M.log(), p_o) + self.kl(M.log(), p_1) + self.kl(M.log(), p_2)) / 3
        return js


