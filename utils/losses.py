import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-6


class MAE(nn.Module):
    def __init__(self, num_classes=10):
        super(MAE, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return loss.mean()


class RCE(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(RCE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()


class NCE(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()


class GCE(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCE, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class BS(object):
    def __call__(self, outputs):
        ## hard booststrapping
        targets = torch.argmax(outputs, dim=1)
        return nn.CrossEntropyLoss()(outputs, targets)

        ## soft bootstrapping
        # probs = torch.softmax(outputs, dim=1)
        # return torch.mean(torch.sum(-torch.log(probs+1e-6)*probs, dim=1))


class DAL(nn.Module):
    def __init__(self, num_classes=10, q=1.5, lamb=1.0):
        super(DAL, self).__init__()
        self.gce = GCE(num_classes=num_classes, q=q)
        self.bs = BS()
        self.num_classes = num_classes
        self.q = q
        self.lamb = lamb

    def forward(self, pred, labels):
        loss = self.gce(pred, labels) + self.lamb * self.bs(pred) / (self.q * np.log(self.num_classes))
        return loss


class SCE(nn.Module):
    def __init__(self, num_classes=10, alpha=1, beta=1):
        super(SCE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = rce.mean()

        loss = self.alpha * ce + self.beta * rce
        return loss


class POLY(nn.Module):
    def __init__(self, num_classes=10, epsilon=1.0):
        super(POLY, self).__init__()
        self.epsilon = epsilon
        self.ce = nn.CrossEntropyLoss()
        self.mae = MAE(num_classes=num_classes)
    
    def forward(self, pred, labels):
        ce = self.ce(pred, labels)
        mae = self.mae(pred, labels)
        loss = ce + self.epsilon * mae
        return loss

        
class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss


class NCEandRCE(nn.Module):
    def __init__(self, num_classes=10, alpha=1., beta=1.):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCE(num_classes=num_classes, scale=alpha)
        self.rce = RCE(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class TCE(nn.Module):
    def __init__(self, num_classes=10, order=6):
        super(TCE, self).__init__()
        self.order = order
        self.num_classes = num_classes

    def forward(self, pred, labels):
        batch_size = len(pred)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-6, max=1.0)
        pred = pred[torch.arange(batch_size), labels].unsqueeze(1)
        matrix = torch.arange(1, self.order + 1).unsqueeze(0).cuda()
        loss = torch.sum((1 - pred) ** matrix / matrix, dim=1)
        return loss.mean()


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=eps).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, dim=1)
    return output.mean()


class JS(torch.nn.Module):
    def __init__(self, num_classes, pi):
        super(JS, self).__init__()
        self.num_classes = num_classes
        self.weights = [pi, 1 - pi]
        self.scale = -1.0 / ((1.0 - self.weights[0]) * np.log((1.0 - self.weights[0])))

    def forward(self, pred, labels):
        preds = [F.softmax(pred, dim=1)]
        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(eps, 1.0).log()

        jsw = sum([w * custom_kl_div(mean_distrib_log, d) for w, d in zip(self.weights, distribs)])
        return self.scale * jsw


class AGCE(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, scale=1.):
        super(AGCE, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return self.scale * loss.mean()


class NCEandAGCE(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, alpha=1., beta=1.):
        super(NCEandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCE(num_classes=num_classes, scale=alpha)
        self.agce = AGCE(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()


class SR(nn.Module):
    def __init__(self, lamb, tau, p):
        super(SR, self).__init__()
        self.lamb = lamb
        self.tau = tau
        self.criterion = nn.CrossEntropyLoss()
        self.norm = pNorm(p)

    def forward(self, pred, labels):
        pred = F.normalize(pred, dim=1)
        loss = self.criterion(pred / self.tau, labels) + self.lamb * self.norm(pred / self.tau)
        return loss


class BTL(nn.Module):
    def __init__(self, num_classes, t1, t2, num_iters=5):
        super(BTL, self).__init__()
        self.num_classes = num_classes
        self.t1, self.t2 = t1, t2
        self.num_iters = num_iters

    def forward(self, pred, labels):
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        probabilities = self.tempered_softmax(pred, self.t2, self.num_iters)
        temp1 = (self.log_t(labels + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * labels
        temp2 = (1 / (2 - self.t1)) * (torch.pow(labels, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2
        return torch.mean(torch.sum(loss_values, dim=-1))

    def log_t(self, u, t):
        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(self, u, t):
        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    def compute_normalization_fixed_point(self, activations, t, num_iters=5):
        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(self.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -self.log_t(1.0 / logt_partition, t) + mu

    def compute_normalization(self, activations, t, num_iters=5):
        if t < 1.0:
            return None
        else:
            return self.compute_normalization_fixed_point(activations, t, num_iters)

    def tempered_softmax(self, activations, t, num_iters=5):
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = self.compute_normalization(activations, t, num_iters)

        return self.exp_t(activations - normalization_constants, t)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
