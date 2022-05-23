"""
ref: https://github.com/Spijkervet/SimCLR
"""
import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NT_Xent_batch(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent_batch, self).__init__()
        self.temperature = temperature

        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.MSELoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z0, z1, z2):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 3 * z0.shape[0]

        sim01 = self.similarity_f(z0.unsqueeze(1), z1.unsqueeze(0)) / self.temperature
        sim02 = self.similarity_f(z0.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        sim12 = self.similarity_f(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        sim00 = self.similarity_f(z0.unsqueeze(1), z0.unsqueeze(0)) / self.temperature
        sim11 = self.similarity_f(z1.unsqueeze(1), z1.unsqueeze(0)) / self.temperature
        sim22 = self.similarity_f(z2.unsqueeze(1), z2.unsqueeze(0)) / self.temperature

        positive_samples = torch.cat((sim00, sim11, sim22), dim=0).reshape(-1)
        negative_samples = torch.cat((sim01, sim02, sim12), dim=0).reshape(-1)

        # labels = torch.zeros(N).to(positive_samples.device).long()
        # logits = torch.cat((positive_samples, negative_samples), dim=1)
        # loss = self.criterion(logits, labels)
        # loss /= N

        labels_pos = torch.ones(positive_samples.shape[0]).to(positive_samples.device).float()
        loss_pos = self.criterion(positive_samples, labels_pos)
        # loss_pos /= positive_samples.shape[0]

        labels_neg = torch.zeros(negative_samples.shape[0]).to(positive_samples.device).float()
        loss_neg = self.criterion(negative_samples, labels_neg)
        # loss_neg /= negative_samples.shape[0]

        return loss_pos + loss_neg
