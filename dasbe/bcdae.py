import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math
from train_multi_mnist import salt_pepper_noise

thresh = 1 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()
act_fun = ActFun.apply

def mem_update(ops, x, mem, spike):
    noise = torch.rand(mem.size(), device=device)
    # mem = mem * decay * (1. - spike) + ops(x) + noise
    mem = ops(x) + noise
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

class bcdae(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(bcdae, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, input_size, bias=False)

        # 1 layer version
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU()
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_size, input_size),
        #     nn.Sigmoid()
        # )

        # change 1 layer to 2 layer
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()
        )

        self.fc1.weight.data.normal_(0, 1)
        # self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.normal_(0, 1)
        # self.fc2.bias.data.fill_(0)

        self.mem = torch.zeros(hidden_size, device=device)
        self.f = torch.zeros(hidden_size, device=device)
        self.spike = torch.zeros(hidden_size, device=device )
        self.sample = mem_update


    def forward(self, x, refractory = 0):
        # x = self.encoder(x)
        # x = self.decoder(x)
        self.f -= 1
        mem,spikes = self.sample(self.encoder, x, self.mem, self.spike)
        if refractory>0:
            spikes = torch.where((spikes>0)&(self.f<0),1,0).type(dtype=torch.float32)
            self.f = torch.where(spikes > 0, refractory * torch.ones_like(spikes, device=device), self.f).type(
                dtype=torch.float32)
        x = self.decoder(spikes)
        # x = self.decoder(self.encoder(x)) # test DAE
        return x, spikes,mem

    def samples_write(self, x, epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.cpu()
        W = H = int(math.sqrt(x.shape[1]))    
        noise_x = copy.deepcopy(x)
        noise_x = salt_pepper_noise(noise_x)
        noise_x = noise_x.to(device)
        samples,_,_ = self.forward(noise_x)
        #pdb.set_trace()
        samples = samples.data.cpu().numpy()[:64]
        fig = plt.figure(figsize=(4, 12))
        gs = gridspec.GridSpec(24, 8)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(H, W), cmap='Greys_r')

        for i, sample in enumerate(noise_x[:64]):
            ax = plt.subplot(gs[i + 8 * 8])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.cpu().detach().numpy().reshape(H, W), cmap='Greys_r')

        for i, sample in enumerate(x[:64]):
            ax = plt.subplot(gs[i + 8 * 8 * 2])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.cpu().detach().numpy().reshape(H, W), cmap='Greys_r')

        if not os.path.exists('out_bcdae/'):
            os.makedirs('out_bcdae/')
        plt.savefig('out_bcdae/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
		#self.c += 1
        plt.close(fig)

mse_loss = nn.BCELoss(size_average = False)

def loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W[0])**2, dim=1) + torch.sum(Variable(W[1])**2, dim=1) + torch.sum(Variable(W[2])**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

# if __name__ == "__main__":
#     net = bcdae(28*28, 400)
#     x = torch.rand((10, 28*28), dtype=torch.float32)
#     y = net(x)
