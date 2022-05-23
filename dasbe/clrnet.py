import torch
import torch.nn as nn

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


def mem_update(ops, x):
    """
    approximate Bornoulli distribution
    """
    input_ = ops(x)
    noise = torch.rand(input_.size(), device=device)
    mem = input_ + noise
    spike = act_fun(mem) # act_fun : approximation firing function
    return input_, spike


class CLRNET(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CLRNET, self).__init__()

        e_hidden_size, d_hidden_size = hidden_size 
        print('e_hidden_size, d_hidden_size',e_hidden_size, d_hidden_size)

        # encoder
        self.encoder = nn.Sequential()
        size1 = input_size
        for hs in range(len(e_hidden_size)):
            if hs<len(e_hidden_size)-1:
                self.encoder.add_module('eLinear{}'.format(hs),nn.Linear(size1, e_hidden_size[hs]))
                self.encoder.add_module('eReLU{}'.format(hs), nn.ReLU())
            else:
                print('hs1=',hs)
                self.encoder.add_module('eLinear{}'.format(hs), nn.Linear(size1, e_hidden_size[hs]))
                self.encoder.add_module('eSigmoid{}'.format(hs), nn.Sigmoid())
            size1 = e_hidden_size[hs]

        # 隐层相关参数
        self.hidden_size = e_hidden_size[-1]
        self.f = torch.zeros(e_hidden_size[-1], device=device)  # refractory flag for sample i

        # 膜电位更新
        self.sample = mem_update

        # decoder
        hs = 0
        self.decoder = nn.Sequential()
        for hs in range(len(d_hidden_size)-1):
            self.decoder.add_module('dLinear{}'.format(hs), nn.Linear(d_hidden_size[hs], d_hidden_size[hs+1]))
            self.decoder.add_module('dReLU{}'.format(hs), nn.ReLU())
        print('hs2=',hs, ', total_len=',len(d_hidden_size))
        if len(d_hidden_size) >1:
            self.decoder.add_module('dLinear{}'.format(hs+1), nn.Linear(d_hidden_size[-1], input_size))
            self.decoder.add_module('dSigmoid{}'.format(hs), nn.Sigmoid())
        else:
            self.decoder.add_module('dLinear{}'.format(hs), nn.Linear(d_hidden_size[-1], input_size))
            self.decoder.add_module('dSigmoid{}'.format(hs), nn.Sigmoid())

    def forward(self, x, refractory = 0):
        self.f -= 1
        mem, spikes = self.sample(self.encoder, x)
        if refractory > 0:
            spikes = torch.where((spikes > 0) & (self.f < 0), 1, 0).type(dtype=torch.float32)
            self.f = torch.where(spikes > 0, refractory * torch.ones_like(spikes, device=device), self.f).type(
                dtype=torch.float32)

        out = self.decoder(spikes)

        return out, spikes, mem
