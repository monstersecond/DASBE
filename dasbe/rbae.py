import torch
import torch.nn as nn
import torch.nn.functional as F

thresh = 1 
lens = 0.5 
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
    
    mem = ops(x) + noise
    spike = act_fun(mem) 
    return mem, spike

class rbae(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(rbae, self).__init__()

        e_hidden_size, d_hidden_size = hidden_size 
        print('e_hidden_size, d_hidden_size', e_hidden_size, d_hidden_size)
        self.encoder = nn.Sequential()
        size1 = input_size
        for hs in range(len(e_hidden_size)):
            if hs < len(e_hidden_size) - 1:
                self.encoder.add_module('eLinear{}'.format(hs), nn.Linear(size1, e_hidden_size[hs]))
                self.encoder.add_module('eReLU{}'.format(hs), nn.ReLU())
            else:
                print('hs1=', hs)
                self.encoder.add_module('eLinear{}'.format(hs), nn.Linear(size1, e_hidden_size[hs]))
                self.encoder.add_module('eSigmoid{}'.format(hs), nn.Sigmoid())
            size1 = e_hidden_size[hs]

        
        self.hidden_weight = nn.Parameter(torch.zeros((e_hidden_size[-1], e_hidden_size[-1])), requires_grad=True)
        self.hidden_weight.data.normal_(0, 1)
        self.hidden_bias = nn.Parameter(torch.zeros((e_hidden_size[-1],)), requires_grad=True)
        self.hidden_activate = nn.Sigmoid()
        self.hidden_size = e_hidden_size[-1]

        self.mem = torch.zeros(e_hidden_size[-1], device=device)
        self.f = torch.zeros(e_hidden_size[-1], device=device)
        self.spike = torch.zeros(e_hidden_size[-1], device=device )
        self.sample = mem_update

        hs = 0
        self.decoder = nn.Sequential()
        for hs in range(len(d_hidden_size)-1):
            self.decoder.add_module('dLinear{}'.format(hs), nn.Linear(d_hidden_size[hs], d_hidden_size[hs+1]))
            self.decoder.add_module('dReLU{}'.format(hs), nn.ReLU())
        print('hs2=', hs, ', total_len=', len(d_hidden_size))
        if len(d_hidden_size) >1:
            self.decoder.add_module('dLinear{}'.format(hs+1), nn.Linear(d_hidden_size[-1], input_size))
            self.decoder.add_module('dSigmoid{}'.format(hs), nn.Sigmoid())
        else:
            self.decoder.add_module('dLinear{}'.format(hs), nn.Linear(d_hidden_size[-1], input_size))
            self.decoder.add_module('dSigmoid{}'.format(hs), nn.Sigmoid())

    def reset(self):
        self.mem = torch.zeros(self.hidden_size, device=device)
        self.f = torch.zeros(self.hidden_size, device=device)
        
    def forward2(self, x, pre_spikes, refractory=0):
        """
        input:
            x: (batch_size, time_window_size, H * W)  0-1 matrix
            pre_spikes: (batch_size, hidden_size)  0-1 matrix
        output:
            outputs: (batch_size * time_window_size, H * W)  
            spikes: (batch_size, hidden_size)  
        """
        outputs = []
        spikes = pre_spikes
        for t in range(x.shape[1]):
            self.f -= 1  
            _, cur_spikes = self.sample(self.encoder, x[:, t], self.mem, self.spike)  
            if refractory > 0:  
                cur_spikes = torch.where((cur_spikes>0)&(self.f<0), 1, 0).type(dtype=torch.float32)
                self.f = torch.where(cur_spikes > 0, refractory*torch.ones_like(cur_spikes, device=device), self.f).type(dtype=torch.float32)
            
            output = self.decoder(self.hidden_activate(spikes.matmul(self.hidden_weight) + cur_spikes + self.hidden_bias))
            spikes = cur_spikes
            outputs.append(output.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1), spikes

    def forward(self, x, pre_spikes):
        """
        input:
            x: (batch_size, time_window_size, H * W)
            pre_spikes: (batch_size, hidden_size)
        output:
            outputs: (batch_size * time_window_size, H * W)
            h: (batch_size, hidden_size)
        """
        outputs = []
        spikes = pre_spikes
        for t in range(x.shape[1]):
            h = self.encoder(x[:, t])
            output = self.decoder(self.hidden_activate(spikes.matmul(self.hidden_weight) + h + self.hidden_bias))
            spikes = h
            outputs.append(output.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1), h
