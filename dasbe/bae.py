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
    
    p = ops(x)
    noise = torch.rand(p.size(), device=device)
    mem = p + noise
    spike = act_fun(mem) 
    return p, spike


class bae(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(bae, self).__init__()

        e_hidden_size, d_hidden_size = hidden_size 
        print('e_hidden_size, d_hidden_size',e_hidden_size, d_hidden_size)
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

        self.mem = torch.zeros(e_hidden_size[-1], device=device)
        self.f = torch.zeros(e_hidden_size[-1], device=device)
        self.spike = torch.zeros(e_hidden_size[-1], device=device )
        self.sample = mem_update
          
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
        hidden = self.encoder(x)
        x = self.decoder(hidden) 
        return x, hidden, hidden


class bae_cnn(nn.Module):
    def __init__(self, input_size):
        super(bae_cnn, self).__init__()
        self.input_size = input_size
        self.encoder_seq1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  
            nn.ReLU(),
            
            nn.Conv2d(64, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            
            nn.Conv2d(32, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
        )

        self.encoder_hidden_size = 32 * 4 * 4
        
        self.encoder_seq2 = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 1000),
            
            
            nn.Sigmoid()
        )

        self.hidden_size = 1000

        decoder_hiddens = [256, 512, input_size]
        self.decoder_seq = nn.Sequential(
            nn.Linear(self.hidden_size, decoder_hiddens[0]),
            nn.ReLU(),
            nn.Linear(decoder_hiddens[0], decoder_hiddens[1]),
            nn.ReLU(),
            nn.Linear(decoder_hiddens[1], decoder_hiddens[2]),
            
            
            nn.Sigmoid()
        )
    
    def encoder(self, x):
        x = self.encoder_seq1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_seq2(x)
        return x

    def decoder(self, x):
        x = self.decoder_seq(x)
        return x
    
    def forward(self, x, refractory = 0):
        hidden = self.encoder(x)
        x = self.decoder(hidden) 
        return x, hidden, hidden
        

if __name__ == "__main__":
    net = bae(400, 100)
    x = torch.rand((1, 400), dtype=torch.float32)
    y = net(x)
