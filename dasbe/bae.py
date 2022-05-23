import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # mem = mem * decay * (1. - spike) + ops(x) + noise
    p = ops(x)
    noise = torch.rand(p.size(), device=device)
    mem = p + noise
    spike = act_fun(mem) # act_fun : approximation firing function
    return p, spike

class bae(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(bae, self).__init__()

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, input_size)

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     #nn.ReLU()
        #     nn.Sigmoid()
        # )
        e_hidden_size, d_hidden_size = hidden_size #d_hidden_size[0] = e_hidden_size[-1]
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
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_size, input_size),
        #     nn.Sigmoid()
        # )
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


        # self.fc1.weight.data.normal_(0, 1)
        # self.fc1.bias.data.fill_(0)
        # self.fc2.weight.data.normal_(0, 1)
        # self.fc2.bias.data.fill_(0)

    # def encoder(self, x):
    #     return F.relu(self.fc1(x))

    # def decoder(self, x):
    #     return torch.sigmoid(self.fc2(x))

    def forward2(self, x, refractory = 0):
        self.f -= 1
        mem,spikes = self.sample(self.encoder, x, self.mem, self.spike)
        if refractory>0:
            spikes = torch.where((spikes>0)&(self.f<0),1,0).type(dtype=torch.float32)
            self.f = torch.where(spikes > 0, refractory * torch.ones_like(spikes, device=device), self.f).type(
                dtype=torch.float32)
        #x = self.decoder(spikes)
        x = self.decoder(self.encoder(x)) # test DAE
        return x, spikes,mem
    
    def forward(self, x, refractory = 0):
        hidden = self.encoder(x)
        x = self.decoder(hidden) # test DAE
        return x, hidden, hidden

class bae_cnn(nn.Module):
    def __init__(self, input_size):
        super(bae_cnn, self).__init__()
        self.input_size = input_size
        self.encoder_seq1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # (b_s, 64, 14, 14)
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=1, padding=1), 
            nn.Conv2d(64, 32, 3, stride=2, padding=1), # (b_s, 32, 7, 7)
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=1, padding=1),  
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # (b_s, 32, 4, 4)
            nn.ReLU(),
        )

        self.encoder_hidden_size = 32 * 4 * 4
        
        self.encoder_seq2 = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 1000),
            # nn.ReLU(),
            # nn.Linear(256, 128),
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
            # nn.ReLU(),
            # nn.Linear(decoder_hiddens[2], decoder_hiddens[3]),
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
        x = self.decoder(hidden) # test DAE
        return x, hidden, hidden
        

if __name__ == "__main__":
    net = bae(400, 100)
    x = torch.rand((1, 400), dtype=torch.float32)
    y = net(x)
