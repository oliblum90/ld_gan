import torch.nn as nn
from init_weights import init_weights



class enc_64(nn.Module):
    
    def __init__(self, 
                 complexity = 64,
                 n_col = 3,
                 n_features = 256):
        
        super(enc_64, self).__init__()
        self.main = nn.Sequential(
            # BLOCK 0: 64 --> 32
            nn.Conv2d(n_col, complexity, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 1: 32 --> 16
            nn.Conv2d(complexity, complexity * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 2: 16 --> 8
            nn.Conv2d(complexity * 2, complexity * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 3: 8 --> 4
            nn.Conv2d(complexity * 4, complexity * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # BLOCK 4: 4 --> 1
            nn.Conv2d(complexity * 8, n_features, 4, 1, 0, bias=False),
            nn.Tanh()
        )
        self.main.apply(init_weights)
        self.main.cuda()

    def forward(self, input):
        output = self.main(input)
        return output
    

    
    
    
    
class vae_enc_64(nn.Module):
    
    def __init__(self, 
                 complexity = 64,
                 n_col = 3,
                 n_features = 256):
        
        super(enc_64, self).__init__()

        # BLOCK 0: 64 --> 32
        b00 = nn.Conv2d(n_col, complexity, 4, 2, 1, bias=False),
        b01 = nn.LeakyReLU(0.2, inplace=True),
        # BLOCK 1: 32 --> 16
        b10 = nn.Conv2d(complexity, complexity * 2, 4, 2, 1, bias=False),
        b11 = nn.BatchNorm2d(complexity * 2),
        b12 = nn.LeakyReLU(0.2, inplace=True),
        # BLOCK 2: 16 --> 8
        b20 = nn.Conv2d(complexity * 2, complexity * 4, 4, 2, 1, bias=False),
        b21 = nn.BatchNorm2d(complexity * 4),
        b22 = nn.LeakyReLU(0.2, inplace=True),
        # BLOCK 3: 8 --> 4
        b30 = nn.Conv2d(complexity * 4, complexity * 8, 4, 2, 1, bias=False),
        b31 = nn.BatchNorm2d(complexity * 8),
        b32 = nn.LeakyReLU(0.2, inplace=True),
        # BLOCK 4: 4 --> 1
        m40 = nn.Conv2d(complexity * 8, n_features, 4, 1, 0, bias=False),
        m41 = nn.tanh()
        
        v40 = nn.Conv2d(complexity * 8, n_features, 4, 1, 0, bias=False),
        v41 = nn.tanh()


    def forward(self, input):
        temp = b00(b01(b01(b10(b11(b12(b20(b21(b22(b30(b31(b32(input))))))))))))
        mean = m40(m41(temp))
        var = v40(v41(temp))
        return mean, var
    
    
    
    
    
    
    
    