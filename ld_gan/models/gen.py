import torch.nn as nn
from init_weights import init_weights



class gen_64(nn.Module):
    
    def __init__(self, 
                 latent_size = 100,
                 complexity = 64,
                 n_col = 3):
        
        super(gen_64, self).__init__()
        self.main = nn.Sequential(
            # BLOCK 0: 1 --> 4
            nn.ConvTranspose2d(latent_size, complexity*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(complexity*8),
            nn.ReLU(True),
            # BLOCK 1: 4 --> 8
            nn.ConvTranspose2d(complexity*8, complexity*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity*4),
            nn.ReLU(True),
            # BLOCK 2: 8 --> 16
            nn.ConvTranspose2d(complexity*4, complexity*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity*2),
            nn.ReLU(True),
            # BLOCK 3: 16 --> 32
            nn.ConvTranspose2d(complexity*2, complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(complexity),
            nn.ReLU(True),
            # BLOCK 4: 32 --> 64
            nn.ConvTranspose2d(complexity, n_col, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.main.apply(init_weights)
        self.main.cuda()

    def forward(self, input):
        output = self.main(input)
        return output
