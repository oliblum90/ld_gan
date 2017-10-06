from torchvision.models.vgg import model_urls
from torch import nn
import torchvision
    
RETURN_LAYERS = [5, 12, 19, 32, 45]
    
class VGG(nn.Module):
    
    def __init__(self):
        
        super(VGG, self).__init__()
        
        model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
        self.vgg = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg.cuda()
        self.vgg.eval()
        

    def forward(self, x, return_layers = RETURN_LAYERS):
        
        features = []
        n_layers = max(return_layers)
        for l in range(n_layers + 1):
            x = list(self.vgg.features.children())[l](x)
            if l in return_layers:
                features.append(x)
        
        return features
    