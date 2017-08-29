from torchvision.models.vgg import model_urls
from torch import nn
import torchvision
    
class VGG(nn.Module):
    
    def __init__(self, n_features = 5):
        
        super(VGG, self).__init__()
        
        model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
        vgg = torchvision.models.vgg19_bn(pretrained=True)
        
        self.l_00 = list(vgg.features.children())[0]
        self.l_01 = list(vgg.features.children())[1]
        self.l_02 = list(vgg.features.children())[2]
        self.l_03 = list(vgg.features.children())[3]
        self.l_04 = list(vgg.features.children())[4]
        self.l_05 = list(vgg.features.children())[5]
        self.l_06 = list(vgg.features.children())[6]
        self.l_07 = list(vgg.features.children())[7]
        self.l_08 = list(vgg.features.children())[8]
        self.l_09 = list(vgg.features.children())[9]
        self.l_10 = list(vgg.features.children())[10]
        self.l_11 = list(vgg.features.children())[11]
        self.l_12 = list(vgg.features.children())[12]
        self.l_13 = list(vgg.features.children())[13]
        self.l_14 = list(vgg.features.children())[14]
        self.l_15 = list(vgg.features.children())[15]
        self.l_16 = list(vgg.features.children())[16]

    def forward(self, x):
        
        x  = self.l_00(x)
        x  = self.l_01(x)
        c1 = self.l_02(x)
        x  = self.l_03(x)
        x  = self.l_04(x)
        c2 = self.l_05(x)
        x  = self.l_06(c2)
        x  = self.l_07(x)
        x  = self.l_08(x)
        c3 = self.l_09(x)
        x  = self.l_10(c3)
        x  = self.l_11(x)
        c4 = self.l_12(x)
        x  = self.l_13(c4)
        x  = self.l_14(x)
        x  = self.l_15(x)
        c5  = self.l_16(x)
        
        return c1, c2, c3, c4, c5
    