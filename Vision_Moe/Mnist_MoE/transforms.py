import torch
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transforms():
    return transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=45)], p=0.07),
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.07),
        transforms.RandomApply([AddGaussianNoise(0., 0.2)], p=0.07),
        transforms.Normalize((0.1307,), (0.3081,))
    ])