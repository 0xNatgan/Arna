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

def get_transforms(expert_type):
    """
    Returns the appropriate transforms for the specified expert type.
    """
    # Common normalization for MNIST
    normalization = transforms.Normalize((0.1307,), (0.3081,))
    
    match expert_type:

        case 'normal':
            return transforms.Compose([
                transforms.ToTensor(),
                normalization
            ])
        
        case 'angle':
            return transforms.Compose([
                transforms.RandomRotation(degrees=35),
                transforms.ToTensor(),
                normalization
            ])
            
        case 'blur':
            return transforms.Compose([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                normalization
            ])
            
        case 'noise':
            return transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., 0.4),
                normalization
            ])
        
        case 'dense':
            # Applies transforms with probability p=0.25 individually
            return transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(degrees=45)], p=0.25),
                transforms.ToTensor(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
                transforms.RandomApply([AddGaussianNoise(0., 0.2)], p=0.25),
                normalization
            ])