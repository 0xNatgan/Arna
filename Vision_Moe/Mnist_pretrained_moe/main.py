import train_experts
import gating_Mnist

if __name__ == "__main__":
    train_experts.train_experts(epochs=5)
    train_experts.train_dense_expert(epochs=8)
    experts_paths = {'normal': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_normal.pth',
                    'angle': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_angle.pth',
                    'noise': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_noise.pth',
                    'blur': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_blur.pth'}
    gating_Mnist.train_gating_network(experts_paths, epochs=10)