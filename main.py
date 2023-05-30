import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser("TRAIN WGAN")

# TODO Ver como posso relacionar este script com o anterior 
parser.add_argument('--epochs', type=int, default=100, help='numer of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for the dataloader')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--l2_pen', type = float, default=1e-3,help ="L2 Regularization") 
parser.add_argument('--g_lr', type=float, default=1e-4, help='use TTUR lr rate for Adam')
parser.add_argument('--d_lr', type=float, default=4e-4, help='use TTUR lr rate for Adam')
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--clip_val', type=float, default=1.0, help="yeah")
parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for wgan gp')

args = parser.parse_args()

@dataclass
class HyperParameters:
    epochs: int = args.epochs

    # Generator learning rate
    generator_learning_rate: float = args.g_lr

    # discriminator learning rate
    discriminator_learning_rate: float = args.d_lr
    batch_size: int = args.batch_size
    l2_penalty: float = 1e-3

    # size of noise input
    latent_size: int = 600

    # alpha value for LeakyReLU
    alpha: float = 0.01
    
    save_interval : int = 10
    adam_betas : tuple[float,float] = (args.beta1,args.beta2,)
    lambda_gp : float = args.lambda_gp
    clip_val : float = args.clip_val


HP = HyperParameters()


if __name__ == "__main__":
    pass

