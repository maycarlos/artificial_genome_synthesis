import argparse

from .types_ import WGAN


def get_cmd_args():
    """
    Retrieves the command line arguments for the training
    """
    parser = argparse.ArgumentParser("artificial genome synthesis")

    train_parser = parser.add_argument_group("train")

    wgan_type = {"cp": WGAN.CP, "gp": WGAN.GP}

    train_parser.add_argument(
        "--wgan_type",
        type=str,
        default="cp",
        choices=["cp", "gp"],
        help="Select the type of training for the WGAN",
    )

    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of epochs of training",
    )

    train_parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="Batch size for the dataloader",
    )

    train_parser.add_argument(
        "--num_workers",
        "-a",
        type=int,
        default=1,
        help="Paralelize the dataloaders for faster load",
    )

    train_parser.add_argument(
        "--l2_pen",
        "-lp",
        type=float,
        default=1e-3,
        help="L2 Regularization",
    )

    train_parser.add_argument(
        "--generator_lr",
        "-glr",
        type=float,
        default=1e-4,
        help="Learning rate for the generator",
    )

    train_parser.add_argument(
        "--discriminator_lr",
        "-dlr",
        type=float,
        default=4e-4,
        help="Learning rate for the discriminator",
    )

    train_parser.add_argument(
        "--leaky_alpha",
        "-la",
        type=float,
        default=0.01,
        help="Leaky RElu alpha",
    )

    train_parser.add_argument(
        "--beta1",
        "-b1",
        type=float,
        default=0.5,
        help="First beta for the adam optimizer",
    )

    train_parser.add_argument(
        "--beta2",
        "-b2",
        type=float,
        default=0.999,
        help="Second beta for the adam optimizer",
    )

    train_parser.add_argument(
        "--n_critic",
        "-nc",
        type=int,
        default=5,
        help="Number of time the critic is trained",
    )

    train_parser.add_argument(
        "--clip_val",
        "-cv",
        type=float,
        default=1.0,
        help="Value for gradient clipping",
    )

    train_parser.add_argument(
        "--lambda_gp",
        "-lgp",
        type=float,
        default=10,
        help="lambda for wgan gp",
    )

    train_parser.add_argument(
        "--save_interval",
        "-s",
        type=int,
        default=5,
        help="save interval",
    )

    train_parser.add_argument(
        "--random_seed",
        type=int,
        default=123,
        help="seed random number generator",
    )

    params = parser.parse_args()

    params.wgan_type = wgan_type[params.wgan_type]

    return params
