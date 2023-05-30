import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path

import ignite
import numpy as np
import pmlb
import torch
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import Events
from torch.nn import BCELoss
from torch.optim import RMSprop

from artificial_genome_synthesis.manage_data import Genotype
from artificial_genome_synthesis.models.GANs import WGAN_Critic, WGAN_Generator
from artificial_genome_synthesis.models.trainers import wgan_cp_trainer_setup
from artificial_genome_synthesis.utils import ENV_CONFIG

mlrun_path = Path(ENV_CONFIG["MLRUNS_PATH"])
os.environ["MLFLOW_TRACKING_URI"] = mlrun_path.as_uri()


# HyperParameter values
@dataclass
class HyperParameters:
    epochs: int = 100

    # Generator learning rate
    generator_learning_rate: float = 1e-4

    # discriminator learning rate
    discriminator_learning_rate: float = 1e-4

    b1: float = 0.5
    b2: float = 0.999

    batch_size: int = 32
    l2_penalty: float = 1e-3

    # size of noise input
    latent_size: int = 600

    # alpha value for LeakyReLU
    alpha: float = 0.01

    # number of artificial genomes (haplotypes) to be created
    ag_size: int = 500


save_that = 20  # epoch interval for saving outputs

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}")

# Custom seed for reproductible results

random_seed = int(ENV_CONFIG["RANDOM_SEED"])

np.random.seed(random_seed)
torch.manual_seed(random_seed)
g = torch.Generator()
g.manual_seed(random_seed)


print(f"Random Seed: {random_seed}")

# Loading and quick data processing
snp_data = pmlb.fetch_data("GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1")

X_raw = snp_data.iloc[:, :-1]
y = snp_data.iloc[:, -1]

X_noname = (
    X_raw.sample(frac=1, random_state=123)
    .reset_index(drop=True)
    .pipe(lambda x: x / 2)
    .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
    .astype(np.float16)
)

X_noname.describe()

gc.collect()

genotype = Genotype(X_noname)

genotype_dataloader = genotype.make_dataloader(
    batch_size=HyperParameters.batch_size,
    shuffle=True,
    drop_last=True,
    generator=g,
    num_workers=8,
    pin_memory=True,
)

X = next(iter(genotype_dataloader))

# Set up the Generator and Discriminator
generator = WGAN_Generator(
    latent_size=HyperParameters.latent_size,
    features=X.shape[1],
    alpha=HyperParameters.alpha,
).to(dev)

critic = WGAN_Critic(features=X.shape[1], alpha=HyperParameters.alpha).to(dev)

generator = torch.compile(generator)
critic = torch.compile(critic)

# print(generator)
# print(critic)

# Specify the Optimizer parameters for both the models and the loss function to be used
gen_optimizer = RMSprop(
    params=generator.parameters(),
    lr=HyperParameters.generator_learning_rate,
    # weight_decay=HyperParameters.l2_penalty,
    # betas=(HyperParameters.b1, HyperParameters.b2),
)

crit_optimizer = RMSprop(
    params=critic.parameters(),
    lr=HyperParameters.discriminator_learning_rate,
    # weight_decay=HyperParameters.l2_penalty,
    # betas=(HyperParameters.b1, HyperParameters.b2),
)

loss = BCELoss()

# Training the models
trainer = wgan_cp_trainer_setup(
    generator=generator,
    critic=critic,
    generator_optimizer=gen_optimizer,
    critic_optimizer=crit_optimizer,
    latent_size=HyperParameters.latent_size,
    n_ags=HyperParameters.ag_size,
    original_data=X_raw,
    labels=y,
    device=dev,
)

# mlflow tracking
mlflow_logger = MLflowLogger(tracking_uri=mlrun_path)

modules_versions = {k: v for k, v in ENV_CONFIG.items() if re.search("version$", k)}

mlflow_logger.log_params(
    {
        "Seed": random_seed,
        "Batch Size": HyperParameters.batch_size,
        "Generator": generator,
        "Critic": critic,
        "Torch version": torch.__version__,
        "Cuda version": torch.version.cuda,
        "Ignite version": ignite.__version__,
        "Discriminator Learning Rate": HyperParameters.discriminator_learning_rate,
        "Generator Learning Rate": HyperParameters.generator_learning_rate,
        **modules_versions,
    }
)

mlflow_logger.attach_output_handler(
    engine=trainer,
    event_name=Events.EPOCH_COMPLETED,
    tag="Epoch Training",
    metric_names=["Critic Loss", "Generator Loss"],
)

end_state = trainer.run(genotype_dataloader, HyperParameters.epochs)

# mlflow_logger.close()

torch.save(generator, "models/WGAN_generator.pkl")
torch.save(critic, "models/WGAN_critic.pkl")
