import gc
import os
from dataclasses import dataclass
from pathlib import Path

import ignite
import numpy as np
import pandas as pd
import pmlb
import torch
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
from ignite.engine import Events
from ignite.handlers import LRScheduler
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from artificial_genome_synthesis.manage_data import Genotype
from artificial_genome_synthesis.models.GANs import (
    GAN_Discriminator,
    GAN_Generator,
)
from artificial_genome_synthesis.models.trainers import gan_trainer_setup
from artificial_genome_synthesis.utils.load_env import ENV_CONFIG

mlrun_path = Path(ENV_CONFIG["MLRUNS_PATH"])
os.environ["MLFLOW_TRACKING_URI"] = mlrun_path.as_uri()


# HyperParameter values
@dataclass
class HyperParameters:
    epochs: int = 100

    # Generator learning rate
    generator_learning_rate: float = 1e-4

    # discriminator learning rate
    discriminator_learning_rate: float = 8e-4
    batch_size: int = 32
    l2_penalty: float = 1e-3

    # size of noise input
    latent_size: int = 600

    # alpha value for LeakyReLU
    alpha: float = 0.01

    # number of artificial genomes (haplotypes) to be created
    ag_size: int = 500

    # Betas for optimizer
    b1 = 0.5
    b2 = 0.999


save_that = 20  # epoch interval for saving outputs

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}")

# ## Custom seed for reproductible results

random_seed = int(ENV_CONFIG["RANDOM_SEED"])

np.random.seed(random_seed)
torch.manual_seed(random_seed)
g = torch.Generator()
g.manual_seed(random_seed)


print(f"Random Seed: {random_seed}")

# # Loading and quick data processing

inpt_file = Path(ENV_CONFIG["CONTROL_INPUT_DATA"])
labs_file = Path(ENV_CONFIG["SUBJECTS_FILE"])

df = pd.read_pickle(inpt_file).iloc[:, :7_502]
df.head()

df_noname = (
    df.sample(frac=1, random_state=random_seed)
    .reset_index(drop=True)
    .drop(df.columns[0:2], axis=1)
    .pipe(lambda x: x / 2)
    .pipe(lambda x: x - np.random.uniform(0, 0.1, size=x.shape))
    .astype(np.float16)
)

df_noname.shape

gc.collect()

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
X_noname.to_csv("yeah_pmlb.hapt", sep="\t")

# ## Dataset and Dataloader setup

control_genotype = Genotype(X_noname)

genotype_dataloader = control_genotype.make_dataloader(
    batch_size=HyperParameters.batch_size,
    shuffle=True,
    drop_last=True,
    generator=g,
    num_workers=8,
    pin_memory=True,
)

X = next(iter(genotype_dataloader))

# ## Set up the Generator and Discriminator

generator = GAN_Generator(
    latent_size=HyperParameters.latent_size,
    features=X.shape[1],
    alpha=HyperParameters.alpha,
).to(dev)

discriminator = GAN_Discriminator(
    features=X.shape[1],
    alpha=HyperParameters.alpha,
).to(dev)

generator = torch.compile(generator)
discriminator = torch.compile(discriminator)

print(generator)

print(discriminator)

# ## Specify the Optimizer parameters for both the models and the loss function to be used

gen_optimizer = Adam(
    params=generator.parameters(),
    lr=HyperParameters.generator_learning_rate,
    weight_decay=HyperParameters.l2_penalty,
    betas=[HyperParameters.b1, HyperParameters.b2],
)

disc_optimizer = Adam(
    params=discriminator.parameters(),
    lr=HyperParameters.discriminator_learning_rate,
    weight_decay=HyperParameters.l2_penalty,
    betas=[HyperParameters.b1, HyperParameters.b2],
)

# gen_lr_scheduler = ReduceLROnPlateau(gen_optimizer, mode = "min", patience =5)
# disc_lr_scheduler = ReduceLROnPlateau(disc_optimizer, mode = "min", patience=5)

# gen_scheduler = LRScheduler(gen_lr_scheduler)
# disc_scheduler = LRScheduler(disc_lr_scheduler)

loss = BCELoss()

# ## Training the models


trainer = gan_trainer_setup(
    generator=generator,
    discriminator=discriminator,
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    loss_function=loss,
    latent_size=HyperParameters.latent_size,
    n_ags=HyperParameters.ag_size,
    original_data=X_noname,
    labels=y,
    device=dev,
)

# trainer.add_event_handler(Events.ITERATION_COMPLETED, gen_scheduler)
# trainer.add_event_handler(Events.ITERATION_COMPLETED, disc_scheduler)

mlflow_logger = MLflowLogger(tracking_uri=mlrun_path)

mlflow_logger.log_params(
    {
        "Seed": random_seed,
        "Batch Size": HyperParameters.batch_size,
        "Generator": generator,
        "Discriminator": discriminator,
        "Torch version": torch.__version__,
        "Cuda version": torch.version.cuda,
        "Ignite version": ignite.__version__,
        "Discriminator Learning Rate": HyperParameters.discriminator_learning_rate,
        "Generator Learning Rate": HyperParameters.generator_learning_rate,
    }
)

mlflow_logger.attach_output_handler(
    engine=trainer,
    event_name=Events.EPOCH_COMPLETED,
    tag="Epoch Training",
    metric_names=["Discriminator Loss", "Generator Loss"],
)

end_state = trainer.run(genotype_dataloader, HyperParameters.epochs)

mlflow_logger.close()

torch.save(generator, "models/GAN_generator.pkl")
torch.save(discriminator, "models/GAN_discriminator.pkl")
