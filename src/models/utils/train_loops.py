from tqdm import tqdm

from ..utils.types_ import Dataloader, Device, LossFunction, Model, Optimizer


# * weight_decay é L2 penalty


def vae_train_loop(
    model: Model,
    dataloader: Dataloader,
    optimizer: Optimizer,
    loss_function: LossFunction,
    device: Device,
):
    dataloader_len = len(dataloader)

    r_bar = "| {n_fmt}/{total_fmt} [{postfix}]"

    loop = tqdm(
        iterable=enumerate(dataloader),
        total=dataloader_len,
        bar_format="{l_bar}{bar}" + r_bar,
    )

    train_losses = []

    for i, (X, _) in loop:
        X_original = X.to(device)

        X_reconstucted, miu, sigma = model(X_original)

        loss = loss_function(X_reconstucted, X_original, miu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

        train_losses.append(loss.item())

    return train_losses
