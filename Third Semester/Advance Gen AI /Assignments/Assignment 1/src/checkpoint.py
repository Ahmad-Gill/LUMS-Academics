import torch


def save_checkpoint(
    model,
    optimizer,
    next_step,
    train_generator,
    val_generator,
    out,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src,
    model,
    optimizer,
    train_generator,
    val_generator,
):
    checkpoint = torch.load(
        src,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    train_generator.set_state(
        checkpoint["train_generator"]
    )

    val_generator.set_state(
        checkpoint["val_generator"]
    )

    return checkpoint["next_step"]