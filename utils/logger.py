def init_history() -> dict:
    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
    }

    return history


def log_epoch(
    history: dict[str, list[float]],
    train_loss: float,
    train_acc: float,
    eval_loss: float,
    eval_acc: float,
):
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["eval_loss"].append(eval_loss)
    history["eval_acc"].append(eval_acc)
