import torch

def configure_optimizers(
    params,
    lr: float,
    beta_1: float,
    beta_2: float,
    weight_decay: float,
    **kwargs,  # accept any params
):
    decay = []
    no_decay = []
    for name, m in params:
        if "channel_embed" in name or "pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": lr,
                "betas": (beta_1, beta_2),
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "lr": lr,
                "betas": (beta_1, beta_2),
                "weight_decay": 0,
            },
        ]
    )
    return optimizer



__all__ = ["configure_lr_scheduler"]
