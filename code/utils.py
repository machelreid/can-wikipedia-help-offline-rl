from torch import optim
from itertools import chain


def get_optimizer(args, model):
    if args["pretrained_lm"]:
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" not in str(type(module)))
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
    return optimizer
