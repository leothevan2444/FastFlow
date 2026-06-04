import argparse
import os

import torch
import yaml
from ignite.contrib import metrics
from tqdm.auto import tqdm

import constants as const
import dataset
import fastflow
import utils


def resolve_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(
    dataloader, model, optimizer, epoch, total_epochs, device, log_interval
):
    model.train()
    loss_meter = utils.AverageMeter()
    progress = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Epoch {}/{}".format(epoch + 1, total_epochs),
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for step, data in progress:
        # forward
        data = data.to(device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % log_interval == 0 or (step + 1) == len(dataloader):
            progress.set_postfix(
                loss="{:.3f}".format(loss_meter.val),
                avg="{:.3f}".format(loss_meter.avg),
            )


def eval_once(dataloader, model, device):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc="     Eval",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for data, targets in progress:
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    tqdm.write("     AUROC: {}".format(auroc))


def train(args):
    device = resolve_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.join(
        args.checkpoint_dir, "exp%d" % len(os.listdir(args.checkpoint_dir))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.to(device)
    print("Using device: {}".format(device))

    for epoch in range(args.epochs):
        train_one_epoch(
            train_dataloader,
            model,
            optimizer,
            epoch,
            args.epochs,
            device,
            args.log_interval,
        )
        if (epoch + 1) % args.eval_interval == 0:
            eval_once(test_dataloader, model, device)
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    device = resolve_device(args.device)
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.to(device)
    print("Using device: {}".format(device))
    eval_once(test_dataloader, model, device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FastFlow on MVTec-AD dataset",
        add_help=False,
    )
    data_group = parser.add_argument_group("data")
    runtime_group = parser.add_argument_group("runtime")
    train_group = parser.add_argument_group("training")

    runtime_group.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )
    data_group.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config file",
    )
    data_group.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to dataset folder",
    )
    data_group.add_argument(
        "--category",
        type=str,
        required=True,
        help="category name",
    )
    runtime_group.add_argument(
        "--eval",
        action="store_true",
        help="run eval only",
    )
    runtime_group.add_argument(
        "--checkpoint",
        type=str,
        help="path to load checkpoint",
    )
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use, e.g. cuda, cuda:0, cuda:1, or cpu",
    )
    train_group.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="batch size",
    )
    train_group.add_argument(
        "--epochs", type=positive_int, default=500, help="number of epochs"
    )
    train_group.add_argument(
        "--log-interval",
        type=positive_int,
        default=10,
        help="log every N train steps",
    )
    train_group.add_argument(
        "--eval-interval",
        type=positive_int,
        default=10,
        help="eval every N epochs",
    )
    train_group.add_argument(
        "--checkpoint-interval",
        type=positive_int,
        default=1,
        help="save checkpoint every N epochs",
    )
    train_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="_fastflow_experiment_checkpoints",
        help="directory to save training checkpoints",
    )
    args = parser.parse_args()
    if args.eval and args.checkpoint is None:
        parser.error("--checkpoint is required when --eval is set")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
