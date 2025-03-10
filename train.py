import argparse
import os
import wandb

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.utils import initialize_config


def main(config, resume):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])
    
    # set feature extraction configurations
    print(config["model"])
    features_config = config["model"]["args"]["feat_config"]
    config["train_dataset"]["args"]["features_config"] = features_config
    config["validation_dataset"]["args"]["features_config"] = features_config
    config["test_dataset"]["args"]["features_config"] = features_config

    # train dataset
    dataset_train = initialize_config(config["train_dataset"])
    train_dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"],
        collate_fn=dataset_train.collate_fn,
    )
    
    wandb.init(project="SELDGCNN", 
        config=config
    )

    # validation dataset
    dataset_val = initialize_config(config["validation_dataset"])
    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=1,
        batch_size=1,
        collate_fn=dataset_val.collate_fn,
    )

    # test dataset
    dataset_test = initialize_config(config["test_dataset"])
    test_dataloader = DataLoader(
        dataset=initialize_config(config["test_dataset"]),
        num_workers=1,
        batch_size=1,
        collate_fn=dataset_test.collate_fn,
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        test_dataloader=test_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SELDGCNN")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)

