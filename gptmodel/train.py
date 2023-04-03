from __future__ import annotations
import logging
from typing import Dict

import torch
from torch import optim
from torch.utils.data import DataLoader

from gptmodel.model import GPTModel
from gptmodel.utils import get_datasets


def train(config: Dict[str, dict], device: str, cached=False):
    logging.info("Preparing data")
    hyper_params = config["hyper_params"]
    tokenizer_name = hyper_params["tokenizer"]
    logging.info("Building datasets and vocab using %s tokenizer", tokenizer_name)
    train_dataset, val_dataset, vocab = get_datasets(
        tokenizer_name, hyper_params["block_size"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyper_params["batch_size"],
        shuffle=True,
        pin_memory=True,
        pin_memory_device=device,
        num_workers=3,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyper_params["batch_size"],
        pin_memory=True,
        pin_memory_device=device,
        num_workers=2,
        persistent_workers=True
    )

    model = GPTModel(
        vocab_size=len(vocab),
        embed_size=hyper_params["embed_size"],
        dropout=hyper_params["dropout"],
        num_heads=hyper_params["num_heads"],
        block_size=hyper_params["block_size"],
        num_layers=hyper_params["num_layers"],
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logging.info("Model has %.4f million trainable parameters", num_params / 1e6)

    train_params = config["train_params"]
    num_epochs = train_params["num_epochs"]

    if cached:
        return model, vocab

    for epoch in range(1, 1 + num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for input_seq, target_seq in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()
            _, loss = model(input_seq, target_seq)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                _, loss = model(input_seq, target_seq)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        logging.info(
            "Epoch %i/%i: train_loss=%.4f, val_loss=%.4f",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
        )

    return model, vocab
