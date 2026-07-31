"""Training loop for GPT-2-Small-Lite models."""
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel, get_scheduler
from tqdm import tqdm

from .config import TrainingConfig, ModelConfig
from .model import create_model, save_checkpoint, enable_gradient_checkpointing


def train_root(
    root_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    checkpoint_dir: Optional[Path] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a root model from scratch with checkpoint saving."""
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainingConfig()

    seed = 1000 + root_idx * 100
    model = create_model(model_config, seed=seed, device=device)

    if train_config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    num_training_steps = len(train_loader) * train_config.epochs
    num_warmup_steps = train_config.warmup_steps

    scheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler() if train_config.fp16 else None

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir) / f"root_{root_idx}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "epochs": []}
    t0 = time.time()

    # Save epoch 0 checkpoint (post-init, pre-training)
    if checkpoint_dir and 0 in train_config.save_epochs:
        val_loss, val_ppl = evaluate(model, val_loader, device, train_config.fp16)
        save_checkpoint(
            model,
            checkpoint_dir / "epoch_0.pt",
            epoch=0,
            config={"model": vars(model_config), "training": vars(train_config)},
            metrics={"val_loss": val_loss, "val_ppl": val_ppl},
        )
        if verbose:
            print(f"[root-{root_idx}] Saved epoch 0 checkpoint (val_ppl={val_ppl:.2f})")

    global_step = 0
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not verbose)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if train_config.fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / train_config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / train_config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": epoch_loss / num_batches})

        avg_train_loss = epoch_loss / num_batches
        val_loss, val_ppl = evaluate(model, val_loader, device, train_config.fp16)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epochs"].append(epoch)

        if verbose:
            elapsed = time.time() - t0
            print(
                f"[root-{root_idx}] Epoch {epoch}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.2f}, elapsed={elapsed:.1f}s"
            )

        if checkpoint_dir and epoch in train_config.save_epochs:
            save_checkpoint(
                model,
                checkpoint_dir / f"epoch_{epoch}.pt",
                epoch=epoch,
                config={"model": vars(model_config), "training": vars(train_config)},
                metrics={"train_loss": avg_train_loss, "val_loss": val_loss, "val_ppl": val_ppl},
            )

    return {
        "model": model,
        "history": history,
        "final_val_ppl": val_ppl,
        "training_time_seconds": time.time() - t0,
        "seed": seed,
    }


def evaluate(
    model: GPT2LMHeadModel,
    val_loader: DataLoader,
    device: str = "cuda",
    fp16: bool = True,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def continue_training(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Continue training a model (for descendants)."""
    model = model.to(device)
    model.train()

    if gradient_checkpointing:
        enable_gradient_checkpointing(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if fp16 else None

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            epoch_loss += loss.item()
            num_batches += 1

        val_loss, val_ppl = evaluate(model, val_loader, device, fp16)
        history["train_loss"].append(epoch_loss / num_batches)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        if verbose:
            print(f"  Epoch {epoch+1}: val_ppl={val_ppl:.2f}")

    return {"model": model, "history": history, "final_val_ppl": val_ppl}
