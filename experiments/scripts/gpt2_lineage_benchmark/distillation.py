"""Distillation: train students to match teacher outputs with quality metrics."""
import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel
from tqdm import tqdm

from .model import create_model
from .config import ModelConfig, DistillationConfig


def distill_student(
    teacher: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    student_seed: int,
    temperature: float = 2.0,
    alpha_ce: float = 0.5,
    alpha_kl: float = 0.5,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    model_config: Optional[ModelConfig] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a distilled student from scratch."""
    if model_config is None:
        model_config = ModelConfig()

    student = create_model(model_config, seed=student_seed, device=device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    scaler = GradScaler()

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Distill epoch {epoch+1}", disable=not verbose)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_logits = teacher_outputs.logits

                # Student forward
                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                student_logits = student_outputs.logits

                # Hard label loss (cross-entropy)
                ce_loss = student_outputs.loss

                # Soft label loss (KL divergence)
                # Only compute on non-padding positions
                mask = (labels != -100).float().unsqueeze(-1)

                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                )
                kl_loss = (kl_loss.sum(dim=-1) * mask.squeeze(-1)).sum() / mask.sum()
                kl_loss = kl_loss * (temperature ** 2)

                # Combined loss
                loss = alpha_ce * ce_loss + alpha_kl * kl_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": epoch_loss / num_batches})

        # Validation
        val_loss, val_ppl = _evaluate_student(student, val_loader, device)
        history["train_loss"].append(epoch_loss / num_batches)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        if verbose:
            print(f"  Epoch {epoch+1}: val_ppl={val_ppl:.2f}")

    return {
        "model": student,
        "type": "distilled_student",
        "temperature": temperature,
        "alpha_ce": alpha_ce,
        "alpha_kl": alpha_kl,
        "epochs": epochs,
        "seed": student_seed,
        "history": history,
        "final_val_ppl": val_ppl,
    }


def _evaluate_student(
    model: GPT2LMHeadModel,
    val_loader: DataLoader,
    device: str = "cuda",
) -> tuple:
    """Evaluate student perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def compute_distillation_quality(
    teacher: GPT2LMHeadModel,
    student: GPT2LMHeadModel,
    val_loader: DataLoader,
    device: str = "cuda",
    num_batches: int = 100,
) -> Dict[str, float]:
    """Compute distillation quality metrics."""
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()

    total_kl = 0.0
    total_agreement = 0.0
    total_tokens = 0
    teacher_total_loss = 0.0
    student_total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            teacher_logits = teacher_outputs.logits
            student_logits = student_outputs.logits

            # Mask for valid tokens
            mask = (labels != -100)
            num_tokens = mask.sum().item()

            # KL divergence (student || teacher)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kl_per_token = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            kl_sum = (kl_per_token * mask.float()).sum().item()
            total_kl += kl_sum

            # Top-1 agreement
            teacher_preds = teacher_logits.argmax(dim=-1)
            student_preds = student_logits.argmax(dim=-1)
            agreement = ((teacher_preds == student_preds) & mask).sum().item()
            total_agreement += agreement

            # Losses for perplexity
            teacher_total_loss += teacher_outputs.loss.item() * num_tokens
            student_total_loss += student_outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    # Compute metrics
    avg_kl = total_kl / total_tokens
    avg_agreement = total_agreement / total_tokens
    teacher_ppl = math.exp(teacher_total_loss / total_tokens)
    student_ppl = math.exp(student_total_loss / total_tokens)
    ppl_ratio = student_ppl / teacher_ppl

    return {
        "kl_divergence": avg_kl,
        "top1_agreement": avg_agreement,
        "teacher_ppl": teacher_ppl,
        "student_ppl": student_ppl,
        "ppl_ratio": ppl_ratio,
    }


def generate_distilled_students(
    teacher: GPT2LMHeadModel,
    root_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[DistillationConfig] = None,
    model_config: Optional[ModelConfig] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Generate distilled students for a teacher with quality metrics."""
    if config is None:
        config = DistillationConfig()

    students = []
    base_seed = 5000 + root_idx * 100

    for i, temperature in enumerate(config.temperatures):
        if verbose:
            print(f"  [root-{root_idx}] Distilling student T={temperature}")

        result = distill_student(
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            student_seed=base_seed + i,
            temperature=temperature,
            alpha_ce=config.alpha_ce,
            alpha_kl=config.alpha_kl,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            model_config=model_config,
            device=device,
            verbose=verbose,
        )

        # Compute quality metrics
        quality = compute_distillation_quality(
            teacher, result["model"], val_loader, device
        )
        result["quality_metrics"] = quality
        result["id"] = f"root{root_idx}_distilled_{i}"
        result["root_idx"] = root_idx
        result["teacher_root_idx"] = root_idx

        if verbose:
            print(f"    Quality: KL={quality['kl_divergence']:.4f}, "
                  f"agreement={quality['top1_agreement']:.2%}, "
                  f"PPL ratio={quality['ppl_ratio']:.2f}")

        students.append(result)

    return students
