"""Configuration for GPT-2-Small-Lite model and training."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """GPT-2-Small-Lite architecture (~35M params)."""
    num_layers: int = 6
    d_model: int = 384
    d_ff: int = 1536  # 4x expansion
    num_heads: int = 6
    head_dim: int = 64
    vocab_size: int = 50257  # GPT-2 tokenizer
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def to_hf_config(self):
        """Convert to HuggingFace GPT2Config."""
        from transformers import GPT2Config
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.max_seq_len,
            n_embd=self.d_model,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            n_inner=self.d_ff,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            layer_norm_epsilon=self.layer_norm_eps,
            bos_token_id=50256,
            eos_token_id=50256,
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters for root models."""
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_epochs: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5])
    eval_steps: int = 500
    logging_steps: int = 100


@dataclass
class DescendantConfig:
    """Configuration for descendant generation."""
    # Continued pretraining (same corpus)
    cont_pt_same_epochs: List[int] = field(default_factory=lambda: [1, 2, 3])
    cont_pt_same_lr: float = 1e-4

    # Continued pretraining (domain shift)
    cont_pt_shift_epochs: List[int] = field(default_factory=lambda: [1, 2])
    cont_pt_shift_lr: float = 1e-4

    # Supervised fine-tuning
    sft_epochs: List[int] = field(default_factory=lambda: [2, 3])
    sft_lr: float = 5e-5

    # LoRA
    lora_ranks: List[int] = field(default_factory=lambda: [4, 8, 16])
    lora_alpha_multiplier: float = 2.0
    lora_epochs: int = 2
    lora_lr: float = 1e-4

    # Pruning
    prune_sparsities: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])

    # Quantization
    quant_levels: List[int] = field(default_factory=lambda: [256, 64])  # INT8, INT6


@dataclass
class DistillationConfig:
    """Configuration for distilled student generation."""
    temperatures: List[float] = field(default_factory=lambda: [1.0, 2.0])
    epochs: int = 5
    learning_rate: float = 3e-4
    alpha_ce: float = 0.5  # weight for hard targets
    alpha_kl: float = 0.5  # weight for soft targets


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    descendant: DescendantConfig = field(default_factory=DescendantConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    # Root allocation
    n_calibration_roots: int = 3
    n_development_roots: int = 3
    n_test_roots: int = 4

    # Data
    dataset: str = "tinystories"  # or "wikitext"
    domain_shift_dataset: str = "wikitext"
    max_train_samples: Optional[int] = None  # None = use all

    # Output
    output_dir: str = "results/lineage_benchmark_gpt2"
    checkpoint_dir: str = "results/lineage_benchmark_gpt2/checkpoints"

    @property
    def n_roots(self) -> int:
        return self.n_calibration_roots + self.n_development_roots + self.n_test_roots

    def get_split(self, root_idx: int) -> str:
        """Return which split a root belongs to."""
        if root_idx < self.n_calibration_roots:
            return "calibration"
        elif root_idx < self.n_calibration_roots + self.n_development_roots:
            return "development"
        else:
            return "test"

    @classmethod
    def from_preset(cls, preset: str) -> "BenchmarkConfig":
        """Create config from preset name.

        Presets:
        - smoke: 2 roots, 500 samples, 1 epoch (~2 min)
        - paper: 8 roots, 50K samples, 3 epochs (~6-8 hr) [Option A]
        - full: 10 roots, 100K samples, 5 epochs (~15-20 hr)
        """
        config = cls()

        if preset == "smoke":
            config.n_calibration_roots = 1
            config.n_development_roots = 1
            config.n_test_roots = 0
            config.max_train_samples = 500
            config.model.num_layers = 4
            config.model.d_model = 256
            config.model.d_ff = 1024
            config.model.num_heads = 4
            config.training.epochs = 1
            config.training.batch_size = 8
            config.descendant.cont_pt_same_epochs = [1]
            config.descendant.cont_pt_shift_epochs = []
            config.descendant.sft_epochs = []
            config.descendant.lora_ranks = [4]
            config.descendant.prune_sparsities = [0.5]
            config.descendant.quant_levels = [256]
            config.distillation.temperatures = [2.0]
            config.distillation.epochs = 1

        elif preset == "paper":
            # 8 roots, 50K samples, 3 epochs
            # Phase 1: ~3 hr, Phase 2: ~2 hr, Phase 3: ~15 min
            config.n_calibration_roots = 2
            config.n_development_roots = 3
            config.n_test_roots = 3
            config.max_train_samples = 50000
            config.training.epochs = 3
            config.training.batch_size = 8
            # Minimal descendants - just what paper needs
            config.descendant.cont_pt_same_epochs = [1]  # 1 fine-tune variant
            config.descendant.cont_pt_shift_epochs = []  # skip domain shift
            config.descendant.sft_epochs = []  # skip SFT (covered by cont_pt)
            config.descendant.lora_ranks = [8]  # 1 LoRA
            config.descendant.lora_epochs = 1
            config.descendant.prune_sparsities = [0.3, 0.5, 0.7]  # 3 for curve
            config.descendant.quant_levels = [256, 64]  # INT8 + INT6
            config.distillation.temperatures = [2.0]  # 1 student
            config.distillation.epochs = 2

        elif preset == "full":
            # Full benchmark: 10 roots, 100K samples, 5 epochs
            config.n_calibration_roots = 3
            config.n_development_roots = 3
            config.n_test_roots = 4
            config.max_train_samples = 100000
            config.training.epochs = 5

        else:
            raise ValueError(f"Unknown preset: {preset}. Use: smoke, paper, full")

        return config
