"""Pydantic models validating the merged Hydra config before execution starts.

Responsibility: every scripts/*.py entrypoint calls this module first.
Invalid configs must fail in milliseconds, not 20 minutes into a GPU run
— see architecture spec Section 7.

Depends on: configs/* (validates the merged OmegaConf dict at runtime)
Consumed by: every scripts/*.py entrypoint
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


# daily_dialog and eli5 are intentionally excluded — both are broken upstream
# on HF Hub as of 2026-07-04 (daily_dialog: legacy-script-only, no longer
# loadable; eli5: removed from the Hub entirely). See configs/data/mixture.yaml
# and docs/dataset_guide.md. Their adapters/registry entries are left in place
# (harmless, unused) in case a working replacement path appears later.
_VALID_DATASET_NAMES: frozenset[str] = frozenset({
    "wikitext", "openwebtext", "bookcorpusopen", "tinystories",
    "ag_news", "cnn_dailymail", "xsum", "fineweb_edu",
})

_VALID_TARGET_MODULES: frozenset[str] = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
})


class RunConfig(BaseModel):
    name: Optional[str] = None
    seed: int
    output_dir: str


class TokenizerConfig(BaseModel):
    name: str
    add_eos_token: bool = True

    @field_validator("name")
    @classmethod
    def name_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("tokenizer.name must be non-empty")
        return v


class ModelConfig(BaseModel):
    name: str
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "bf16", "fp16", "fp32"]
    device_map: str = "auto"
    tokenizer: TokenizerConfig

    @field_validator("name")
    @classmethod
    def name_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("model.name must be non-empty")
        return v


class DataSourceConfig(BaseModel):
    name: str
    config: Optional[str] = None
    weight: Optional[float] = None
    sample_count: int | Literal["all"]
    streaming: bool = False

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_DATASET_NAMES:
            raise ValueError(
                f"{v!r} is not an allowed dataset name. "
                f"Allowed: {sorted(_VALID_DATASET_NAMES)}"
            )
        return v

    @field_validator("sample_count", mode="before")
    @classmethod
    def sample_count_positive(cls, v: Any) -> Any:
        if isinstance(v, int) and v <= 0:
            raise ValueError("sample_count must be > 0 when an integer")
        return v


class MixtureConfig(BaseModel):
    sampling_strategy: Literal["proportional_interleave", "epoch_balanced"]
    packing_granularity: Literal["within_dataset", "across_mixture"]
    sequence_length: int
    sources: list[DataSourceConfig]

    @field_validator("sequence_length")
    @classmethod
    def seq_len_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sequence_length must be > 0")
        return v

    @field_validator("sources")
    @classmethod
    def sources_valid(cls, v: list[DataSourceConfig]) -> list[DataSourceConfig]:
        if len(v) != 8:
            raise ValueError(f"sources must contain exactly 8 entries, got {len(v)}")
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate dataset names in mixture.sources")
        return v

    @model_validator(mode="after")
    def weights_consistent(self) -> "MixtureConfig":
        """Either all weights are None (derive from sample_count) or all are explicit floats."""
        weights = [s.weight for s in self.sources]
        all_none = all(w is None for w in weights)
        all_set = all(w is not None for w in weights)
        if not (all_none or all_set):
            raise ValueError(
                "All sources must have weight=None (proportional from sample_count) "
                "or all must carry explicit float weights — mixed state is not allowed."
            )
        return self


class LoRAConfig(BaseModel):
    r: Literal[8, 16, 32, 64]
    alpha: int
    dropout: float
    bias: Literal["none", "all", "lora_only"]
    target_modules: list[str]
    task_type: Literal["CAUSAL_LM"]

    @field_validator("alpha")
    @classmethod
    def alpha_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("alpha must be > 0")
        return v

    @field_validator("dropout")
    @classmethod
    def dropout_range(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("dropout must satisfy 0 <= x < 1")
        return v

    @field_validator("target_modules")
    @classmethod
    def target_modules_valid(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_modules must be non-empty")
        invalid = set(v) - _VALID_TARGET_MODULES
        if invalid:
            raise ValueError(
                f"Invalid target_modules: {sorted(invalid)}. "
                f"Allowed: {sorted(_VALID_TARGET_MODULES)}"
            )
        return v


class TrainingConfig(BaseModel):
    num_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    precision: Literal["bf16", "fp16", "4bit", "8bit"]
    gradient_checkpointing: bool
    dataloader_num_workers: int

    @field_validator(
        "num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
        "logging_steps", "eval_steps", "save_steps",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("max_grad_norm")
    @classmethod
    def max_grad_norm_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_grad_norm must be > 0")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def warmup_ratio_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("warmup_ratio must satisfy 0 <= x <= 1")
        return v

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight_decay must be >= 0")
        return v

    @field_validator("dataloader_num_workers")
    @classmethod
    def workers_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dataloader_num_workers must be >= 0")
        return v


class OptimizerConfig(BaseModel):
    name: Literal["paged_adamw_8bit", "adamw"]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    @field_validator("lr")
    @classmethod
    def lr_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("lr must satisfy 0 < x < 1.0 (caught likely typo, e.g. lr: 5.0)")
        return v

    @field_validator("betas")
    @classmethod
    def betas_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        for beta in v:
            if not (0.0 < beta < 1.0):
                raise ValueError(f"each beta must be in (0, 1), got {beta}")
        return v

    @field_validator("eps")
    @classmethod
    def eps_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("eps must be > 0")
        return v

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight_decay must be >= 0")
        return v


class SchedulerConfig(BaseModel):
    name: Literal["cosine", "linear", "constant"]
    num_warmup_steps: Optional[int] = None   # resolved at runtime
    num_training_steps: Optional[int] = None  # resolved at runtime


class EvaluationConfig(BaseModel):
    val_split_fraction: float
    length_buckets: list[int]
    num_generation_samples: int
    generation_max_new_tokens: int

    @field_validator("val_split_fraction")
    @classmethod
    def val_split_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("val_split_fraction must satisfy 0 < x < 1")
        return v

    @field_validator("length_buckets")
    @classmethod
    def length_buckets_valid(cls, v: list[int]) -> list[int]:
        if any(b <= 0 for b in v):
            raise ValueError("all length_buckets must be > 0")
        if v != sorted(set(v)) or len(v) != len(set(v)):
            raise ValueError("length_buckets must be strictly increasing with no duplicates")
        return v

    @field_validator("num_generation_samples", "generation_max_new_tokens")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class BenchmarkingConfig(BaseModel):
    batch_sizes_to_test: list[int]
    precision_modes_to_test: list[Literal["bf16", "fp16", "4bit", "8bit"]]
    sequence_lengths_to_test: list[int]
    lora_ranks_to_test: list[Literal[8, 16, 32, 64]]
    num_warmup_iterations: int
    num_measured_iterations: int

    @field_validator("batch_sizes_to_test", "sequence_lengths_to_test")
    @classmethod
    def nonempty_positive_ints(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("must be non-empty")
        if any(x <= 0 for x in v):
            raise ValueError("all values must be > 0")
        return v

    @field_validator("precision_modes_to_test", "lora_ranks_to_test")
    @classmethod
    def nonempty_list(cls, v: list) -> list:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("num_warmup_iterations")
    @classmethod
    def warmup_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_warmup_iterations must be >= 0")
        return v

    @field_validator("num_measured_iterations")
    @classmethod
    def measured_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_measured_iterations must be > 0")
        return v


class LoggingConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    tags: list[str] = []
    log_generation_table: bool
    log_checkpoints_as_artifacts: bool

    @field_validator("project")
    @classmethod
    def project_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("logging.project must be non-empty")
        return v


class InferenceConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool

    @field_validator("max_new_tokens")
    @classmethod
    def max_new_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_new_tokens must be > 0")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if not (0.0 < v <= 2.0):
            raise ValueError("temperature must satisfy 0 < x <= 2")
        return v

    @field_validator("top_p")
    @classmethod
    def top_p_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("top_p must satisfy 0 < x <= 1")
        return v


class RootConfig(BaseModel):
    run: RunConfig
    model: ModelConfig
    data: MixtureConfig
    lora: LoRAConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    evaluation: EvaluationConfig
    benchmarking: BenchmarkingConfig
    logging: LoggingConfig
    inference: InferenceConfig


def validate_config(merged_config: dict[str, Any]) -> RootConfig:
    """Validate the fully-merged Hydra config against all Pydantic schemas.

    Pass the result of OmegaConf.to_container(cfg, resolve=True) here.

    Raises:
        pydantic.ValidationError: on any constraint violation, with field path and message.
    """
    return RootConfig.model_validate(merged_config)
