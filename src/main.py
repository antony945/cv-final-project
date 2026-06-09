"""Entry point — Hydra-managed LeJEPA pre-training.

Usage:
    uv run python -m src.main                              # train with base config
    uv run python -m src.main +experiment=pretrain_test    # quick sanity test
    uv run python -m src.main +experiment=pretrain_local   # meaningful local run
    uv run python -m src.main epochs=100 bs=32             # CLI overrides

Standalone scripts (no Hydra):
    uv run python -m src.verify                   # check dataset + forward pass
    uv run python -m src.visualize --checkpoint checkpoints/lejepa_xxs_final.pth
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

# Pin HF cache to absolute path before hydra changes cwd
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    task = cfg.get("task", "pretrain")
    if task == "pretrain":
        from src.train import pretrain_lejepa
        pretrain_lejepa(cfg)
    elif task == "finetune":
        from src.train import finetune_depth
        finetune_depth(cfg)
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'pretrain' or 'finetune'.")


if __name__ == "__main__":
    main()
