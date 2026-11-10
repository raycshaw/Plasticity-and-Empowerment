"""Pinned checkpoints and the screened dataset for the sampled RLVR experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/gsm8k50/gsm8k50_injections.parquet'
CHECKPOINTS = [
    (0, 'Raghav-Singhal/1pp-1.7b-raw-sft', '0282a9acbab0c0f4147874b04be180be78b61fbd'),
    (50, 'VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k', '8e47e5742ae93c215cd92a6798ccdac4d385596d'),
    (100, 'VityaVitalich/1pp-1.7b-raw-sft-grpo-gsm8k', 'e7da9caa9e8e724cf326595a98fa4cf3ce8eb807'),
]
