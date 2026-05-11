#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3.5-VL 35B-A3B Pretraining Script

This script provides a direct Python interface to run Qwen3.5-VL 35B-A3B pretraining
without using SLURM. It's useful for local testing or environments without SLURM.

Usage:
    python pretrain_qwen35_vl_35b_a3b.py --help
    
    # Run with default settings (mock data)
    python pretrain_qwen35_vl_35b_a3b.py
    
    # Run with custom settings
    python pretrain_qwen35_vl_35b_a3b.py \
        --seq-length 4096 \
        --train-iters 100 \
        --global-batch-size 32 \
        --micro-batch-size 1 \
        --tp 2 \
        --pp 1 \
        --ep 16

Note:
    This script uses the VLM-specific forward_step function from vlm_step.py,
    which handles both text and visual inputs for multimodal pretraining.
"""

import mindspeed.megatron_adaptor
import argparse
import logging
import os
import sys

import torch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import megatron.core.utils as util
from megatron.bridge.recipes.qwen_vl import qwen35_vl_35b_a3b_pretrain_config
# from megatron.bridge.recipes.kimi
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


util.configure_nvtx_profiling(False)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Qwen3.5-VL pretraining."""
    parser = argparse.ArgumentParser(
        description="Qwen3.5-VL 35B-A3B Pretraining Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # Model and data settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen35_vl_35b_a3b",
        help="Model name (default: qwen35_vl_35b_a3b)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length for training (default: 4096)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="mock",
        choices=["mock"],
        help="Dataset type (default: mock)",
    )
    
    # Parallelism settings
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=8,
        help="Expert parallelism degree (default: 16)",
    )
    parser.add_argument(
        "--cp",
        type=int,
        default=1,
        help="Context parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--sp",
        type=bool,
        default=True,
        help="Sequence parallelism (default: True)",
    )
    
    # Training settings
    parser.add_argument(
        "--train-iters",
        type=int,
        default=300000,
        help="Number of training iterations (default: 300000)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=32,
        help="Global batch size (default: 32)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Maximum learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
        help="Minimum learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=500,
        help="Learning rate warmup iterations (default: 500)",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Checkpoint saving interval (default: 500)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function for Qwen3.5-VL 35B-A3B pretraining."""
    args = parse_args()
    
    logger.info(f"Starting Qwen3.5-VL 35B-A3B pretraining with args: {args}")
    
    # Load default configuration
    # cfg: ConfigContainer = qwen35_vl_35b_a3b_pretrain_config("/data/z00893055/projects/verl/qwen35-megatron-0425/Qwen3.5-35B-A3B-jianceng")
    cfg: ConfigContainer = qwen35_vl_35b_a3b_pretrain_config("/data/weights/Qwen3.5-35B-A3B/")
    
    
    # Override configuration with command-line arguments
    overrides = {
        "model.tensor_model_parallel_size": args.tp,
        "model.pipeline_model_parallel_size": args.pp,
        "model.expert_model_parallel_size": args.ep,
        "model.context_parallel_size": args.cp,
        "model.sequence_parallel": args.sp,
        "train.train_iters": args.train_iters,
        "train.global_batch_size": args.global_batch_size,
        "train.micro_batch_size": args.micro_batch_size,
        "scheduler.max_lr": args.lr,
        "scheduler.min_lr": args.min_lr,
        "scheduler.lr_warmup_iters": args.lr_warmup_iters,
        "logger.log_interval": args.log_interval,
        "checkpoint.save_interval": args.save_interval,
        "checkpoint.save": args.save_dir,
        "checkpoint.load": args.save_dir,
    }
    
    # Process configuration with overrides
    # cfg = process_config_with_overrides(cfg, overrides)
    
    # Run pretraining
    pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()
