"""Driver script.
"""

import gym
import setup_path
import airgym

from pathlib import Path

from src.core import *
from src.test import *
from src.train import *
from src.functions import *
from src.agent import Daedalous
from src.metrics import MetricLogger


if __name__ == "__main__":
    # Seed random number generators
    seed_generators(42)
    # Initialize output data directory
    export_data_dir = Path("data")
    # Initialize output data filepaths
    model_save_dir, memory_save_dir, log_save_dir, datetime_tag = create_data_dir(
        export_data_dir)
    # Get latest checkpoint data
    model_checkpoint, mem_checkpoint = get_last_chkpt(
        export_data_dir, "2021-06-30T21-06-27", True, True)
    # Build environment
    env = gym.make("airgym:multirotor-v6", quant_areas=36, step_length=1)
    # Create Rainbow agent
    daedalous = Daedalous(env, model_save_dir, memory_save_dir,
                          model_checkpoint, mem_checkpoint)
    # Declare a logger instance to output experiment data
    logger = MetricLogger(log_save_dir)
    # Fit agent
    train(env, daedalous, logger)
    # Test agent
    test(env, daedalous)
    # Plot log data
    experiment_data_plots(export_data_dir, datetime_tag)
