"""
Shared code for the hidden state analysis of the RL² agents

Usage in a notebook inside experiments/analysis/<method>/<system>/:

    sys.path.insert(0, os.path.abspath("../.."))
    from analysis_utils import SYSTEMS, load_model, collect_avg_hidden_states

    system = SYSTEMS["cartpole"]
    model = load_model(system)
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

# Project folders, independent of the folder the notebook runs in
ANALYSIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ANALYSIS_DIR.parent
ROOT_DIR = EXPERIMENTS_DIR.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# The system files live in experiments/, the agent and utils in the project root
for path in (ROOT_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cartpole_wind
import halfcheetah_action
import pendulum_wind


@dataclass(frozen=True)
class System:
    model_factory: Callable
    make_envs: Callable
    sample_tasks_eval: Callable  # evenly spaced tasks
    sample_tasks_train: Callable  # random tasks as in training
    checkpoint: str  # file name in checkpoints/
    parameter_name: str  # task parameter: the environment variable that defines a task
    is_discrete: bool
    action_dim: int
    action_low: Optional[float] = None
    action_high: Optional[float] = None


SYSTEMS = {
    "cartpole": System(
        model_factory=cartpole_wind.cartpole_model_factory,
        make_envs=cartpole_wind.make_envs,
        sample_tasks_eval=cartpole_wind.sample_tasks_eval,
        sample_tasks_train=cartpole_wind.sample_tasks_train,
        checkpoint="cartpole_wind.pt",
        parameter_name="wind_mag",
        is_discrete=True,
        action_dim=2,
    ),
    "pendulum": System(
        model_factory=pendulum_wind.pendulum_model_factory,
        make_envs=pendulum_wind.make_envs,
        sample_tasks_eval=pendulum_wind.sample_tasks_eval,
        sample_tasks_train=pendulum_wind.sample_tasks_train,
        checkpoint="pendulum_wind.pt",
        parameter_name="wind_torque",
        is_discrete=False,
        action_dim=1,
        action_low=-2.0,
        action_high=2.0,
    ),
    "halfcheetah": System(
        model_factory=halfcheetah_action.halfcheetah_model_factory,
        make_envs=halfcheetah_action.make_envs,
        sample_tasks_eval=halfcheetah_action.sample_tasks_eval,
        sample_tasks_train=halfcheetah_action.sample_tasks_train,
        checkpoint="halfcheetah.pt",
        parameter_name="a_m",
        is_discrete=False,
        action_dim=6,
        action_low=-1.0,
        action_high=1.0,
    ),
}


def set_seeds(seed):
    """CartPole samples its random tasks with random, Pendulum and HalfCheetah with np.random"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def task_parameters(system, task_configs):
    """Task parameter of every task config (e.g. the wind magnitude) as an array"""
    return np.array([cfg[system.parameter_name] for cfg in task_configs])


def load_model(system):
    """Create the RL2 model of a system and load its trained weights"""
    model = system.model_factory()
    checkpoint = torch.load(
        CHECKPOINT_DIR / system.checkpoint, map_location=torch.device("cpu")
    )

    # Support both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Evaluation mode: the state and reward normalizers are not updated during the rollouts
    model.eval()
    return model


def select_action(system, policy_out):
    """
    Discrete: the action with the highest logit
    Continuous: the mean of the policy, squashed into the action range like in training
    """
    if system.is_discrete:
        return torch.argmax(policy_out, dim=-1).squeeze(0)

    mu, _ = policy_out
    mu = mu.squeeze(0)

    if system.action_low is None or system.action_high is None:
        return mu

    scale = (system.action_high - system.action_low) / 2.0
    bias = (system.action_high + system.action_low) / 2.0
    return scale * torch.tanh(mu) + bias


def collect_avg_hidden_states(model, system, envs, seed, steps=3000):
    """Play every task of envs for steps and average the GRU hidden state per task.

    The hidden state starts at zero and is not reset between episodes, because the task
    (e.g. the wind) stays the same. The last action and reward are passed on after an
    episode ends and only the done flag marks the episode boundary. The seed fixes the
    initial states of the environments so the rollout is reproducible.

    Returns an array of shape (number of tasks, h_dim).
    """
    num_envs = envs.num_envs
    h_dim = model.feat_extractor.h_dim

    obs, _ = envs.reset(seed=seed)

    hidden_state = torch.zeros(1, num_envs, h_dim)
    if system.is_discrete:
        prev_action = torch.zeros(num_envs, dtype=torch.long)
    else:
        prev_action = torch.zeros(num_envs, system.action_dim)
    prev_reward = torch.zeros(num_envs, 1)
    prev_done = torch.ones(num_envs, 1)

    # Sum of the hidden states over all steps one row per task
    hidden_sum = np.zeros((num_envs, h_dim), dtype=np.float64)

    with torch.no_grad():
        for _ in range(steps):
            policy_out, _, hidden_state = model(
                torch.tensor(obs).view(1, num_envs, -1).float(),
                prev_action.unsqueeze(0),
                prev_reward.view(1, num_envs, 1),
                prev_done.view(1, num_envs, 1),
                hidden_state,
            )

            hidden_sum += hidden_state.squeeze(0).cpu().numpy()

            action = select_action(system, policy_out)
            obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())

            # Update the RL2 context (no reset of action and reward after an episode ends)
            done = np.logical_or(terminated, truncated)
            prev_action = action
            prev_reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
            prev_done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

    # The model computes in float32, so the average is stored in float32 as well
    return (hidden_sum / steps).astype(np.float32)
