from re import L
import numpy as np
import cv2
from typing import Union, List
import torch

from infrastructure.replay_buffer import Transition, RenderedTransition
from agents.basic_agent import RLAgent

def get_image(env):
    img = env.render()
    img = np.moveaxis(img, 2, 0)
    return img

def compute_log_probs(actions, action_means, variances):
    """
    Compute the log probabilities of the actions under a Gaussian distribution with means `action_means` and standard deviations `action_std_devs`.

    :param actions: The actions to compute the log probabilities of.
    :param action_means: The means of the action distribution.
    """
    assert actions.shape == action_means.shape, f"Actions shape {actions.shape} does not match action means shape {action_means.shape}"
    if not isinstance(variances, torch.Tensor):
        variances = torch.tensor(variances)
    log_probs = -0.5 * (((actions - action_means) ** 2 / variances) + torch.log(2 * torch.pi * variances))
    log_probs = torch.sum(log_probs, dim=-1)  # Sum across action dimensions if actions are vectors
    return log_probs

def fast_sample_action(action_mean: torch.Tensor, action_var: torch.Tensor, device='cpu'):
    """
    Sample an action from a normal distribution with mean `action_mean` and standard deviation `action_std_dev`.
    NOTE: if using this instead of the Multivariate Gaussian, MUST assume diagonal covariance matrix (and => independence...)
    :param action_mean: The mean[s] of the action distribution.
    :param action_std_dev: The standard deviation[s] of the action distribution.
    """
    assert action_mean.shape == action_var.shape, f"Action mean shape {action_mean.shape} does not match action std dev shape {action_var.shape}"
    if not isinstance(action_var, torch.Tensor):
        action_var = torch.tensor(action_var, device=device)
    action_std_dev = torch.sqrt(action_var)

    sampled_actions = action_mean + torch.randn_like(action_mean) * action_std_dev
    return sampled_actions

def sample_trajectory(env, policy: RLAgent, max_path_length: int, render=False) -> Union[Transition, RenderedTransition]:
    """Sample a rollout in the environment from a policy.
    
    :param env: The environment to sample from.
    :param policy: The policy to use for sampling.
    :param max_path_length: The maximum length of the rollout.
    :param render: Whether to render the environment.
    """
    ob, _ = env.reset()

    obs, acs, log_probs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render()[0]
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        ac, log_prob,  = policy.get_action(ob)
        ac = ac.numpy()
        log_prob = log_prob.numpy()
        next_ob, rew, done, _, _ = env.step(ac)
        
        steps += 1
        rollout_done = done or steps >= max_path_length
        obs.append(ob)
        acs.append(ac)
        log_probs.append(log_prob)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob

        if rollout_done:
            break

    trajectory_data = {
        "observations": np.array(obs),
        "actions": np.array(acs),
        "log_probs": np.array(log_probs),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_obs),
        "dones": np.array(terminals)
    }
    if render:
        trajectory_data["image_obs"] = np.array(image_obs)

    return trajectory_data

def sample_n_trajectories(env, policy, n, max_path_length, render=False) -> Union[Transition, RenderedTransition]:
    """Collect n trajectories of the policy
    
    :param env: The environment to sample from.
    :param policy: The policy to use for sampling.
    :param n: The number of trajectories to sample.
    :param max_path_length: The maximum length of the rollout.
    :param render: Whether to render the environment.
    """
    paths = [sample_trajectory(env, policy, max_path_length, render) for _ in range(n)]
    
    # Extract all data from paths in a more efficient manner
    paths = [sample_trajectory(env, policy, max_path_length, render) for _ in range(n)]

    combined = {key: np.concatenate([p[key] for p in paths]) for key in paths[0]}
    if render:
        combined["image_obs"] = np.concatenate([p["image_obs"] for p in paths])

    return combined
