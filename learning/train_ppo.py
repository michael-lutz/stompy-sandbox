from re import T
import time
import gym
import wandb
import argparse
from pyvirtualdisplay.display import Display
import torch
import os
import numpy as np
import time

# Local imports
from infrastructure.replay_buffer import ReplayBuffer, Transition
from infrastructure.utils import get_image
from envs.stompy_walk import StompyWalkEnv
from envs.humanoid_walk import HumanoidWalkEnv
from envs.stompy_standup import StompyStandupEnv
from agents.ppo_copy import PPOAgent, RolloutBuffer
from agents.basic_agent import ContinuousPolicyNetwork, ValueNetwork

def initialize_env(env_name):
    if env_name == 'StompyWalk':
        env = StompyWalkEnv(render_mode='rgb_array')
    elif env_name == 'StompyStandup':
        env = StompyStandupEnv(render_mode='rgb_array')
    elif env_name == 'HumanoidWalk':
        env = HumanoidWalkEnv(render_mode='rgb_array')
    else:
        env = gym.make(env_name, render_mode='rgb_array')
    return env

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent for humanoid locomotion.")
    parser.add_argument('--env', type=str, default='Humanoid', help='Gym environment ID')
    parser.add_argument('--exp_name', type=str, default='ppo_humanoid', help='Name of the experiment')
    parser.add_argument('--iters', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='Maximum timesteps per episode')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizers')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.95, help='GAE parameter')
    parser.add_argument('--clip_param', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for PPO updates')
    parser.add_argument('--update_interval', type=int, default=4000, help='Number of steps before updating')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, help='Action std decay rate (for continuous action space)')
    parser.add_argument('--min_action_std', type=float, default=0.1, help='Minimum action std (for continuous action space)')
    parser.add_argument('--action_std_decay_interval', type=int, default=2.5e5, help='Action std decay frequency (for continuous action space)')
    parser.add_argument('--record_video', action='store_true', help='Record video of the agent')
    parser.add_argument('--video_interval', type=int, default=50, help='Interval for recording videos')
    parser.add_argument('--wandb_project', type=str, default='ppo_humanoid', help='Weights & Biases project name')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logging')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on')
    return parser.parse_args()

def main():
    args = parse_args()

    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available() and args.device != 'cpu'): 
        device = torch.device(args.device) 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    # Initialize the environment
    
    print(f"Training PPO agent on {args.env} environment")
    log_dir = args.log_dir + '/' + args.exp_name + '-' + time.strftime("%d-%m-%Y_%H-%M-%S") + "/"
    # make directory for logging if doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Initialize networks and agent
    env = initialize_env(args.env)
    ac_dim = env.action_space.shape[0]
    ob_dim = env.observation_space.shape[0]
    print(f"Action space dimension: {ac_dim}, Observation space dimension: {ob_dim}")

    ppo_agent = PPOAgent(state_dim=ob_dim, action_dim=ac_dim, lr_actor=args.learning_rate, lr_critic=args.learning_rate,
              gamma=args.gamma, K_epochs=args.ppo_epochs, eps_clip=args.clip_param, 
              device=device, action_std_init=0.6)

    wandb.init(project=args.wandb_project)

    # Training loop
    time_step = 0

    for iteration in range(args.iters):
        render_trajectories = False
        if args.record_video and iteration % args.video_interval == 0 and iteration > 0:
            render_trajectories = True
        
        obs, _ = env.reset()
        current_ep_reward = 0

        images = []

        render_times = []
        get_action_times = []
        step_times = []
        update_times = []
        decay_times = []


        for t in range(0, args.max_timesteps):
            start = time.time()
            if render_trajectories:
                images.append(get_image(env))
            end = time.time()
            render_times.append(end-start)

            start = time.time()
            action = ppo_agent.get_action(obs)
            end = time.time()
            get_action_times.append(end-start)
            start = time.time()
            obs, reward, done, _, _ = env.step(action)
            end = time.time()
            step_times.append(end-start)

            ppo_agent.buffer.rewards.append(reward) # TODO: find way to add elements to buffer all at once...
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            start = time.time()
            if time_step % args.update_interval == 0:
                ppo_agent.update(time_step)
            end = time.time()
            update_times.append(end-start)

            start = time.time()
            if time_step % args.action_std_decay_interval == 0:
                ppo_agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)
            end = time.time()
            decay_times.append(end-start)

            if done:
                break
        print(f"Episode {iteration} finished after {t+1} timesteps with reward {current_ep_reward}")
        wandb.log({'reward': current_ep_reward, 'step': iteration})

        print(f"Average render time: {np.mean(render_times)}")
        print(f"Average get action time: {np.mean(get_action_times)}")
        print(f"Average step time: {np.mean(step_times)}")
        print(f"Average update time: {np.mean(update_times)}")
        print(f"Average decay time: {np.mean(decay_times)}")
    

        
        if render_trajectories:
            if 'model' in dir(env):
                fps = 1/env.model.opt.timestep
            else:
                fps = env.metadata['render_fps']
            images = np.array(images)
            images = images[np.newaxis, ...]
        
            # log to wandb
            wandb.log({'training_rollouts': wandb.Video(images, fps=fps, format="mp4"), 'step': iteration})

            # reinitialize environment... hacky fix to avoid black screen bug
            env = initialize_env(args.env)

    env.close()

if __name__ == "__main__":
    display = Display(visible=False, size=(640, 480))
    display.start()
    main()
    display.stop()
