from typing import TypedDict, Union
import numpy as np

class RenderedTransition(TypedDict):
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    image_obs: np.ndarray

class Transition(TypedDict):
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = np.empty((capacity,), dtype=np.ndarray)  # Assume dtype and shapes are predefined
        self.actions = np.empty((capacity,), dtype=np.ndarray)
        self.log_probs = np.empty((capacity,), dtype=np.ndarray)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty((capacity,), dtype=np.ndarray)
        self.dones = np.empty((capacity,), dtype=bool)

    def __len__(self) -> int:
        """
        Return the number of transitions currently stored in the replay buffer.
        """
        return self.size

    def sample(self, batch_size: int) -> Transition:
        """
        Sample a batch of transitions from the replay buffer.

        :param batch_size: The number of transitions to sample.

        Use like:
            batch = replay_buffer.sample(batch_size)
        """
        assert self.size >= batch_size
        assert self.observations is not None and self.actions is not None and self.rewards is not None and self.next_observations is not None and self.dones is not None and self.log_probs is not None
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "log_probs": self.log_probs[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices]
        }
    
    def get_all(self) -> Transition:
        """
        Return all transitions stored in the replay buffer.

        Use like:
            transitions = replay_buffer.get_all()
        """
        assert self.observations is not None and self.actions is not None and self.rewards is not None and self.next_observations is not None and self.dones is not None and self.log_probs is not None
        return {
            "observations": self.observations,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones
        }

    def empty(self) -> bool:
        """
        Return whether the replay buffer is empty.
        """
        return self.size == 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: Union[np.ndarray, float, int],
        next_observation: np.ndarray,
        done: Union[np.ndarray, bool],
    ):
        """
        Insert a single transition into the replay buffer.

        :param observation: The observation at the current time step.
        :param action: The action taken at the current time step.
        :param reward: The reward received at the current time step.
        :param next_observation: The observation at the next time step.
        :param done: Whether the episode has terminated at the current time step.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        assert self.observations is not None and self.actions is not None and self.rewards is not None and self.next_observations is not None and self.dones is not None
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.log_probs = np.empty((self.max_size, *log_prob.shape), dtype=log_prob.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == self.rewards.shape[1:]
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == self.dones.shape[1:]

        idx = self.size % self.max_size
        self.observations[idx] = observation
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.next_observations[idx] = next_observation
        self.dones[idx] = done

        self.size += 1