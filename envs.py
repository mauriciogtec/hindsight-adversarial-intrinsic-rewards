import numpy as np
import gym
from gym import spaces


class SchedulerEnv(gym.Env):
    """
    Scheduler MDP. The goal is to learn the best 'budget' times to intervene.
    """

    def __init__(
        self,
        d: int = 2,
        phi: float = 0.9,
        scale: float = 0.1,
        budget: int = 5,
        horizon: int = 100,
        penalty: float = 1.0,
    ):
        super().__init__()

        self.d = d
        self.dims = d + 2  # considering the remaining actions
        self.scale = scale
        self.phi = phi
        self.budget = budget
        self.horizon = horizon
        self.penalty = penalty

        # Action space is either 0 or 1
        self.action_space = spaces.Discrete(2)

        # Observation space bounds for the state
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.dims,), dtype=np.float32
        )

        self.reset()
        self._returns_buffer = []

    def reset(self):
        self.offset = np.random.rand(self.d) * 2 * np.pi
        self._states = [np.sin(self.offset)]
        self._t = 0
        for t in range(self.horizon):
            t_frac = 2 * np.pi * (t + 1) / self.horizon
            next_state = np.sin(self.offset + self.budget * t_frac)
            self._states.append(next_state)
        self._states = np.array(self._states)
        if self.d == 1:
            self._states = self._states.reshape(-1, 1)
        self._cum_reward = 0
        self._done = False
        self._remaining_actions = self.budget
        self._last_action = None
        self._potential_rewards = self._states[1:, 0]
        self._top_k = np.argsort(self._potential_rewards)[-self.budget :]
        self._action_indicator = np.zeros(self.horizon)
        self._action_indicator[self._top_k] = 1
        return self._get_state()

    def _get_state(self):
        frac_remaining = self._remaining_actions / self.budget
        t_frac = self._t / self.horizon
        return np.array([*self._states[self._t], t_frac, frac_remaining])

    def step(self, action):
        if action == 1 and self._remaining_actions == 0:
            self._penalize = True
        else:
            self._penalize = False
        self._remaining_actions = max(0, self._remaining_actions - action)
        self._last_action = action
        self._t += 1
        self._check_done()
        return self._get_state(), self._get_reward(), self._done, {}

    def _check_done(self):
        self._done = self._t >= self.horizon
        if self._done:
            self._returns_buffer.append(self._cum_reward)

    def _get_reward(self):
        reward = 0

        if self._last_action == 1 and not self._penalize:
            reward += self._potential_rewards[self._t - 1]

        if self._penalize:
            reward -= self.penalty

        self._cum_reward += reward
        return reward

    def render(self, mode: str = "human"):
        ...  # Visualization is not implemented in this environment

    def close(self):
        ...  # Cleanup tasks, if any, can be added here


if __name__ == "__main__":
    # this test should work
    import matplotlib.pyplot as plt

    env = SchedulerEnv()
    done = False
    states = []
    rewards = []
    while not done:
        a = env.action_space.sample()  # using gym's sample method for random actions
        s, r, done, _ = env.step(a)
        states.append(s)
        rewards.append(r)
    states = np.array(states)
    rewards = np.array(rewards)

    plt.plot(states)
    plt.show()

    plt.plot(np.cumsum(rewards))
    plt.show()
