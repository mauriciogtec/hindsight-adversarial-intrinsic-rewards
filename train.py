import logging
from functools import partial

import hydra
from omegaconf import DictConfig

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs import SchedulerEnv
from gymnasium import Env


def make_env(rank: int, seed: int, **kwargs) -> Env:
    """Auxiliry function to make parallel vectorized envs"""
    set_random_seed(seed + rank)

    def _init():
        return SchedulerEnv(**kwargs)

    return _init


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    kwargs = {}
    val_kwargs = {"penalty": cfg.eval.penalty}
    env_promise = [make_env(i, cfg.seed, **kwargs) for i in range(cfg.num_envs)]
    env_promise_val = [make_env(i, cfg.seed, **val_kwargs) for i in range(cfg.num_envs)]

    # vectorize environments
    if cfg.parallel:
        fun = partial(SubprocVecEnv, start_method="spawn")
    else:
        fun = DummyVecEnv

    env = fun(env_promise)
    env_val = fun(env_promise_val)

    # RL training code here
    logging.info("Creating RL model")

    rl_model = hydra.utils.instantiate(
        cfg.algo, env=env, verbose=0, tensorboard_log="./logs/rl_tensorboard/"
    )

    # Create a callback to evaluate the agent
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=f"./logs/train/{cfg.algo_name}/{cfg.seed}/best_model",
        log_path=f"./logs/train/{cfg.algo_name}/{cfg.seed}/",
        eval_freq=cfg.eval.freq,  # Evaluation frequency
        n_eval_episodes=cfg.eval.episodes,
    )

    # Training the agent
    logging.info("Training RL model")
    rl_model.learn(
        total_timesteps=cfg.training_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
