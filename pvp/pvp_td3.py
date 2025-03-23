import copy
import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional

import numpy as np
import torch as th
import torch
from torch.nn import functional as F

from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from pvp.sb3.td3.td3 import TD3

logger = logging.getLogger(__name__)


class PVPTD3(TD3):
    def __init__(self, use_balance_sample=True, q_value_bound=1., *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs["cql_coefficient"]
            kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        self.extra_config = {}
        for k in ["no_done_for_positive", "no_done_for_negative", "reward_0_for_positive", "reward_0_for_negative",
                  "reward_n2_for_intervention", "reward_1_for_all", "use_weighted_reward", "remove_negative",
                  "adaptive_batch_size", "add_bc_loss", "only_bc_loss", "with_human_proxy_value_loss",
                  "with_agent_proxy_value_loss", "simple_batch"]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in ["agent_data_ratio", "bc_loss_weight"]:
            if k in kwargs:
                self.extra_config[k] = kwargs.pop(k)

        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample
        super(PVPTD3, self).__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super(PVPTD3, self)._setup_model()
        if self.use_balance_sample:
            self.human_data_buffer = HACOReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs
            )
        else:
            self.human_data_buffer = self.replay_buffer

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        with_human_proxy_value_loss = self.extra_config["with_human_proxy_value_loss"]
        with_agent_proxy_value_loss = self.extra_config["with_agent_proxy_value_loss"]

        stat_recorder = defaultdict(list)

        should_concat = False
        if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
            replay_data_human = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env, return_all=True
            )
            human_data_size = len(replay_data_human.observations)
            human_data_size = max(1, self.extra_config["agent_data_ratio"] * human_data_size)
            human_data_size = int(human_data_size)
            should_concat = True

        elif self.human_data_buffer.pos > 0:
            replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env, return_all=True)
        elif self.replay_buffer.pos > 0:
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        else:
            gradient_steps = 0

        for step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            if self.extra_config["simple_batch"]:
                if self.replay_buffer.pos == 0:
                    replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                elif self.human_data_buffer.pos == 0:
                    replay_data = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                else:
                    replay_data_agent = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                    replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            elif self.extra_config["adaptive_batch_size"]:
                if should_concat:
                    replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            else:

                if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                    replay_data_agent = self.replay_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    replay_data_human = self.human_data_buffer.sample(int(batch_size / 2), env=self._vec_normalize_env)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
                elif self.human_data_buffer.pos > batch_size:
                    replay_data = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
                elif self.replay_buffer.pos > batch_size:
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                else:
                    break

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions_behavior.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_behavior_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            current_q_novice_values = self.critic(replay_data.observations, replay_data.actions_novice)

            stat_recorder["q_value_behavior"].append(current_q_behavior_values[0].mean().item())
            stat_recorder["q_value_novice"].append(current_q_novice_values[0].mean().item())

            # Compute critic loss
            critic_loss = []
            for (current_q_behavior, current_q_novice) in zip(current_q_behavior_values, current_q_novice_values):
                l = F.mse_loss(current_q_behavior, target_q_values)

                if with_human_proxy_value_loss:
                    l += th.mean(
                        replay_data.interventions * self.cql_coefficient * F.mse_loss(
                            current_q_behavior, self.q_value_bound * th.ones_like(current_q_behavior), reduction="none"
                        )
                    )

                if with_agent_proxy_value_loss:
                    l += th.mean(
                        replay_data.interventions * self.cql_coefficient * F.mse_loss(
                            current_q_novice, -self.q_value_bound * th.ones_like(current_q_behavior), reduction="none"
                        )
                    )

                critic_loss.append(l)
            critic_loss = sum(critic_loss)

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["train/critic_loss"] = critic_loss.item()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                new_action = self.actor(replay_data.observations)

                # BC loss on human data
                bc_loss = F.mse_loss(replay_data.actions_behavior, new_action, reduction="none").mean(axis=-1)
                masked_bc_loss = (replay_data.interventions.flatten() *
                                  bc_loss).sum() / (replay_data.interventions.flatten().sum() + 1e-5)

                if self.extra_config["only_bc_loss"]:
                    actor_loss = masked_bc_loss
                    # Critics will be completely useless.

                else:
                    actor_loss = -self.critic.q1_forward(replay_data.observations, new_action).mean()
                    if self.extra_config["add_bc_loss"]:
                        actor_loss += masked_bc_loss * self.extra_config["bc_loss_weight"]
                        # actor_loss += bc_loss.mean() * self.extra_config["bc_loss_weight"]

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                stat_recorder["train/actor_loss"] = actor_loss.item()
                stat_recorder["train/masked_bc_loss"] = masked_bc_loss.item()
                stat_recorder["train/bc_loss"] = bc_loss.mean().item()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super(PVPTD3, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(
        self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
                                                                                          io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super(PVPTD3, self).save_replay_buffer(path_replay)

    def load_replay_buffer(
        self,
        path_human: Union[str, pathlib.Path, io.BufferedIOBase],
        path_replay: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.human_data_buffer = load_from_pkl(path_human, self.verbose)
        assert isinstance(
            self.human_data_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.human_data_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.human_data_buffer.handle_timeout_termination = False
            self.human_data_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        super(PVPTD3, self).load_replay_buffer(path_replay, truncate_last_traj)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_timesteps: int = 2000,
        buffer_save_timesteps: int = 2000,
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = True,
        load_buffer: bool = False,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        warmup: bool = False,
        warmup_steps: int = 5000,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
        callback.on_training_start(locals(), globals())
        if warmup:
            assert load_buffer, "warmup is useful only when load buffer"
            print("Start warmup with steps: " + str(warmup_steps))
            self.train(batch_size=self.batch_size, gradient_steps=warmup_steps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            if save_buffer and self.num_timesteps > 0 and self.num_timesteps % buffer_save_timesteps == 0:
                buffer_location_human = os.path.join(
                    save_path_human, "human_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                buffer_location_replay = os.path.join(
                    save_path_replay, "replay_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self


class PVPES(PVPTD3):
    actor_update_count = 0

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        stat_recorder = defaultdict(list)

        for step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data_agent = None
            replay_data_human = None

            if self.extra_config["adaptive_batch_size"]:
                if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
                    replay_data_human = self.human_data_buffer.sample(
                        int(batch_size), env=self._vec_normalize_env, return_all=True
                    )

                    human_data_size = len(replay_data_human.observations)
                    human_data_size = max(1, self.extra_config["agent_data_ratio"] * human_data_size)
                    human_data_size = int(human_data_size)

                    replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env)

                elif self.human_data_buffer.pos > 0:
                    replay_data_human = self.human_data_buffer.sample(
                        batch_size, env=self._vec_normalize_env, return_all=True
                    )
                elif self.replay_buffer.pos > 0:
                    replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                else:
                    break

            else:

                if self.replay_buffer.pos > batch_size and self.human_data_buffer.pos > batch_size:
                    replay_data_agent = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                    replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                elif self.human_data_buffer.pos > batch_size:
                    replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)
                elif self.replay_buffer.pos > batch_size:
                    replay_data_agent = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                else:
                    break

            current_q_novice_values = current_q_behavior_values = None
            if replay_data_human is not None:
                # Augment the reward / dones here.

                current_q_behavior_values = self.critic(
                    replay_data_human.observations, replay_data_human.actions_behavior
                )
                current_q_behavior_values = np.mean([q.mean().item() for q in current_q_behavior_values])
                current_q_novice_values = self.critic(replay_data_human.observations, replay_data_human.actions_novice)
                current_q_novice_values = np.mean([q.mean().item() for q in current_q_novice_values])

                replay_data_human_positive = copy.deepcopy(replay_data_human)
                replay_data_human_negative = replay_data_human

                if self.extra_config["reward_0_for_positive"]:
                    replay_data_human_positive.rewards.fill_(0)
                else:

                    if self.extra_config["use_weighted_reward"]:
                        w = (replay_data_human_positive.takeover_log_prob)
                        w = torch.exp(w)
                        w = torch.clamp(w, 0, 1)
                        w = 1 - w
                        # w = (w - w.min()) / (w.max() - w.min())
                        replay_data_human_positive.rewards.copy_(w)
                    else:
                        replay_data_human_positive.rewards.fill_(1)

                if self.extra_config["reward_0_for_negative"]:
                    replay_data_human_negative.rewards.fill_(0)
                else:
                    if self.extra_config["use_weighted_reward"]:
                        # w = (-replay_data_human_negative.takeover_log_prob)
                        # w = (w - w.min()) / (w.max() - w.min())
                        w = (replay_data_human_positive.takeover_log_prob)
                        w = torch.exp(w)
                        w = torch.clamp(w, 0, 1)
                        w = 1 - w
                        replay_data_human_negative.rewards.copy_(-w)
                    else:
                        replay_data_human_negative.rewards.fill_(-1)

                replay_data_human_negative.actions_behavior.copy_(replay_data_human_negative.actions_novice)

                if self.extra_config["no_done_for_negative"]:
                    pass
                else:
                    replay_data_human_negative.dones.fill_(1)

                if self.extra_config["no_done_for_positive"]:
                    pass
                else:
                    replay_data_human_positive.dones.fill_(1)
                if self.extra_config["remove_negative"]:
                    replay_data_human = replay_data_human_positive
                else:
                    replay_data_human = concat_samples(replay_data_human_positive, replay_data_human_negative)

            if self.extra_config["reward_1_for_all"]:
                if replay_data_agent is not None:
                    replay_data_agent.rewards.fill_(1)

            if replay_data_human is not None and replay_data_agent is None:
                replay_data = replay_data_human
            elif replay_data_human is None and replay_data_agent is not None:
                replay_data = replay_data_agent
            else:
                replay_data = concat_samples(replay_data_agent, replay_data_human)

            if self.extra_config["reward_n2_for_intervention"]:
                replay_data.rewards[replay_data.next_intervention_start.bool()] = -2

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions_behavior.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # PZH NOTE: For Early Stop PVP, we can consider the environments dones when human involved.
                # and at this moment an instant reward +1 or -1 is given.
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # print("BS: ", len(replay_data.observations))

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions_behavior)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            # critic_losses.append(critic_loss.item())

            stat_recorder["q_value_behavior"].append(
                current_q_behavior_values if current_q_behavior_values is not None else float("nan")
            )
            stat_recorder["q_value_novice"].append(
                current_q_novice_values if current_q_novice_values is not None else float("nan")
            )

            stat_recorder["q_value"].append(current_q_values[0].mean().item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.logger.record("train/critic_loss", critic_loss.item())

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations
                                                                                          )).mean()

                if self.extra_config["add_bc_loss"] and replay_data_human.dones.shape[0] > 0:
                    bc_loss = F.mse_loss(replay_data_human.actions_behavior, self.actor(replay_data_human.observations))
                    actor_loss += bc_loss

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                self.logger.record("train/actor_loss", actor_loss.item())

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                self.actor_update_count += 1
                # print("Actor update count: ", self.actor_update_count)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))
