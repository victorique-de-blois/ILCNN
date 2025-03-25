"""
Implemented based on SAC (as it uses stochastic policy)
"""
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
from pvp.sb3.haco.haco_buffer import PrefReplayBuffer


def biased_bce_with_logits(adv1, adv2, y, bias=1.0, cbias = 0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1 - cbias
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()

    return loss, accuracy

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
        for k in ["agent_data_ratio", "bc_loss_weight", "dpo_loss_weight", "alpha", "bias"]:
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
            stat_recorder["critic_loss"] = critic_loss.item()

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
                stat_recorder["actor_loss"] = actor_loss.item()
                stat_recorder["masked_bc_loss"] = masked_bc_loss.item()
                stat_recorder["bc_loss"] = bc_loss.mean().item()

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
                self.logger.info("Saving..." + str(buffer_location_human))
                self.logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self



class COMB(PVPTD3):
    def __init__(self, *args, **kwargs):
        super(COMB, self).__init__(*args, **kwargs)
        self.preference_buffer = HACOReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs
        )
        self.imagreplay_buffer = PrefReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                future_steps=1,
                **self.replay_buffer_kwargs,
        )
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["preference_buffer", "imagreplay_buffer", "human_data_buffer"]
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer])

        stat_recorder = defaultdict(list)

        for step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            if self.human_data_buffer.pos == 0:
                break
            replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
            preference_data = self.imagreplay_buffer.sample(int(batch_size), env=self._vec_normalize_env)
            
            new_action = self.actor(replay_data.observations)
            bc_loss = F.mse_loss(replay_data.actions_behavior, new_action, reduction="none").mean()
            
            pos_obs, pos_action = preference_data.pos_observations.squeeze(), preference_data.pos_actions.squeeze()
            neg_obs, neg_action = preference_data.neg_observations.squeeze(), preference_data.neg_actions.squeeze()
            
            def get_log_prob(obs, target_action):
                mean = self.actor(obs)
                log_prob = -((mean - target_action) ** 2).sum(dim = -1)
                return log_prob
            
            alpha, bias = self.extra_config["alpha"], self.extra_config["bias"]
            log_prob_pos = get_log_prob(pos_obs, pos_action)
            log_prob_neg = get_log_prob(neg_obs, neg_action)
            adv_pos, adv_neg = alpha * log_prob_pos, alpha * log_prob_neg
            label = torch.ones_like(adv_pos)
            dpo_loss, accuracy = biased_bce_with_logits(adv_neg, adv_pos, label.float(), bias=bias)
            
            bc_loss_weight, dpo_loss_weight = self.extra_config["bc_loss_weight"], self.extra_config["dpo_loss_weight"]
            loss = bc_loss_weight * bc_loss + dpo_loss_weight * dpo_loss
            
            self.actor.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            
            stat_recorder["bc_loss"].append(bc_loss.item() if bc_loss is not None else float('nan'))
            stat_recorder["cpl_loss"].append(dpo_loss.item() if dpo_loss is not None else float('nan'))
            stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy is not None else float('nan'))
            stat_recorder["loss"].append(loss.item() if loss is not None else float('nan'))

        self._n_updates += gradient_steps
        self.logger.record("train/predicted_steps", self.preference_buffer.pos)
        self.logger.record("train/human_involved_steps", self.human_data_buffer.pos)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

# class COMB(TD3):
#     def __init__(
#         self,
#         policy: Union[str, Type[HACOPolicy]],
#         env: Union[GymEnv, str],
#         learning_rate: float = 1e-4,
#         buffer_size: int = 100,  # Shrink the size to reduce memory consumption when testing
#         learning_starts: int = 100,
#         batch_size: int = 256,
#         tau: float = 0.005,
#         gamma: float = 0.99,
#         train_freq: Union[int, Tuple[int, str]] = 1,
#         gradient_steps: int = 1,
#         action_noise: Optional[ActionNoise] = None,
#         replay_buffer_class: Optional[HACOReplayBuffer] = HACOReplayBuffer,  # PZH: !! Use HACO Replay Buffer
#         replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
#         optimize_memory_usage: bool = True,
#         ent_coef: Union[str, float] = "auto",
#         target_update_interval: int = 1,
#         target_entropy: Union[str, float] = "auto",
#         use_sde: bool = False,
#         sde_sample_freq: int = -1,
#         use_sde_at_warmup: bool = False,
#         tensorboard_log: Optional[str] = None,
#         create_eval_env: bool = False,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         seed: Optional[int] = None,
#         device: Union[th.device, str] = "auto",
#         _init_setup_model: bool = True,

#         # PZH: Our new introduce hyper-parameters
#         cql_coefficient=1,
#         monitor_wrapper=False,
#         future_steps=5,
#         bias=0,
#         cbias=0,
#         alpha=0.1,
#         poso="pos_observations",
#         posa="pos_actions",
#         nego="neg_observations",
#         nega="neg_actions",
#         cpl_loss_weight=1.0,
#         bc_loss_weight=1.0,
#         use_bc_only=False,
#         use_bcmse_only=False,
#         stop_freq=5,
#         use_ref=False,
#         imgweight=1.0,
#         img_future_steps=1,
#         **kwargs,
#     ):

#         assert replay_buffer_class == HACOReplayBuffer
#         self.bias = bias
#         self.cbias, self.alpha, self.bc_loss_weight, self.use_bc_only = cbias, alpha, bc_loss_weight, use_bc_only
#         self.poso, self.posa, self.nego, self.nega = poso, posa, nego, nega
#         self.cpl_loss_weight, self.use_bcmse_only = cpl_loss_weight, use_bcmse_only
#         self.use_ref = use_ref

#         super().__init__(
#             policy,
#             env,
#             learning_rate,
#             buffer_size,
#             learning_starts,
#             batch_size,
#             tau,
#             gamma,
#             train_freq,
#             gradient_steps,
#             action_noise,
#             replay_buffer_class=replay_buffer_class,
#             replay_buffer_kwargs=replay_buffer_kwargs,
#             policy_kwargs=policy_kwargs,
#             tensorboard_log=tensorboard_log,
#             verbose=verbose,
#             device=device,
#             create_eval_env=create_eval_env,
#             seed=seed,
#             optimize_memory_usage=optimize_memory_usage,
#             monitor_wrapper=monitor_wrapper
#         )
#         # PZH: Define some new variables
#         self.cql_coefficient = cql_coefficient
#         from pvp.sb3.haco.haco_buffer import PrefReplayBuffer
#         # self.future_steps = future_steps
#         self.stop_freq = stop_freq
#         self.imgweight=imgweight
#         self.img_future_steps = img_future_steps
        
#         self.imagreplay_buffer = PrefReplayBuffer(
#                 self.buffer_size,
#                 self.observation_space,
#                 self.action_space,
#                 self.device,
#                 n_envs=self.n_envs,
#                 optimize_memory_usage=self.optimize_memory_usage,
#                 future_steps=self.img_future_steps,
#                 **self.replay_buffer_kwargs,
#         )

#     # def _create_aliases(self) -> None:
#     #     super()._create_aliases()
#     #     self.cost_critic = self.policy.cost_critic
#     #     self.cost_critic_target = self.policy.cost_critic_target
#     def _store_transition(
#         self,
#         replay_buffer,
#         buffer_action: np.ndarray,
#         new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
#         reward: np.ndarray,
#         dones: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         if infos[0]["takeover"] or infos[0]["takeover_start"]:
#             replay_buffer = self.human_data_buffer
#         super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
#     def _create_aliases(self) -> None:
#         import copy
#         self.policy_target = copy.deepcopy(self.policy)
#     def train(self, gradient_steps: int, batch_size: int = 64) -> None:
#         # Switch to train mode (this affects batch norm / dropout)
#         self.policy.actor.set_training_mode(True)
#         # Update optimizers learning rate
#         self._update_learning_rate([self.policy.actor.optimizer])

#         stat_recorder = defaultdict(list)

#         for gradient_step in range(gradient_steps):
#             # ========== Compute the CPL loss ==========
#             bc_loss_weight = self.bc_loss_weight 

#             # Sample replay buffer
#             replay_data_human = None
#             replay_data_agent = None
#             if self.human_data_buffer.pos > 0:
#                 replay_data_human = self.human_data_buffer.sample(batch_size, env=self._vec_normalize_env)

#             if self.imagreplay_buffer.pos > 0:
#                 replay_data = self.imagreplay_buffer.sample(batch_size, env=self._vec_normalize_env)
#             else:
#                 loss = None
#                 break

#             alpha = self.alpha
#             accuracy = cpl_loss = bc_loss = None
            
#             mask = replay_data.mask
            
#             pos_obs, pos_action = getattr(replay_data, self.poso), getattr(replay_data, self.posa)
#             mask_reshape = mask.view(-1)

#             def get_log_prob(policy, pos_obs, pos_action):
#                     pos_obs_reshape = th.reshape(pos_obs, (-1, pos_obs.shape[-1]))
#                     pos_action_reshape = th.reshape(pos_action, (-1, pos_action.shape[-1]))
#                     mean = policy.actor(pos_obs_reshape)
#                     log_prob_pos = -((mean - pos_action_reshape) ** 2).sum(dim = -1)
#                     #log_prob_pos = th.reshape(log_prob_pos, pos_obs.shape[:-1]) * mask
#                     #log_prob_pos = log_prob_pos.sum(dim = -1)
#                     log_prob_pos = log_prob_pos * mask_reshape
#                     return log_prob_pos
            
#             log_prob_pos = get_log_prob(self.policy, pos_obs, pos_action)
#             if self.use_ref:
#                 with th.no_grad():
#                     log_prob_pos_ref = get_log_prob(self.policy_ref, pos_obs, pos_action)
#                 log_prob_pos -= log_prob_pos_ref

            
#             neg_obs, neg_action = getattr(replay_data, self.nego), getattr(replay_data, self.nega)
            
#             log_prob_neg = get_log_prob(self.policy, neg_obs, neg_action)
            
#             if self.use_ref:
#                 with th.no_grad():
#                     log_prob_neg_ref = get_log_prob(self.policy_ref, neg_obs, neg_action)
#                 log_prob_neg -= log_prob_neg_ref
            
#             adv_pos, adv_neg = alpha * log_prob_pos, alpha * log_prob_neg
#             label = torch.ones_like(adv_pos)
#             cpl_loss, accuracy = biased_bce_with_logits(adv_neg, adv_pos, label.float(), bias=self.bias, cbias=self.cbias)

#             if replay_data_human is not None:
#                 new_action = self.policy.actor(replay_data_human.observations)
#                 bc_loss = F.mse_loss(replay_data_human.actions_behavior, new_action, reduction="none").mean() ##-log_probs_tmp1.mean()
#                 stat_recorder["new_action_steering"] = new_action[:, 0].mean().item()
#                 stat_recorder["new_action_abs_steering"] = th.abs(new_action[:, 0]).mean().item()
#                 stat_recorder["new_action_accerler"] = new_action[:, 1].mean().item()

#             # Aggregate losses
#             if bc_loss is None and cpl_loss is None:
#                 break
            
#             if not self.use_bc_only and not self.use_bcmse_only:
#                 loss = bc_loss_weight * (bc_loss
#                                      if bc_loss is not None else 0.0) + self.cpl_loss_weight * (cpl_loss if cpl_loss is not None else 0.0)

#             self._optimize_actor(actor_loss=loss)

#             # Stats
#             stat_recorder["bc_loss"].append(bc_loss.item() if bc_loss is not None else float('nan'))
#             stat_recorder["cpl_loss"].append(cpl_loss.item() if cpl_loss is not None else float('nan'))
#             stat_recorder["cpl_accuracy"].append(accuracy.item() if accuracy is not None else float('nan'))
#             stat_recorder["loss"].append(loss.item() if loss is not None else float('nan'))


#         self._n_updates += gradient_steps

#         self.logger.record("train/num_traj", self.imagreplay_buffer.pos)
#         self.logger.record("train/n_updates", self._n_updates)
#         self.logger.record("train/human_involved_steps", self.human_data_buffer.pos)
#         try:
#             import wandb
#             wandb.log(self.logger.name_to_value, step=self.num_timesteps)
#         except:
#             pass
#         for key, values in stat_recorder.items():
#             self.logger.record("train/{}".format(key), np.mean(values))

#     def _optimize_actor(self, actor_loss):
#         self.policy.actor.optimizer.zero_grad()
#         actor_loss.backward()
#         self.policy.actor.optimizer.step()

#     # def _optimize_critics(self, merged_critic_loss):
#     #     self.critic.optimizer.zero_grad()
#     #     merged_critic_loss.backward()
#     #     self.critic.optimizer.step()

#     def _excluded_save_params(self) -> List[str]:
#         return super()._excluded_save_params() + ["imagreplay_buffer", "human_data_buffer", "policy_ref"]
    
#     def _get_torch_save_params(self):
#         ret = super()._get_torch_save_params()
#         # print(1)
#         return (['policy'], [])

#     def _setup_model(self) -> None:
#         super()._setup_model()
#         self.human_data_buffer = HACOReplayBuffer(
#             self.buffer_size,
#             self.observation_space,
#             self.action_space,
#             self.device,
#             n_envs=self.n_envs,
#             optimize_memory_usage=self.optimize_memory_usage,
#             **self.replay_buffer_kwargs
#         )
#         # self.human_data_buffer