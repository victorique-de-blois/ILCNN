#goal: add fake (imag) pos samples to neg trajs. see performance of algos
import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.policy.env_input_policy import EnvInputPolicy

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv

FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()


def get_expert():
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})

    # Initialize agent
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metadrive_pvp_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert = get_expert()


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None
    from collections import deque 
    advantages = deque(maxlen = 200)
    drawn_points = []
    
    def __init__(self, config):
        super(FakeHumanEnv, self).__init__(config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins ** 2)
        else:
            return super(FakeHumanEnv, self).action_space

    # def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
    #     if self.config["use_discrete"]:
    #         print(111)
    #         return int(actions)
    #     else:
    #         return actions

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(FakeHumanEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "free_level": 0.9,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
                "future_steps": 20,
                "takeover_see": 20,
                "stop_freq": 10,
                "stop_img_samples": 3, 
                "expert_noise": 0,
            }
        )
        return config

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action
    def get_state(self) -> dict:
        import copy
        state = copy.deepcopy(self.vehicle.get_state())
        return copy.deepcopy(state)

    def set_state(self, state: dict):
        self.vehicle.set_state(state)

    def decide_takeover(self, obs, future_steps_predict):
        predicted_traj_real, info_real = self.predict_agent_future_trajectory(obs, future_steps_predict)
        assert info_real["failure"] == (info_real["total_reward"] < 0)
        return info_real["failure"]
    
    def store_preference_pairs(self, predicted_traj, future_steps_preference, expert_action):
        for step in range(min(len(predicted_traj) - 1, future_steps_preference)):
            step_info = {
                "obs": predicted_traj[step]["obs"].copy(),
                "action": expert_action.copy(),
                "next_obs": predicted_traj[step]["obs"].copy(),
                "done": False,
            }
            positive_traj = [step_info].copy()
            negative_traj = predicted_traj[step+1:]
            self.model.imagreplay_buffer.add(positive_traj, negative_traj)
    
    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover
        
        future_steps = self.config["future_steps"]
        stop_freq = self.config["stop_freq"]
        stop_img_samples = self.config["stop_img_samples"]
        expert_noise_bound = self.config["expert_noise"]
        
        if self.expert is None:
                global _expert
                self.expert = _expert
        
        last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
        distribution = self.expert.get_distribution(last_obs)
        log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
        action_prob = log_prob.exp().detach().cpu().numpy()
        action_prob = action_prob[0]
        expert_action, _  = self.expert.predict(self.last_obs, deterministic=True)
        enoise = np.random.randn(2) * expert_noise_bound
        expert_action = np.clip(enoise + expert_action, self.action_space.low, self.action_space.high)
        
        if (self.total_steps % stop_freq == 0):
            self.takeover = self.decide_takeover(self.last_obs, future_steps)

        if self.takeover:
            predicted_traj, info2 = self.predict_agent_future_trajectory(self.last_obs, future_steps, action_behavior=self.agent_action.copy())
            if self.config["use_discrete"]:
                expert_action = self.continuous_to_discrete(expert_action)
                expert_action = self.discrete_to_continuous(expert_action)
            actions = expert_action
            if hasattr(self, "model") and hasattr(self.model, "imagreplay_buffer"):
                self.store_preference_pairs(predicted_traj, stop_img_samples, expert_action.copy())
            
        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        
        self.vehicle.real = False
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()

        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            self.render(
                # mode="top_down",
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    # "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])
        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        """Compared to original one, here we don't call expert_policy, but directly get self.last_takeover."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        if hasattr(self,"drawer"):
                drawer = self.drawer # create a point drawer
        else:
                self.drawer = self.engine.make_point_drawer(scale=3)
                
        self.last_obs = o
        self.last_takeover = False
        for npp in self.drawn_points:
            npp.detachNode()
            self.drawer._dying_points.append(npp)
        self.drawn_points = []
        return o, info
    

if __name__ == "__main__":
    env = FakeHumanEnv(dict(use_render=True, num_scenarios=1, traffic_density=0))
    env.reset()
    ss = 0
    while True:
        if ss < 10:
            _, _, done, info = env.step([0, 1])
        else:
            _, _, done, info = env.step([0, 0.1])
        ss += 1
        # done = tm or tc
        # env.render(mode="topdown")
        if done:
            print(info)
            env.reset()
            ss = 0
