import copy
import time
from collections import deque

import numpy as np
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicyWithoutBrake
from metadrive.utils.math import safe_clip

ScreenMessage.SCALE = 0.1
from metadrive.utils.math import norm
from panda3d.core import LVector3
import math
import torch
from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector, panda_heading

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    "traffic_density": 0.06,

    # Reward and cost setting:    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    "cos_similarity": False,  # If True, the takeover cost will be the cos sim between a_h and a_n. Useless in PVP.

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyWithoutBrake,
    "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel].
    "only_takeover_start_cost": False,  # If True, only return a cost when takeover starts. Useless in PVP.

    # Visualization
    "vehicle_config": {
        "show_dest_mark": True,  # Show the destination in a cube.
        "show_line_to_dest": True,  # Show the line to the destination.
        "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
    },
    "horizon": 1500,
}


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    Human-in-the-loop Env Wrapper for the Safety Env in MetaDrive.
    Add code for computing takeover cost and add information to the interface.
    """
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    takeover = False
    takeover_recorder = deque(maxlen=2000)
    agent_action = None
    in_pause = False
    start_time = time.time()

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return config

    def reset(self, *args, **kwargs):
        self.takeover = False
        self.agent_action = None
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        # The training code is for older version of gym, so we discard the additional info from the reset.
        return obs

    def _get_step_return(self, actions, engine_info):
        """Compute takeover cost here."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        d = tm or tc

        shared_control_policy = self.engine.get_policy(self.agent.id)
        last_t = self.takeover
        self.takeover = shared_control_policy.takeover if hasattr(shared_control_policy, "takeover") else False
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
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        """Out of road condition"""
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """Add additional information to the interface."""
        self.agent_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        while self.in_pause:
            self.engine.taskMgr.step()

        self.takeover_recorder.append(self.takeover)
        if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        self.total_steps += 1

        self.total_takeover_count += 1 if self.takeover else 0
        ret[-1]["total_takeover_count"] = self.total_takeover_count

        return ret

    def stop(self):
        """Toggle pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Introduce additional key 'e' to the interface."""
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist

    def get_state(self):
        """
        Fetch more information
        """
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicle = self.vehicle
        state = super(BaseVehicle, vehicle).get_state()
        state.update(
            {
                "steering": vehicle.steering,
                "throttle_brake": vehicle.throttle_brake,
                "crash_vehicle": vehicle.crash_vehicle,
                "crash_object": vehicle.crash_object,
                "crash_building": vehicle.crash_building,
                "crash_sidewalk": vehicle.crash_sidewalk,
                "crash_human": vehicle.crash_human,
                "size": (vehicle.LENGTH, vehicle.WIDTH, vehicle.HEIGHT),
                "length": vehicle.LENGTH,
                "width": vehicle.WIDTH,
                "height": vehicle.HEIGHT,
                "red_light": vehicle.red_light,
                "yellow_light": vehicle.yellow_light,
                "green_light": vehicle.green_light,
                "on_yellow_continuous_line": vehicle.on_yellow_continuous_line,
                "on_white_continuous_line": vehicle.on_white_continuous_line,
                "on_broken_line": vehicle.on_broken_line,
                "on_crosswalk": vehicle.on_crosswalk,
                "contact_results": list(vehicle.contact_results) if vehicle.contact_results is not None else [],
            }
        )
        state.update({
            "last_current_action": list(vehicle.last_current_action),
            "last_position": vehicle.last_position,
            "last_heading_dir": vehicle.last_heading_dir,
            "dist_to_left_side": vehicle.dist_to_left_side,
            "dist_to_right_side": vehicle.dist_to_right_side,
            "last_velocity": vehicle.last_velocity,
            "last_speed": vehicle.last_speed,
            "out_of_route": vehicle.out_of_route,
            "on_lane": vehicle.on_lane,
            "spawn_place": vehicle.spawn_place,
            "takeover": vehicle.takeover,
            "expert_takeover": vehicle.expert_takeover,
            "energy_consumption": vehicle.energy_consumption,
            "break_down": vehicle.break_down,
        })
        
        if vehicle.navigation is not None:
            state["spawn_road"] = vehicle.navigation.spawn_road
            state["destination"] = (vehicle.navigation.final_road.start_node, vehicle.navigation.final_road.end_node) if vehicle.navigation.final_road is not None else None
            state["checkpoints"] = vehicle.navigation.checkpoints 
            state["_target_checkpoints_index"] = vehicle.navigation._target_checkpoints_index

            state["current_road"] = (vehicle.navigation.current_road.start_node, vehicle.navigation.current_road.end_node) if vehicle.navigation.current_road is not None else None
            state["next_road"] = (vehicle.navigation.next_road.start_node, vehicle.navigation.next_road.end_node) if vehicle.navigation.next_road is not None else None
            state["final_road"] = (vehicle.navigation.final_road.start_node, vehicle.navigation.final_road.end_node) if vehicle.navigation.final_road is not None else None

            state["current_ref_lane_indices"] = [lane.index for lane in vehicle.navigation.current_ref_lanes] if vehicle.navigation.current_ref_lanes is not None else None
            state["next_ref_lane_indices"] = [lane.index for lane in vehicle.navigation.next_ref_lanes] if vehicle.navigation.next_ref_lanes is not None else None
            state["total_length"] = vehicle.navigation.total_length
            state["travelled_length"] = vehicle.navigation.travelled_length
            state["_last_long_in_ref_lane"] = vehicle.navigation._last_long_in_ref_lane

            state["_navi_info"] = vehicle.navigation._navi_info.tolist() if hasattr(vehicle.navigation, "_navi_info") and vehicle.navigation._navi_info is not None else None
            state["navi_arrow_dir"] = vehicle.navigation.navi_arrow_dir if hasattr(vehicle.navigation, "navi_arrow_dir") else None
            
        return copy.deepcopy(state)
        
    def set_state(self, state):
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        vehicle = self.vehicle
        super(BaseVehicle, vehicle).set_state(state)
        vehicle.set_throttle_brake(float(state["throttle_brake"]))
        vehicle.set_steering(float(state["steering"]))
        vehicle.last_current_action = deque(state["last_current_action"], maxlen=2)
        vehicle.last_position = state["last_position"]
        vehicle.last_heading_dir = state["last_heading_dir"]
        vehicle.dist_to_left_side = state["dist_to_left_side"]
        vehicle.dist_to_right_side = state["dist_to_right_side"]
        vehicle.last_velocity = state["last_velocity"]
        vehicle.last_speed = state["last_speed"]
        vehicle.out_of_route = state["out_of_route"]
        vehicle.on_lane = state["on_lane"]
        vehicle.spawn_place = state["spawn_place"]
        vehicle.takeover = state["takeover"]
        vehicle.expert_takeover = state["expert_takeover"]
        vehicle.energy_consumption = state["energy_consumption"]
        vehicle.break_down = state["break_down"]
        vehicle.crash_vehicle = state["crash_vehicle"]
        vehicle.crash_human = state["crash_human"]
        vehicle.crash_object = state["crash_object"]
        vehicle.crash_sidewalk = state["crash_sidewalk"]
        vehicle.crash_building = state["crash_building"]
        vehicle.red_light = state["red_light"]
        vehicle.yellow_light = state["yellow_light"]
        vehicle.green_light = state["green_light"]
        vehicle.on_yellow_continuous_line = state["on_yellow_continuous_line"]
        vehicle.on_white_continuous_line = state["on_white_continuous_line"]
        vehicle.on_broken_line = state["on_broken_line"]
        vehicle.on_crosswalk = state["on_crosswalk"]
        vehicle.contact_results = set(state["contact_results"]) if "contact_results" in state else set()
        if vehicle.navigation is not None:
            from metadrive.component.road_network import Road
            vehicle.navigation.spawn_road = state.get("spawn_road", None)
            dest = state.get("destination", None)
            if dest is not None:
                # 通过目的地信息重构 final_road 对象
                vehicle.navigation.final_road = Road(dest[0], dest[1])
            else:
                vehicle.navigation.final_road = None

            # 恢复路线规划相关信息
            vehicle.navigation.checkpoints = state.get("checkpoints", None)
            vehicle.navigation._target_checkpoints_index = state.get("_target_checkpoints_index", None)

            # 恢复当前、下一、最终道路（重构 Road 对象）
            current_road = state.get("current_road", None)
            if current_road is not None:
                vehicle.navigation.current_road = Road(current_road[0], current_road[1])
            else:
                vehicle.navigation.current_road = None

            next_road = state.get("next_road", None)
            if next_road is not None:
                vehicle.navigation.next_road = Road(next_road[0], next_road[1])
            else:
                vehicle.navigation.next_road = None

            final_road = state.get("final_road", None)
            if final_road is not None:
                vehicle.navigation.final_road = Road(final_road[0], final_road[1])
            else:
                vehicle.navigation.final_road = None

            # 恢复参考车道信息：这里假定你能通过 vehicle.navigation.map.road_network.get_lane(lane_index)
            current_ref_lane_indices = state.get("current_ref_lane_indices", None)
            if current_ref_lane_indices is not None:
                vehicle.navigation.current_ref_lanes = [vehicle.navigation.map.road_network.get_lane(idx) for idx in current_ref_lane_indices]
            else:
                vehicle.navigation.current_ref_lanes = None

            next_ref_lane_indices = state.get("next_ref_lane_indices", None)
            if next_ref_lane_indices is not None:
                vehicle.navigation.next_ref_lanes = [vehicle.navigation.map.road_network.get_lane(idx) for idx in next_ref_lane_indices]
            else:
                vehicle.navigation.next_ref_lanes = None

            # 恢复路线长度信息
            vehicle.navigation.total_length = state.get("total_length", 0.0)
            vehicle.navigation.travelled_length = state.get("travelled_length", 0.0)
            vehicle.navigation._last_long_in_ref_lane = state.get("_last_long_in_ref_lane", 0.0)

            # 恢复导航信息向量
            navi_info_list = state.get("_navi_info", None)
            if navi_info_list is not None:
                vehicle.navigation._navi_info = np.array(navi_info_list)
            else:
                vehicle.navigation._navi_info = None

            # 恢复箭头方向等信息
            vehicle.navigation.navi_arrow_dir = state.get("navi_arrow_dir", None)

    
    def predict_agent_future_trajectory(self, current_obs, action_behavior = None, return_all_states = False):
        info = dict()
        saved_state = self.get_state()
        
        all_states = []
        traj = []
        obs = current_obs
        total_reward = 0
        failure = False
        
        for step in range(self.config["future_steps"]):
            old_pos = copy.deepcopy(self.vehicle.position)
            action = action_behavior
            if action_behavior is None:
                action = self.agent_action
                if hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            
            if self.config["use_discrete"]:
                action = self.discrete_to_continuous(action)

            #actions = self._preprocess_actions(action) 
            dt = self.config["physics_world_step_size"] * self.config["decision_repeat"]
            self.vehicle.before_step(action)
                
            params = self.vehicle.get_dynamics_parameters()
            mass = params["mass"]
            max_engine_force = params["max_engine_force"]
            max_brake_force = params["max_brake_force"]

            throttle = self.vehicle.throttle_brake
            if throttle >= 0:
                if self.vehicle.speed >= self.vehicle.max_speed_m_s:
                    a = 0.0
                else:
                    engine_force = max_engine_force * throttle
                    a = engine_force / mass * 4
            else:
                brake_force = max_brake_force * abs(throttle)
                a = -brake_force / mass * 4

            new_speed = self.vehicle.speed + a * dt
            new_speed = max(new_speed, 0.0)

            step_info = self.vehicle.after_step()
            current_steering = self.vehicle.steering
            max_steering_rad = math.radians(self.vehicle.config["max_steering"])

            L = self.vehicle.FRONT_WHEELBASE + self.vehicle.REAR_WHEELBASE
            new_heading = self.vehicle.heading_theta + (new_speed / L) * math.tan(current_steering * max_steering_rad) * dt

            new_x = self.vehicle.position[0] + new_speed * dt * math.cos(new_heading)
            new_y = self.vehicle.position[1] + new_speed * dt * math.sin(new_heading)
            new_position = [new_x, new_y]
            new_velocity = [new_speed * math.cos(new_heading), new_speed * math.sin(new_heading)]

            self.set_position(new_position)
            self.vehicle.set_heading_theta(new_heading)
            self.vehicle.set_velocity(new_velocity)
            self.vehicle.navigation.update_localization(self.vehicle)
            r = self.reward_function('default_agent')[0]
            total_reward += r
            
            if return_all_states:
                all_states.append(self.get_state())
            d = self.done_function('default_agent')[0]

            new_obs = self.get_single_observation().observe(self.vehicle)
            
            traj.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": r,
                "next_obs": new_obs.copy(),
                "done": d,
                "pos": old_pos,
                "next_pos": copy.deepcopy(self.vehicle.position),
            })
            obs = new_obs.copy()
            
            if d:
                failure = (r < 0)
                break
        
        self.set_state(saved_state)
        
        failure = failure or (total_reward <= 10) #CHY: Failure if too slow.
        info["all_states"] = all_states
        info["failure"] = failure
        info["total_reward"] = total_reward
        
        return traj, info
    
    def set_position(self, position, height=None):
        vehicle = self.vehicle
        assert len(position) == 2 or len(position) == 3
        if len(position) == 3:
            height = position[-1]
            position = position[:-1]
        else:
            if height is None:
                height = vehicle.origin.getPos()[-1]
        vehicle.origin.setPos(panda_vector(position, height))
    

    def draw_points(self, points, colors=None):
        """
        Draw a set of points with colors
        Args:
            points: a set of 3D points
            colors: a list of color for each point

        Returns: None

        """
        from panda3d.core import VBase4, NodePath, Material
        from panda3d.core import LVecBase4f
        from metadrive.engine.asset_loader import AssetLoader
        drawer = self.drawer
        new_points = []
        for k, point in enumerate(points):
            if len(drawer._dying_points) > 0:
                np = drawer._dying_points.pop()
            else:
                np = NodePath("debug_point")
                model = drawer.engine.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                model.setScale(drawer.scale)
                model.reparentTo(np)
            material = Material()
            if colors:
                material.setBaseColor(LVecBase4f(*colors[k]))
            else:
                material.setBaseColor(LVecBase4f(1, 1, 1, 1))
            material.setShininess(64)
            # material.setEmission((1, 1, 1, 1))
            np.setMaterial(material, True)
            np.setPos(*point)
            np.reparentTo(drawer)
            drawer._existing_points.append(np)
            new_points.append(np)
        return new_points

if __name__ == "__main__":
    env = HumanInTheLoopEnv({
        "manual_control": True,
        "use_render": True,
    })
    env.reset()
    while True:
        _, _, done, _ = env.step([0, 0])
        if done:
            env.reset()
