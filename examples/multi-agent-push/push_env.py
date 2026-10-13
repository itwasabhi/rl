# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A MuJoCo Playground (MJX) scene where N car-like robots push a box to a goal.

:class:`PushEnv` is a plain single-agent ``MjxEnv`` whose flat action and
observation vectors are laid out per car, so that
:class:`~torchrl.envs.MujocoPlaygroundWrapper` can split them into one agent per
car with :meth:`PushEnv.agent_mapping`. See ``mappo.py`` for the wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, State
from torchrl.envs import MujocoPlaygroundAgentMapping, MujocoPlaygroundAgentSpec

# Size of the egocentric block each car observes (independent of the car count).
CAR_OBS_SIZE = 8
# Size of the part of the observation shared by every car.
SHARED_OBS_SIZE = 4

_CAR_COLORS = ("0.9 0.3 0.2 1", "0.2 0.5 0.9 1", "0.2 0.7 0.3 1", "0.9 0.7 0.2 1")


@dataclass(frozen=True)
class PushTaskConfig:
    """Task parameters of :class:`PushEnv`. Use ``dataclasses.replace`` to tweak.

    Args:
        arena_half_size (float): half side of the square arena, in meters.
            Defaults to ``1.5``.
        box_start_range (float): the box starts uniformly in
            ``[-box_start_range, box_start_range]^2`` at every reset.
            Defaults to ``0.5``.
        goal_distance_range (tuple of float): the goal is sampled at a distance
            uniform in ``(low, high)`` from the box start, in a random direction
            (clipped to stay inside the arena). Defaults to ``(0.6, 1.0)``.
        car_spawn_radius (tuple of float): cars start on a ring around the box,
            evenly spaced in angle with a random offset; the ring radius is
            uniform in ``(low, high)``. Defaults to ``(0.4, 0.7)``.
        max_speed (float): forward speed (m/s) of an action ``v = 1``.
            Defaults to ``1.0``.
        max_yaw_rate (float): yaw rate (rad/s) of an action ``omega = 1``.
            Defaults to ``2.5``.
        ctrl_dt (float): control period in seconds. Defaults to ``0.05``.
        sim_dt (float): physics step in seconds; ``ctrl_dt / sim_dt`` physics
            steps run per control step. Defaults to ``0.01``.
        episode_length (int): number of control steps per episode. Episodes never
            end earlier, so every vmapped env resets on the same step (the
            wrapper does not support partial resets). Defaults to ``200``.
        success_radius (float): the box counts as delivered when it is within this
            distance of the goal. Defaults to ``0.12``.
        builtin_reward (bool): if ``False`` the env reward is always ``0`` so a
            transform can supply another one. Defaults to ``True``.
        w_progress (float): weight of the per-step decrease of the box-goal
            distance. Defaults to ``10.0``.
        w_reach (float): weight of the (negative) mean car-box distance.
            Defaults to ``0.05``.
        w_success (float): bonus per step while the box is within
            ``success_radius`` of the goal. Defaults to ``0.5``.
        w_action (float): weight of the (negative) mean squared action.
            Defaults to ``0.01``.
    """

    arena_half_size: float = 1.5
    box_start_range: float = 0.5
    goal_distance_range: tuple[float, float] = (0.6, 1.0)
    car_spawn_radius: tuple[float, float] = (0.4, 0.7)
    max_speed: float = 1.0
    max_yaw_rate: float = 2.5
    ctrl_dt: float = 0.05
    sim_dt: float = 0.01
    episode_length: int = 200
    success_radius: float = 0.12
    builtin_reward: bool = True
    w_progress: float = 10.0
    w_reach: float = 0.05
    w_success: float = 0.5
    w_action: float = 0.01


def _scene_xml(num_cars: int, cfg: PushTaskConfig) -> str:
    """Generate the MJCF of the arena, the box, the goal marker and the cars."""
    a = cfg.arena_half_size
    walls = "\n".join(
        f'<geom name="wall_{i}" type="box" pos="{px} {py} 0.05" size="{sx} {sy} 0.05" '
        f'rgba="0.4 0.4 0.4 1"/>'
        for i, (px, py, sx, sy) in enumerate(
            [
                (a + 0.05, 0, 0.05, a + 0.1),
                (-a - 0.05, 0, 0.05, a + 0.1),
                (0, a + 0.05, a + 0.1, 0.05),
                (0, -a - 0.05, a + 0.1, 0.05),
            ]
        )
    )
    cars, actuators = [], []
    for i in range(num_cars):
        rgba = _CAR_COLORS[i % len(_CAR_COLORS)]
        cars.append(
            f"""
    <body name="car_{i}" pos="0 0 0.03">
      <joint name="car_{i}_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="car_{i}_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="car_{i}_yaw" type="hinge" axis="0 0 1" damping="0.01" armature="0.01"/>
      <geom name="car_{i}_chassis" type="box" size="0.09 0.06 0.03" mass="1" rgba="{rgba}"/>
      <geom name="car_{i}_nose" type="sphere" size="0.025" pos="0.09 0 0.02"
            contype="0" conaffinity="0" rgba="1 1 1 1"/>
    </body>"""
        )
        # Velocity servos on the three planar dofs; `step` sets their targets so
        # that the commanded motion always follows the heading.
        actuators.append(
            f"""
    <velocity name="car_{i}_vx" joint="car_{i}_x" kv="40" ctrlrange="-3 3" forcerange="-30 30"/>
    <velocity name="car_{i}_vy" joint="car_{i}_y" kv="40" ctrlrange="-3 3" forcerange="-30 30"/>
    <velocity name="car_{i}_wz" joint="car_{i}_yaw" kv="2" ctrlrange="-6 6" forcerange="-3 3"/>"""
        )
    return f"""
<mujoco model="multi_agent_push">
  <option timestep="{cfg.sim_dt}" gravity="0 0 0" integrator="implicitfast" iterations="8" ls_iterations="8"/>
  <worldbody>
    <geom name="floor" type="plane" size="{a + 0.3} {a + 0.3} 0.1" rgba="0.85 0.85 0.8 1"
          contype="0" conaffinity="0"/>
    {walls}
    <body name="goal" mocap="true" pos="0 0 0.002">
      <geom type="cylinder" size="{cfg.success_radius} 0.002" rgba="0.2 0.9 0.2 0.5"
            contype="0" conaffinity="0"/>
    </body>
    <body name="box" pos="0 0 0.04">
      <joint name="box_x" type="slide" axis="1 0 0" damping="3"/>
      <joint name="box_y" type="slide" axis="0 1 0" damping="3"/>
      <joint name="box_yaw" type="hinge" axis="0 0 1" damping="0.02" armature="0.005"/>
      <geom name="box" type="box" size="0.1 0.1 0.04" mass="0.5" rgba="0.8 0.6 0.3 1"/>
    </body>
    {"".join(cars)}
  </worldbody>
  <actuator>{"".join(actuators)}
  </actuator>
</mujoco>"""


class PushEnv(MjxEnv):
    """``num_cars`` non-holonomic cars push a box to a goal on a flat arena.

    Each car is a unicycle: its action is ``(v, omega)`` in ``[-1, 1]^2``
    (forward speed and yaw rate), and the velocity it commands always points
    along its heading, so it cannot slide sideways by itself (contacts with the
    box or another car can still nudge it). All cars share one team reward.

    The flat action has size ``2 * num_cars`` (car ``i`` owns entries
    ``[2i, 2i + 1]``). The flat observation has size
    ``CAR_OBS_SIZE * num_cars + SHARED_OBS_SIZE`` and is laid out as one block
    per car followed by a shared block, which is what :meth:`agent_mapping`
    slices:

    - car block (egocentric, in the car frame, ``CAR_OBS_SIZE = 8``): box
      position, goal position, nearest other car position (zeros for a single
      car), forward speed and yaw rate;
    - shared block (``SHARED_OBS_SIZE = 4``): box-to-goal vector in the world
      frame, sine and cosine of the box yaw.

    Args:
        task (PushTaskConfig, optional): the task parameters. Defaults to
            ``PushTaskConfig()``. See also the :meth:`push_to_goal` preset.
        num_cars (int, optional): number of cars. It sets the model structure, so
            it is fixed for the lifetime of the env. Defaults to ``2``.
        impl (str, optional): MJX backend, ``"jax"`` or ``"warp"``. Defaults to
            ``"jax"``.
        config_overrides (dict, optional): flattened overrides of the underlying
            ``ConfigDict`` (``ctrl_dt``, ``sim_dt``, ``episode_length``, ``impl``).
            Defaults to ``None``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> env = PushEnv.push_to_goal(num_cars=3)
        >>> state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        >>> state.obs.shape, env.action_size
        ((28,), 6)
        >>> state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    """

    def __init__(
        self,
        task: PushTaskConfig | None = None,
        num_cars: int = 2,
        impl: str = "jax",
        config_overrides: dict | None = None,
    ):
        if num_cars < 1:
            raise ValueError(f"num_cars must be at least 1, got {num_cars}.")
        task = PushTaskConfig() if task is None else task
        config = config_dict.create(
            ctrl_dt=task.ctrl_dt,
            sim_dt=task.sim_dt,
            episode_length=task.episode_length,
            impl=impl,
        )
        super().__init__(config, config_overrides=config_overrides)
        self.task = task
        self.num_cars = num_cars
        self._mj_model = mujoco.MjModel.from_xml_string(_scene_xml(num_cars, task))
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        # Planar joints: qpos and qvel share the same addresses.

        def adr(name):
            return int(self._mj_model.joint(name).qposadr[0])

        self._car_adr = jnp.asarray(
            [
                [adr(f"car_{i}_{ax}") for ax in ("x", "y", "yaw")]
                for i in range(num_cars)
            ]
        )
        self._box_adr = jnp.asarray([adr(f"box_{ax}") for ax in ("x", "y", "yaw")])

    @classmethod
    def push_to_goal(cls, num_cars: int = 2, **task_overrides) -> PushEnv:
        """Default task: the box and goal are resampled at every reset.

        Args:
            num_cars (int, optional): number of cars. Defaults to ``2``.

        Keyword Args:
            **task_overrides: fields of :class:`PushTaskConfig` to override.
        """
        return cls(replace(PushTaskConfig(), **task_overrides), num_cars=num_cars)

    @property
    def xml_path(self) -> str:
        return ""  # the scene is generated in code

    @property
    def action_size(self) -> int:
        return 2 * self.num_cars

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def agent_mapping(self) -> MujocoPlaygroundAgentMapping:
        """Mapping that gives each car its own action slice and observation view.

        Car ``i`` acts on entries ``[2i, 2i + 1]`` and sees its own egocentric
        block plus the shared block. ``homogenization_mode="max"`` prepends a
        one-hot agent id so a single network can serve every car.
        """
        shared = list(
            range(
                CAR_OBS_SIZE * self.num_cars,
                CAR_OBS_SIZE * self.num_cars + SHARED_OBS_SIZE,
            )
        )
        return MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name=f"agent_{i}",
                    action_indices=[2 * i, 2 * i + 1],
                    observation_indices=list(
                        range(CAR_OBS_SIZE * i, CAR_OBS_SIZE * (i + 1))
                    )
                    + shared,
                )
                for i in range(self.num_cars)
            ],
            homogenization_mode="max",
        )

    def reset(self, rng: jax.Array) -> State:
        cfg = self.task
        k_box, k_dir, k_dist, k_ring, k_off, k_yaw = jax.random.split(rng, 6)
        box_xy = jax.random.uniform(
            k_box, (2,), minval=-cfg.box_start_range, maxval=cfg.box_start_range
        )
        direction = jax.random.uniform(k_dir, (), minval=0.0, maxval=2 * jnp.pi)
        dist = jax.random.uniform(
            k_dist,
            (),
            minval=cfg.goal_distance_range[0],
            maxval=cfg.goal_distance_range[1],
        )
        limit = cfg.arena_half_size - 0.25
        goal_xy = jnp.clip(
            box_xy + dist * jnp.stack([jnp.cos(direction), jnp.sin(direction)]),
            -limit,
            limit,
        )
        radius = jax.random.uniform(
            k_ring, (), minval=cfg.car_spawn_radius[0], maxval=cfg.car_spawn_radius[1]
        )
        offset = jax.random.uniform(k_off, (), minval=0.0, maxval=2 * jnp.pi)
        angles = offset + 2 * jnp.pi * jnp.arange(self.num_cars) / self.num_cars
        car_xy = box_xy + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1)
        car_yaw = jax.random.uniform(
            k_yaw, (self.num_cars,), minval=-jnp.pi, maxval=jnp.pi
        )

        qpos = jnp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._car_adr[:, 0]].set(car_xy[:, 0])
        qpos = qpos.at[self._car_adr[:, 1]].set(car_xy[:, 1])
        qpos = qpos.at[self._car_adr[:, 2]].set(car_yaw)
        qpos = qpos.at[self._box_adr[:2]].set(box_xy)
        data = mjx.make_data(self.mj_model, impl=self._config.impl)
        data = data.replace(
            qpos=qpos,
            mocap_pos=jnp.concatenate([goal_xy, jnp.asarray([0.002])])[None],
        )
        info = {"steps": jnp.zeros(()), "prev_dist": jnp.linalg.norm(goal_xy - box_xy)}
        return State(
            data=data,
            obs=self._obs(data),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics={"dist_to_goal": info["prev_dist"], "success": jnp.zeros(())},
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.task
        action = jnp.clip(action, -1.0, 1.0).reshape(self.num_cars, 2)
        v = action[:, 0] * cfg.max_speed
        omega = action[:, 1] * cfg.max_yaw_rate

        def substep(data, _):
            # Non-holonomic drive: the commanded planar velocity follows the
            # current heading, re-read at every physics step.
            yaw = data.qpos[self._car_adr[:, 2]]
            ctrl = jnp.stack([v * jnp.cos(yaw), v * jnp.sin(yaw), omega], -1)
            return mjx.step(self.mjx_model, data.replace(ctrl=ctrl.reshape(-1))), None

        data, _ = jax.lax.scan(substep, state.data, (), self.n_substeps)

        box_xy = data.qpos[self._box_adr[:2]]
        dist = jnp.linalg.norm(data.mocap_pos[0, :2] - box_xy)
        success = (dist < cfg.success_radius).astype(jnp.float32)
        steps = state.info["steps"] + 1
        if cfg.builtin_reward:
            car_xy = data.qpos[self._car_adr[:, :2]]
            reach = jnp.linalg.norm(car_xy - box_xy, axis=-1).mean()
            reward = (
                cfg.w_progress * (state.info["prev_dist"] - dist)
                - cfg.w_reach * reach
                + cfg.w_success * success
                - cfg.w_action * jnp.square(action).mean()
            )
        else:
            reward = jnp.zeros(())
        return State(
            data=data,
            obs=self._obs(data),
            reward=reward,
            done=(steps >= cfg.episode_length).astype(jnp.float32),
            metrics={"dist_to_goal": dist, "success": success},
            info={"steps": steps, "prev_dist": dist},
        )

    def _obs(self, data: mjx.Data) -> jax.Array:
        """Egocentric block per car, followed by the shared block."""
        car_xy = data.qpos[self._car_adr[:, :2]]
        yaw = data.qpos[self._car_adr[:, 2]]
        box_xy = data.qpos[self._box_adr[:2]]
        box_yaw = data.qpos[self._box_adr[2]]
        goal_xy = data.mocap_pos[0, :2]
        cos, sin = jnp.cos(yaw), jnp.sin(yaw)

        def to_car_frame(world_vec):  # (..., 2) world-frame vectors -> car frame
            x, y = world_vec[..., 0], world_vec[..., 1]
            return jnp.stack([cos * x + sin * y, -sin * x + cos * y], -1)

        box_rel = to_car_frame(box_xy - car_xy)
        goal_rel = to_car_frame(goal_xy - car_xy)
        if self.num_cars > 1:
            diff = car_xy[None, :, :] - car_xy[:, None, :]  # [i, j] = car_j - car_i
            dist = jnp.linalg.norm(diff, axis=-1) + jnp.eye(self.num_cars) * 1e6
            nearest = jnp.argmin(dist, axis=-1)
            other_rel = to_car_frame(diff[jnp.arange(self.num_cars), nearest])
        else:
            other_rel = jnp.zeros((1, 2))
        vel = data.qvel[self._car_adr]  # same dof layout as qpos for planar joints
        forward_speed = cos * vel[:, 0] + sin * vel[:, 1]
        cars = jnp.concatenate(
            [box_rel, goal_rel, other_rel, forward_speed[:, None], vel[:, 2:3]], -1
        )
        shared = jnp.concatenate(
            [goal_xy - box_xy, jnp.stack([jnp.sin(box_yaw), jnp.cos(box_yaw)])]
        )
        return jnp.concatenate([cars.reshape(-1), shared])
