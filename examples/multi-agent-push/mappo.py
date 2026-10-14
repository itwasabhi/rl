# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""MAPPO / IPPO on a MuJoCo Playground scene where N cars push a box to a goal.

:class:`push_env.PushEnv` is a plain single-agent MJX env. Three TorchRL pieces
turn it into a cooperative multi-agent task and train it:

- :class:`~torchrl.envs.MujocoPlaygroundWrapper` with the env's
  ``agent_mapping()`` splits the flat action and observation into one group per
  car (``agent_0``, ``agent_1``, ...), all sharing the team reward;
- :class:`~torchrl.envs.Stack` regroups them under a single ``"agents"`` key
  with an agent dimension, the layout the multi-agent modules and losses expect;
- :class:`~torchrl.objectives.multiagent.MAPPOLoss` (centralized critic) or
  :class:`~torchrl.objectives.multiagent.IPPOLoss` (per-agent critic) trained
  from a :class:`~torchrl.collectors.Collector`.

Usage::

    python examples/multi-agent-push/mappo.py --num_cars 2 --frames 1_000_000
    python examples/multi-agent-push/mappo.py --num_cars 3 --algo ippo
"""
from __future__ import annotations

import argparse
from typing import Literal

import torch
from push_env import CAR_OBS_SIZE, PushEnv
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    MujocoPlaygroundWrapper,
    RewardSum,
    Stack,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MultiAgentMLP,
    PopArtValueNorm,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import IPPOLoss, MAPPOLoss


def make_env(
    num_envs: int, num_cars: int, max_steps: int, device: torch.device | str
) -> TransformedEnv:
    """Build the cooperative multi-agent push env with ``num_envs`` vmapped copies."""
    raw = PushEnv.push_to_goal(num_cars=num_cars, episode_length=max_steps)
    base = MujocoPlaygroundWrapper(
        raw, agent_mapping=raw.agent_mapping(), batch_size=[num_envs], device=device
    )
    names = [f"agent_{i}" for i in range(num_cars)]
    # The running episode return is accumulated per car, before stacking:
    # Stack's inverse drops the whole "agents" group from the step input, which
    # would erase a running total stored under it.
    return TransformedEnv(
        base,
        Compose(
            RewardSum(
                in_keys=[(name, "reward") for name in names],
                out_keys=[(name, "episode_reward") for name in names],
            ),
            Stack(
                in_keys=names,
                out_key="agents",
                in_key_inv="agents",
                out_keys_inv=names,
            ),
        ),
    )


def make_actor(env: TransformedEnv) -> ProbabilisticActor:
    """Decentralized actor: one shared network applied to each car's own view."""
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    n_agents = env.observation_spec["agents", "observation"].shape[-2]
    action_spec = env.full_action_spec_unbatched[("agents", "action")]
    backbone = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_spec.shape[-1],
            n_agents=n_agents,
            centralized=False,
            share_params=True,
            depth=2,
            num_cells=128,
            activation_class=nn.Tanh,
            device=env.device,
        ),
        NormalParamExtractor(),
    )
    return ProbabilisticActor(
        module=TensorDictModule(
            backbone,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        ),
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
    )


def make_critic(env: TransformedEnv, *, centralized: bool) -> TensorDictModule:
    """Value network: sees every car's observation if ``centralized`` (MAPPO)."""
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    n_agents = env.observation_spec["agents", "observation"].shape[-2]
    return TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralized=centralized,
            share_params=True,
            depth=2,
            num_cells=128,
            activation_class=nn.Tanh,
            device=env.device,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )


def make_loss(
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    algo: Literal["mappo", "ippo"],
    device: torch.device | str,
) -> MAPPOLoss:
    """MAPPO (centralized critic, PopArt value norm) or IPPO loss on the push keys."""
    mappo = algo == "mappo"
    loss_module = (MAPPOLoss if mappo else IPPOLoss)(
        actor_network=actor,
        critic_network=critic,
        value_norm=PopArtValueNorm(shape=1, device=device) if mappo else None,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
    )
    # The reward is per agent (identical across cars); done/terminated sit at the
    # root and are broadcast over the agent dimension by the value estimator.
    loss_module.set_keys(
        value=("agents", "state_value"),
        action=("agents", "action"),
        reward=("agents", "reward"),
    )
    return loss_module


def evaluate(env: TransformedEnv, actor: ProbabilisticActor, max_steps: int) -> dict:
    """Deterministic rollout; returns the mean team return and final box-goal distance."""
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(max_steps, actor, break_when_any_done=False)
    n_agents = rollout["agents", "observation"].shape[-2]
    final = rollout["next", "agents", "observation"][:, -1, 0]
    # Observations are [agent id | car block | shared block]; the shared block
    # starts with the box-to-goal vector.
    start = n_agents + CAR_OBS_SIZE
    return {
        "return": rollout["next", "agents", "episode_reward"][:, -1].mean().item(),
        "final_box_goal_dist": final[:, start : start + 2].norm(dim=-1).mean().item(),
    }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # One collector batch is one full episode in each vmapped env.
    num_envs = max(1, args.frames_per_batch // args.max_steps)
    frames_per_batch = num_envs * args.max_steps
    env = make_env(num_envs, args.num_cars, args.max_steps, device)
    env.set_seed(args.seed)
    eval_env = make_env(args.eval_envs, args.num_cars, args.max_steps, device)

    actor = make_actor(env)
    critic = make_critic(env, centralized=args.algo == "mappo")
    loss_module = make_loss(actor, critic, args.algo, device)
    collector = Collector(
        env,
        actor,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=args.frames,
        auto_register_policy_transforms=True,
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    total_frames = 0
    for it, data in enumerate(collector):
        with torch.no_grad():
            loss_module.value_estimator(
                data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        replay_buffer.extend(data.reshape(-1))
        total_frames += data.numel()

        for _ in range(args.epochs):
            for _ in range(frames_per_batch // args.minibatch_size):
                losses = loss_module(replay_buffer.sample())
                loss = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                )
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()
        collector.update_policy_weights_()

        done = data["next", "done"].squeeze(-1)
        episode_return = data["next", "agents", "episode_reward"][done].mean().item()
        message = (
            f"[{args.algo}] iter={it:03d} frames={total_frames:>8d} "
            f"train_return={episode_return:+.3f}"
        )
        if args.eval_interval and it % args.eval_interval == 0:
            metrics = evaluate(eval_env, actor, args.max_steps)
            message += (
                f" eval_return={metrics['return']:+.3f}"
                f" final_box_goal_dist={metrics['final_box_goal_dist']:.3f}"
            )
        torchrl_logger.info(message)

    collector.shutdown()
    for e in (env, eval_env):
        if not e.is_closed:
            e.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=("mappo", "ippo"), default="mappo")
    p.add_argument("--num_cars", type=int, default=2)
    p.add_argument("--frames", type=int, default=1_000_000)
    p.add_argument("--frames_per_batch", type=int, default=6_400)
    p.add_argument("--minibatch_size", type=int, default=400)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_envs", type=int, default=16)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
