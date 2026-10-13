# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, TensorDict

from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.batched_envs import ParallelEnv
from torchrl.envs.libs.mujoco_playground import (
    _has_mujoco_playground,
    KNOWN_MARL_MAPPINGS,
    MujocoPlaygroundAgentMapping,
    MujocoPlaygroundAgentSpec,
    MujocoPlaygroundEnv,
    MujocoPlaygroundWrapper,
)
from torchrl.envs.transforms import FrameSkipTransform
from torchrl.envs.utils import check_env_specs
from torchrl.testing import get_available_devices

_has_jax = importlib.util.find_spec("jax") is not None

# The multi-agent push example lives next to its training script, which imports
# its sibling ``push_env`` module, so its folder has to be importable.
if _has_mujoco_playground:
    sys.path.insert(
        0,
        str(
            pathlib.Path(__file__).resolve().parents[2]
            / "examples"
            / "multi-agent-push"
        ),
    )
    import jax
    import jax.numpy as jnp
    import mappo as push_mappo
    import push_env

# Default config: force JAX backend on all envs (CPU-only machines lack other backends)
_JAX_CONFIG = {"impl": "jax"}

# Fast flat-obs environment used for most tests
_FLAT_OBS_ENV = "CartpoleBalance"

# Multi-agent tests use CheetahRun (action_size=6, flat obs)
_MARL_ENV = "CheetahRun"

# dm_control_suite environments known to exist (subset)
_KNOWN_DM_ENVS = ["CartpoleBalance", "CheetahRun", "WalkerWalk"]

# Locomotion environments known to exist (subset)
_KNOWN_LOCOMOTION_ENVS = ["Go1JoystickFlatTerrain"]

# Manipulation environments known to exist (subset)
_KNOWN_MANIPULATION_ENVS = ["PandaPickCube"]

# Dict-obs environment used to exercise the dict-observation code path.
_DICT_OBS_ENV = "Go1JoystickFlatTerrain"


# Module-level autouse fixture: every test in this module needs JAX to be set
# up before any wrapper is instantiated. Previously this was duplicated as a
# class-level fixture on each test class.
@pytest.fixture(autouse=True)
def _setup_jax():
    if not _has_jax:
        yield
        return
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    yield


# Cached environment dims for the multi-agent test env, avoiding O(n_tests)
# construction of a throwaway env just to read `action_size` /
# `observation_size`. The env build is expensive (jit tracing), so we pay it
# once per test session.
@pytest.fixture(scope="session")
def marl_env_sizes():
    if not _has_mujoco_playground:
        pytest.skip("mujoco_playground not installed")
    env = MujocoPlaygroundEnv(_MARL_ENV, config_overrides=_JAX_CONFIG)
    try:
        return {
            "action_size": env._env.action_size,
            "observation_size": env._env.observation_size,
        }
    finally:
        env.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("envname", [_FLAT_OBS_ENV])
class TestMujocoPlayground:
    def test_constructor_wrapper(self, envname, device):
        """MujocoPlaygroundWrapper works from a pre-built env."""
        from mujoco_playground import dm_control_suite

        base_env = dm_control_suite.load(envname, config_overrides=_JAX_CONFIG)
        env = MujocoPlaygroundWrapper(base_env, device=device)
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()

    def test_constructor_env(self, envname, device):
        """MujocoPlaygroundEnv wraps env by name."""
        env0 = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        from mujoco_playground import dm_control_suite

        base_env = dm_control_suite.load(envname, config_overrides=_JAX_CONFIG)
        env1 = MujocoPlaygroundWrapper(base_env, device=device)

        env0.set_seed(0)
        torch.manual_seed(0)
        r0 = env0.rollout(5)
        env1.set_seed(0)
        torch.manual_seed(0)
        r1 = env1.rollout(5)
        assert_allclose_td(r0.data, r1.data)
        env0.close()
        env1.close()

    def test_seeding(self, envname, device):
        """Same seed produces identical rollouts."""
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = MujocoPlaygroundEnv(
                envname, device=device, config_overrides=_JAX_CONFIG
            )
            torch.manual_seed(0)
            np.random.seed(0)
            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=20))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_batch_size(self, envname, batch_size, device):
        """batch_size propagates to reset/rollout output shapes."""
        env = MujocoPlaygroundEnv(
            envname, batch_size=batch_size, device=device, config_overrides=_JAX_CONFIG
        )
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=10)
        env.close()
        del env
        assert tdreset.batch_size == torch.Size(batch_size)
        assert tdrollout.batch_size[:-1] == torch.Size(batch_size)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_spec_rollout(self, envname, batch_size, device):
        """check_env_specs passes for all batch sizes."""
        env = MujocoPlaygroundEnv(
            envname, batch_size=batch_size, device=device, config_overrides=_JAX_CONFIG
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_parallel(
        self, envname, batch_size, parallel, maybe_fork_ParallelEnv, device, n=1
    ):
        """Env works inside SerialEnv and ParallelEnv."""

        def make_env():
            env = MujocoPlaygroundEnv(
                envname,
                batch_size=batch_size,
                device=device,
                config_overrides=_JAX_CONFIG,
            )
            env.set_seed(1)
            return env

        if parallel:
            env = maybe_fork_ParallelEnv(n, make_env)
        else:
            env = SerialEnv(n, make_env)
        check_env_specs(env)
        tensordict = env.rollout(3)
        assert tensordict.shape == torch.Size([n, *batch_size, 3])

    def test_num_workers_returns_parallel_env(self, envname, device):
        """num_workers > 1 uses worker-originated metadata."""
        env = MujocoPlaygroundEnv(
            envname, num_workers=3, device=device, config_overrides=_JAX_CONFIG
        )
        try:
            assert isinstance(env, ParallelEnv)
            assert env.num_workers == 3
            assert env._metadata_from_workers
            assert env._use_buffers == (not env.meta_data.has_dynamic_specs)
            assert not env.is_closed
            env.reset()
        finally:
            env.close()

    def test_set_seed_and_reset_works(self, envname, device):
        """Setting seed then reset produces a valid TensorDict."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        try:
            final_seed = env.set_seed(0)
            assert final_seed is not None
            td = env.reset()
            assert isinstance(td, TensorDict)
        finally:
            env.close()

    def test_flat_obs_has_observation_key(self, envname, device):
        """Flat-obs environments expose an 'observation' key."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        env.set_seed(0)
        assert "observation" in env.observation_spec.keys()
        td = env.reset()
        assert "observation" in td.keys()
        env.close()

    def test_config_override(self, envname, device):
        """config_overrides are forwarded to the environment constructor."""
        env = MujocoPlaygroundEnv(
            envname,
            config_overrides={"impl": "jax", "episode_length": 200},
            device=device,
        )
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundAvailableEnvs:
    def test_available_envs_contains_all_suites(self, device):
        """available_envs spans dm_control_suite, locomotion, and manipulation."""
        envs = MujocoPlaygroundEnv.available_envs
        assert len(envs) > 0
        for name in _KNOWN_DM_ENVS:
            assert name in envs, f"{name} missing from available_envs"
        for name in _KNOWN_LOCOMOTION_ENVS:
            assert name in envs, f"{name} missing from available_envs"
        for name in _KNOWN_MANIPULATION_ENVS:
            assert name in envs, f"{name} missing from available_envs"

    @pytest.mark.parametrize("envname", _KNOWN_DM_ENVS[:1])
    def test_dm_control_suite_env(self, envname, device):
        """dm_control_suite env constructs and resets."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()

    def test_env_name_property(self, device):
        """env_name property returns correct name."""
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        assert env.env_name == _FLAT_OBS_ENV
        env.close()

    def test_repr(self, device):
        """__repr__ includes env name, batch_size, device."""
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        r = repr(env)
        assert _FLAT_OBS_ENV in r
        assert "batch_size" in r
        env.close()


def _two_agent_mapping_from_sizes(
    obs_size, act_size, homogenization_mode="none"
) -> MujocoPlaygroundAgentMapping:
    """Build a 2-agent mapping that splits obs/action in half."""
    half_act = act_size // 2
    half_obs = obs_size // 2
    return MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                name="agent_0",
                action_indices=list(range(half_act)),
                observation_indices=list(range(half_obs)),
            ),
            MujocoPlaygroundAgentSpec(
                name="agent_1",
                action_indices=list(range(half_act, act_size)),
                observation_indices=list(range(half_obs, obs_size)),
            ),
        ],
        homogenization_mode=homogenization_mode,
    )


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundMultiAgent:
    def _make_env(self, device, marl_env_sizes, mapping=None, batch_size=()):
        if mapping is None:
            mapping = _two_agent_mapping_from_sizes(
                marl_env_sizes["observation_size"],
                marl_env_sizes["action_size"],
            )
        return MujocoPlaygroundEnv(
            _MARL_ENV,
            batch_size=batch_size,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )

    def test_construction_two_agents(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        env.close()

    def test_action_spec_is_nested_composite(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        from torchrl.data.tensor_specs import Composite

        assert isinstance(env.action_spec, Composite)
        assert "agent_0" in env.action_spec.keys()
        assert "agent_1" in env.action_spec.keys()
        env.close()

    def test_reset_output_keys(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        td = env.reset()
        assert "agent_0" in td.keys()
        assert "agent_1" in td.keys()
        assert "done" in td.keys()
        assert "terminated" in td.keys()
        assert "observation" in td["agent_0"].keys()
        assert "observation" in td["agent_1"].keys()
        env.close()

    def test_reset_obs_shapes_none_mode(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        act_size = marl_env_sizes["action_size"]
        half_obs = obs_size // 2
        mapping = _two_agent_mapping_from_sizes(obs_size, act_size)
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()
        assert td["agent_0", "observation"].shape[-1] == half_obs
        assert td["agent_1", "observation"].shape[-1] == obs_size - half_obs
        env.close()

    def test_state_not_in_tensordict(self, device, marl_env_sizes):
        # JAX state is kept as instance var; it must not appear in TensorDict output
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        td = env.reset()
        assert "state" not in td.keys()
        assert "state" not in td["agent_0"].keys()
        assert "state" not in td["agent_1"].keys()
        env.close()

    def test_rollout_works(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        td = env.rollout(3)
        assert td.shape[-1] == 3
        env.close()

    def test_reward_duplicated_to_all_agents(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        td = env.rollout(1)
        r0 = td[..., 0]["next", "agent_0", "reward"]
        r1 = td[..., 0]["next", "agent_1", "reward"]
        torch.testing.assert_close(r0, r1)
        env.close()

    def test_action_shapes_in_rollout(self, device, marl_env_sizes):
        act_size = marl_env_sizes["action_size"]
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        td = env.rollout(2)
        half = act_size // 2
        assert td["agent_0", "action"].shape[-1] == half
        assert td["agent_1", "action"].shape[-1] == act_size - half
        env.close()

    def test_validation_overlapping_actions_raises(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1, 2],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3, 4, 5],  # index 2 overlaps
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="overlap"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_validation_missing_action_indices_raises(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],  # only 2 of 6 — missing 2,3,4,5
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3],  # still missing 4,5
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="Missing indices"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_validation_out_of_range_action_index_raises(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        act_size = marl_env_sizes["action_size"]
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)) + [999],
                    observation_indices=list(range(obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="out of range"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_homogenization_max_shapes(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        # Unequal action splits: 2 vs 4
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3, 4, 5],
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="max",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()

        max_act = 4  # max(2, 4)
        max_obs = max(obs_size // 2, obs_size - obs_size // 2) + 2  # +2 agents one-hot
        assert td["agent_0", "observation"].shape[-1] == max_obs
        assert td["agent_1", "observation"].shape[-1] == max_obs
        assert env.action_spec["agent_0", "action"].shape[-1] == max_act
        assert env.action_spec["agent_1", "action"].shape[-1] == max_act
        env.close()

    def test_homogenization_concat_shapes(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        act_size = marl_env_sizes["action_size"]
        mapping = _two_agent_mapping_from_sizes(
            obs_size, act_size, homogenization_mode="concat"
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()

        assert td["agent_0", "observation"].shape[-1] == obs_size
        assert td["agent_1", "observation"].shape[-1] == obs_size
        assert env.action_spec["agent_0", "action"].shape[-1] == act_size
        assert env.action_spec["agent_1", "action"].shape[-1] == act_size
        env.close()

    def test_check_env_specs_none_mode(self, device, marl_env_sizes):
        env = self._make_env(device, marl_env_sizes)
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_check_env_specs_max_mode(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        act_size = marl_env_sizes["action_size"]
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(2, act_size)),
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="max",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_check_env_specs_concat_mode(self, device, marl_env_sizes):
        obs_size = marl_env_sizes["observation_size"]
        act_size = marl_env_sizes["action_size"]
        mapping = _two_agent_mapping_from_sizes(
            obs_size, act_size, homogenization_mode="concat"
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    @pytest.mark.parametrize("batch_size", [(4,), (4, 3)])
    def test_batch_size_with_agent_mapping(self, device, marl_env_sizes, batch_size):
        env = self._make_env(device, marl_env_sizes, batch_size=batch_size)
        env.set_seed(0)
        td = env.reset()
        assert td.batch_size == torch.Size(batch_size)
        assert td["agent_0", "observation"].shape[: len(batch_size)] == torch.Size(
            batch_size
        )
        assert td["agent_1", "observation"].shape[: len(batch_size)] == torch.Size(
            batch_size
        )
        env.close()

    def test_no_mapping_regression(self, device, marl_env_sizes):
        """Without `agent_mapping`, the env exposes the single-agent layout
        (flat ``observation``), and with the same env+seed the multi-agent
        variant exposes the per-agent layout."""
        env_plain = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        env_marl = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            config_overrides=_JAX_CONFIG,
            agent_mapping=_two_agent_mapping_from_sizes(
                marl_env_sizes["observation_size"],
                marl_env_sizes["action_size"],
            ),
        )
        try:
            env_plain.set_seed(42)
            env_marl.set_seed(42)
            td_plain = env_plain.rollout(5)
            td_marl = env_marl.rollout(5)
            assert "observation" in td_plain.keys()
            assert "agent_0" not in td_plain.keys()
            assert "agent_0" in td_marl.keys()
            assert "agent_1" in td_marl.keys()
            assert "observation" not in td_marl.keys()
        finally:
            env_marl.close()
            env_plain.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
class TestKnownMarlMappings:
    """Tests for KNOWN_MARL_MAPPINGS and string-based agent_mapping lookup."""

    def test_known_mappings_present(self):
        expected = {
            "ant_4x2",
            "halfcheetah_6x1",
            "hopper_3x1",
            "humanoid_9|8",
            "walker2d_2x3",
        }
        assert set(KNOWN_MARL_MAPPINGS.keys()) == expected

    def test_known_mappings_are_agent_mapping_instances(self):
        for name, mapping in KNOWN_MARL_MAPPINGS.items():
            assert isinstance(mapping, MujocoPlaygroundAgentMapping), name
            assert len(mapping.agents) > 0, name
            for agent in mapping.agents:
                assert isinstance(agent, MujocoPlaygroundAgentSpec), name
                assert len(agent.action_indices) > 0, name
                assert len(agent.observation_indices) > 0, name

    def test_string_mapping_resolves(self):
        # CheetahRun has action_size=6 — matches halfcheetah_6x1
        with pytest.warns(UserWarning, match="MABrax"):
            env = MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="halfcheetah_6x1",
                config_overrides=_JAX_CONFIG,
            )
        env.set_seed(0)
        td = env.reset()
        for i in range(6):
            assert f"agent_{i}" in td.keys(), f"agent_{i} missing from reset td"
        env.close()

    def test_string_mapping_check_env_specs(self):
        with pytest.warns(UserWarning, match="MABrax"):
            env = MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="halfcheetah_6x1",
                config_overrides=_JAX_CONFIG,
            )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown agent_mapping"):
            MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="bogus_mapping",
                config_overrides=_JAX_CONFIG,
            )

    def test_mismatch_env_raises(self):
        # ant_4x2 expects action_size=8; CheetahRun has action_size=6 — mismatch
        with pytest.warns(UserWarning, match="MABrax"), pytest.raises(ValueError):
            MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="ant_4x2",
                config_overrides=_JAX_CONFIG,
            )

    def test_object_mapping_does_not_warn(self):
        """Passing a mapping object directly should not trigger the MABrax warning."""
        import warnings

        mapping = KNOWN_MARL_MAPPINGS["halfcheetah_6x1"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            env = MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )
        env.set_seed(0)
        td = env.reset()
        assert "agent_0" in td.keys()
        env.close()

    def test_string_mapping_returns_deep_copy(self):
        """Mutating the resolved mapping must not leak into KNOWN_MARL_MAPPINGS."""
        original_n_agents = len(KNOWN_MARL_MAPPINGS["halfcheetah_6x1"].agents)
        with pytest.warns(UserWarning, match="MABrax"):
            env = MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="halfcheetah_6x1",
                config_overrides=_JAX_CONFIG,
            )
        try:
            # `_agent_mapping` is frozen but `.agents` is a list and is
            # mutable; verify the wrapper's copy is not the module-level one.
            assert (
                env._agent_mapping.agents
                is not KNOWN_MARL_MAPPINGS["halfcheetah_6x1"].agents
            )
            env._agent_mapping.agents.clear()
            assert (
                len(KNOWN_MARL_MAPPINGS["halfcheetah_6x1"].agents) == original_n_agents
            )
        finally:
            env.close()


@pytest.mark.skipif(
    not _has_mujoco_playground,
    reason="mujoco_playground not installed",
)
class TestMujocoPlaygroundDictObs:
    """Tests for dict-observation environments (e.g. locomotion suite)."""

    def _is_dict_obs(self):
        from mujoco_playground import registry

        env = registry.load(_DICT_OBS_ENV, config_overrides=_JAX_CONFIG)
        return isinstance(env.observation_size, dict)

    def test_dict_obs_env_reset(self):
        env = MujocoPlaygroundEnv(
            _DICT_OBS_ENV, device="cpu", config_overrides=_JAX_CONFIG
        )
        if not isinstance(env._env.observation_size, dict):
            env.close()
            pytest.skip(f"{_DICT_OBS_ENV} unexpectedly exposes a non-dict observation.")
        env.set_seed(0)
        td = env.reset()
        # Each observation key from the env should appear at the top level of
        # the reset TensorDict.
        for key in env._env.observation_size:
            assert key in td.keys(), f"obs key '{key}' missing from reset td"
        env.close()

    def test_dict_obs_check_env_specs(self):
        env = MujocoPlaygroundEnv(
            _DICT_OBS_ENV, device="cpu", config_overrides=_JAX_CONFIG
        )
        if not isinstance(env._env.observation_size, dict):
            env.close()
            pytest.skip(f"{_DICT_OBS_ENV} unexpectedly exposes a non-dict observation.")
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_dict_obs_max_mode_raises_not_implemented(self):
        """`homogenization_mode != 'none'` is not yet supported for dict obs."""
        env = MujocoPlaygroundEnv(
            _DICT_OBS_ENV, device="cpu", config_overrides=_JAX_CONFIG
        )
        if not isinstance(env._env.observation_size, dict):
            env.close()
            pytest.skip(f"{_DICT_OBS_ENV} unexpectedly exposes a non-dict observation.")
        # Use the first obs key to build a minimal dict-obs mapping covering
        # the action space.
        obs_sizes = env._env.observation_size
        act_size = env._env.action_size
        env.close()

        first_key = next(iter(obs_sizes))
        key_len = (
            int(obs_sizes[first_key][0])
            if hasattr(obs_sizes[first_key], "__len__")
            else int(obs_sizes[first_key])
        )
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)),
                    observation_indices={first_key: list(range(key_len // 2))},
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(act_size // 2, act_size)),
                    observation_indices={first_key: list(range(key_len // 2, key_len))},
                ),
            ],
            homogenization_mode="max",
        )
        with pytest.raises(NotImplementedError, match="dict-observation"):
            MujocoPlaygroundEnv(
                _DICT_OBS_ENV,
                device="cpu",
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundFrameSkip:
    """frame_skip is not implemented natively; the EnvBase metaclass should
    auto-append a FrameSkipTransform (see ``EnvBase._has_frame_skip``)."""

    def test_no_frame_skip_is_plain_env(self, device):
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        try:
            assert not isinstance(env, TransformedEnv)
        finally:
            env.close()

    @pytest.mark.parametrize("skip", [2, 3])
    def test_frame_skip_auto_transform(self, device, skip):
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV,
            device=device,
            frame_skip=skip,
            config_overrides=_JAX_CONFIG,
        )
        try:
            assert isinstance(env, TransformedEnv)
            assert isinstance(env.transform, FrameSkipTransform)
            assert env.transform.frame_skip == skip
            env.set_seed(0)
            check_env_specs(env)
            td = env.rollout(3)
            assert td.shape[-1] == 3
        finally:
            env.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundPush:
    """The multi-agent push example (``examples/multi-agent-push``)."""

    @staticmethod
    def _joint(env, name, what="qposadr"):
        return int(getattr(env.mj_model.joint(name), what)[0])

    @pytest.mark.parametrize("num_cars", [2, 3])
    def test_stacked_env_specs_and_synchronous_done(self, device, num_cars):
        max_steps = 6
        env = push_mappo.make_env(3, num_cars, max_steps, device)
        try:
            env.set_seed(0)
            check_env_specs(env)
            rollout = env.rollout(max_steps)
            # one egocentric block + shared block + the agent id of "max" mode
            obs_dim = push_env.CAR_OBS_SIZE + push_env.SHARED_OBS_SIZE + num_cars
            assert rollout["agents", "observation"].shape == (
                3,
                max_steps,
                num_cars,
                obs_dim,
            )
            assert rollout["agents", "action"].shape == (3, max_steps, num_cars, 2)
            # No partial resets in the wrapper: every vmapped env must finish on
            # the same step, and not before.
            done = rollout["next", "done"].squeeze(-1)
            assert done[:, -1].all()
            assert not done[:, :-1].any()
            # The running episode return keeps accumulating across the episode.
            torch.testing.assert_close(
                rollout["next", "agents", "episode_reward"][:, -1],
                rollout["next", "agents", "reward"].sum(1),
            )
        finally:
            env.close()

    def test_cars_push_box_to_goal(self, device):
        env = push_env.PushEnv.push_to_goal(num_cars=2)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        qpos = jnp.zeros(env.mjx_model.nq)
        for name, value in {
            "box_x": 0.0,
            "box_y": 0.0,
            "car_0_x": -0.5,
            "car_0_y": 0.05,
            "car_1_x": -0.5,
            "car_1_y": -0.05,
        }.items():
            qpos = qpos.at[self._joint(env, name)].set(value)
        # Goal 1 m ahead of the box along +x; both cars face +x behind the box.
        data = state.data.replace(qpos=qpos, mocap_pos=jnp.asarray([[1.0, 0.0, 0.002]]))
        state = state.replace(
            data=data, info={"steps": jnp.zeros(()), "prev_dist": jnp.ones(())}
        )
        step = jax.jit(env.step)
        forward = jnp.asarray([1.0, 0.0, 1.0, 0.0])
        total_reward = 0.0
        for _ in range(30):
            state = step(state, forward)
            total_reward += float(state.reward)
        box_x = float(state.data.qpos[self._joint(env, "box_x")])
        # 30 steps at 1 m/s would carry a free car 1.5 m, the box starts 0.5 m
        # ahead: it only moves if the cars really push it.
        assert box_x > 0.5
        assert float(state.metrics["dist_to_goal"]) < 0.6
        assert total_reward > 0

    def test_car_cannot_slide_sideways(self, device):
        env = push_env.PushEnv.push_to_goal(num_cars=1)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        for _ in range(20):
            state = step(state, jnp.asarray([1.0, 0.5]))
        qvel = state.data.qvel
        vx = float(qvel[self._joint(env, "car_0_x", "dofadr")])
        vy = float(qvel[self._joint(env, "car_0_y", "dofadr")])
        yaw = float(state.data.qpos[self._joint(env, "car_0_yaw")])
        forward_speed = vx * np.cos(yaw) + vy * np.sin(yaw)
        lateral_speed = -vx * np.sin(yaw) + vy * np.cos(yaw)
        assert forward_speed == pytest.approx(env.task.max_speed, rel=0.1)
        assert abs(lateral_speed) < 0.1 * forward_speed

    @pytest.mark.parametrize("algo", ["mappo", "ippo"])
    def test_loss_gradients_on_collected_data(self, device, algo):
        env = push_mappo.make_env(2, 2, 8, device)
        try:
            env.set_seed(0)
            actor = push_mappo.make_actor(env)
            critic = push_mappo.make_critic(env, centralized=algo == "mappo")
            loss_module = push_mappo.make_loss(actor, critic, algo, device)
            with torch.no_grad():
                data = env.rollout(8, actor)
                loss_module.value_estimator(
                    data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )
            losses = loss_module(data.reshape(-1))
            sum(v for k, v in losses.items() if k.startswith("loss_")).backward()
            grads = [p.grad for p in loss_module.parameters() if p.grad is not None]
            assert grads
            assert all(torch.isfinite(g).all() for g in grads)
        finally:
            env.close()

    @pytest.mark.parametrize("algo", ["mappo", "ippo"])
    def test_training_script_smoke(self, device, algo):
        # Two short iterations of the example's own loop, evaluation included.
        push_mappo.main(
            argparse.Namespace(
                algo=algo,
                num_cars=2,
                frames=32,
                frames_per_batch=16,
                minibatch_size=8,
                max_steps=8,
                epochs=1,
                lr=3e-4,
                eval_envs=2,
                eval_interval=1,
                seed=0,
            )
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
