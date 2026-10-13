# Multi-agent push: N cars, one box, one goal (MuJoCo Playground + MAPPO)

`push_env.py` defines a custom [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(MJX) scene: `num_cars` car-like robots push a box across a walled arena to a goal.
`mappo.py` wraps it with `MujocoPlaygroundWrapper`, one agent per car, and trains
the team with MAPPO (or IPPO) from a standard `Collector`.

The pieces, from the env up:

| piece | what it does |
|---|---|
| `PushEnv` (`push_env.py`) | a plain single-agent `MjxEnv`; scene generated in Python for any `num_cars`; task knobs in `PushTaskConfig`, preset `PushEnv.push_to_goal()` |
| `PushEnv.agent_mapping()` | `MujocoPlaygroundAgentMapping`: car `i` gets action entries `[2i, 2i+1]` and its own observation block plus the shared block |
| `MujocoPlaygroundWrapper` | splits the flat action/observation into `agent_0`, `agent_1`, ... and copies the team reward to every agent |
| `RewardSum`, `Stack` | per-car episode return, then regroup the cars under `"agents"` with an agent dimension |
| `MAPPOLoss` / `IPPOLoss` | decentralized actor, centralized (MAPPO) or per-agent (IPPO) critic |

## The scene

- **Cars** are non-holonomic (unicycle) robots. A car's action is `(v, omega)` in
  `[-1, 1]^2`, forward speed and yaw rate. The velocity it commands always points
  along its heading, so it cannot drive sideways; it can still be nudged by contacts.
- **Box** slides on the floor with viscous friction and rotates when pushed off-center.
- **Goal** is a green disc, resampled with the box start and the car poses at every reset.
- **Reward** is one team reward: box-to-goal progress, a small reaching term, a bonus
  inside `success_radius`, and an action cost. Set `builtin_reward=False` in
  `PushTaskConfig` to get a zero reward and supply your own with a transform.
- **Observation** is egocentric per car (box, goal and nearest other car in the car
  frame, own speed and yaw rate) plus a shared block (box-to-goal vector, box yaw).
  The per-car block has the same size for any number of cars, so a policy trained
  with `N` cars can be evaluated with another `N`. `homogenization_mode="max"`
  prepends a one-hot agent id so all cars can share one network.

## Setup

```bash
pip install "torchrl[mujoco_playground]"   # jax + the `playground` package
```

## Train

```bash
python examples/multi-agent-push/mappo.py --num_cars 2 --frames 1_000_000
python examples/multi-agent-push/mappo.py --num_cars 3 --algo ippo
```

`--frames_per_batch` sets how many environment steps are collected per iteration;
the script vmaps `frames_per_batch / max_steps` envs so one iteration is one full
episode in each. The logged `eval_return` and `final_box_goal_dist` come from a
deterministic rollout of the actor in a separate batch of envs.

The MJX simulation runs on whatever device JAX finds (a GPU is much faster than CPU).

What to expect: on 4 CPU cores one 6,400-frame iteration takes about 30 seconds. In a
365k-frame CPU run with the defaults the team return rose from about -6.6 to around 0
and the final box-goal distance fell from 0.84 m to roughly 0.55-0.65 m, but the
curve stayed noisy and the box was not yet delivered reliably. Treat the defaults as a
working starting point, not a solved task: expect to need more frames (a GPU makes
this practical) and some tuning of the reward weights in `PushTaskConfig`.

## Caveats

- **Fixed-length episodes.** `MujocoPlaygroundWrapper` resets the whole vmapped batch
  at once (no partial resets), so the env never terminates early: every env finishes
  on the same step, and success only earns a bonus.
- **The time limit is reported as `terminated`.** The wrapper sets `terminated = done`,
  which slightly biases value bootstrapping at the end of an episode.
- **The number of cars is fixed per env instance.** It changes the MJX model, and MJX
  needs static shapes under `jit`; pick it when constructing `PushEnv`.
- `RewardSum` sits before `Stack` on purpose: `Stack`'s inverse drops the whole
  `"agents"` group from the step input, which would erase a running total stored under it.
