from stable_baselines3 import PPO as SB3PPO
import train

import argparse

parser = argparse.ArgumentParser(description="SB3 PPO discrete platformer balancer")
parser.add_argument("--timesteps", type=int, default=600_000)
parser.add_argument("--envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--ent", type=float, default=0.05)
parser.add_argument("--stochastic", action="store_true", help="sample actions at inference for variety")
args = parser.parse_args()
model = SB3PPO.load("ppo_platformer_sb3_vecnorm_v2")
normalizer = train.load_normalizer("vecnormalize_sb3_v2.pkl", seed=args.seed)

# Example predictions (pass prev_time_limit for proper normalization)
tests = [
    ((3.0, 1.0, 160.0), 180.0),
    ((0.0, 0.0, 120.0), 180.0),
    ((0.0, 0.0, 160.0), 200.0),
    ((1.0, 3.0, 210.0), 240.0),
    ((4.0, 0.0, 200.0), 220.0),
    ((7.0, 5.0, 90.0), 120.0),
]
for (inp, prev_lim) in tests:
    out = train.predict_next_level(model, normalizer, inp, prev_time_limit=prev_lim,
                             deterministic=not args.stochastic)
    print(f"Input {inp} (prev_limit={prev_lim}) -> {out}")
