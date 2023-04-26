import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

config = {
    "env_name": "BreakoutNoFrameskip-v4",
    "num_envs": 8,
    "total_timesteps": int(10e6),
    "seed": 661550378,
}


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=8 => 8 environments)
env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"])  # BreakoutNoFrameskip-v4

print("ENV ACTION SPACE: ", env.action_space.n)

# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

testing = True
if testing:
    model_path = "models/9990000.zip"
    model = PPO.load(model_path, env=env)
else:
    model = PPO(policy="CnnPolicy",
                env=env,
                batch_size=256,
                clip_range=0.1,
                ent_coef=0.01,
                gae_lambda=0.9,
                gamma=0.99,
                learning_rate=2.5e-4,
                max_grad_norm=0.5,
                n_epochs=4,
                n_steps=128,
                vf_coef=0.5,
                verbose=1,
                )

TIMESTEPS = 10000
for i in range(1000):
    if testing:
        obs = env.reset()
        while testing:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print("action*****************" + str(action))
            print("obs*****************" + str(obs[0]))
    else:
        model.learn(
            total_timesteps=TIMESTEPS,
        )
        model.save(f"models/{TIMESTEPS * i}")
env.close()