import torch

class Config:
    def __init__(self):
        self.start_timesteps=25e3
        self.max_timesteps=1e6
        self.expl_noise=0.1
        self.batch_size=256
        self.discount=0.99
        self.tau=0.005
        self.policy_noise=0.22
        self.noise_clip=0.5
        self.policy_freq=2
        self.save_model=True
        self.load_model="" # Model load file name, "" doesn't load, "default" uses file_name
