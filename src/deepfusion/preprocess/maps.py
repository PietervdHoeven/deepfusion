from deepfusion.models.autoencoders import AutoEncoder3D
import pandas as pd
import numpy as np
import torch

model = AutoEncoder3D.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=99-step=6000.ckpt")
