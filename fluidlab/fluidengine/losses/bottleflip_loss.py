import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from sklearn.neighbors import KDTree
from fluidlab.fluidengine.simulators import MPMSimulator
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
import matplotlib.pyplot as plt
from .shapematching_loss import ShapeMatchingLoss

@ti.data_oriented
class BottleFlipLoss(ShapeMatchingLoss):
    def __init__(self, type, **kwargs):
        super(BottleFlipLoss, self).__init__(
            matching_mat=MILK,
            temporal_range_type='all',
            **kwargs
        )

    def build(self, sim):
        super(BottleFlipLoss, self).build(sim)

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 0.025 * (121.3 - cur_step_loss)
        loss = 0.025 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info

    def get_final_loss(self):
        self.compute_total_loss_kernel(self.temporal_range[0], self.temporal_range[1])
        self.expand_temporal_range()
        loss_info = {
            'loss': self.total_loss[None],
            'last_step_loss': self.step_loss[self.max_loss_steps-1],
            'temporal_range': self.temporal_range[1],
            'reward': np.sum((121.3 - self.step_loss.to_numpy()) * 0.025)
        }

        return loss_info