import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(math.pi *
                                    (t_cur / restart_period)))


class CyclicRWithRestarts(_LRScheduler):
    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7,
                 eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_steps=0.5):

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' is not specified "
                                   f"in param_group[{i}] when resumin an"
                                   f" optimizer")

        self.base_lrs = [group['initial_lr'] for group
                         in optimizer.param_groups]
        self.min_lrs = [group['minimum_lr'] for group
                        in optimizer.param_groups]

        self.base_weight_decays = [group['weight_decay'] for group
                                   in optimizer.param_groups]

        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb

        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        else:
            raise NotImplemented()

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.iteration = 0
        self.total_iterations = 0

        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1

        self.eta_min = 0
        self.eta_max = 1

        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

    def epoch_step(self):
        if self.verbose:
            logging.debug(f"Epoch step: {self.last_epoch}")
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.step()

    def step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            if self.verbose:
                logging.debug(f"Iteration step: {self.iteration}")
            self.iteration += 1
            self.total_iterations += 1
        except IndexError:
            raise StopIteration("Epoch size and batch size in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min)
                 * self.policy_fn(t_cur, self.restart_period))

        weight_decay_norm_multi = math.sqrt(self.batch_size / (self.epoch_size * self.restart_period))
        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr
               in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]
        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                logging.debug(f"Restart {self.restarts + 1} at epoch {self.last_epoch}")
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False

        return zip(lrs, weight_decays)

    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(
                self.eta_min, self.eta_max, self.total_iterations
            )

    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(
                self.eta_min, self.eta_max
            )
