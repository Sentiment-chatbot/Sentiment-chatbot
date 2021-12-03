""" We modified and redistributed the original code of the pytorch class CosineAnnealingWarmRestarts """
""" Original code: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts """

import math

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts

# Learning scheduler based on SGDR(Stochastic Gradient Descent with Warm Restarts)
# Add functions ver. (Warm up start, Gradually decreasing eta_max val)
class ModifiedCosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_up=0, T_mult=1, eta_min=0, eta_max=0.1, gamma=1., last_epoch=-1, verbose=False):
        """
            New things:
                T_up: Taking epochs to starting warm-up
                eta_max: maximum val of learning rate
                gamma: coefficient for gradual decreasing
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_up = T_up
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_cur = last_epoch

        self.base_eta_max = eta_max # learning rate will be decreased gradually
        self.eta_min = eta_min
        self.eta_max = eta_max # In precise, this indicated the 'current eta_max'
        self.gamma = gamma

        self.cycle = 0

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.T_cur == -1: # Initial
            return self.base_lrs
        elif self.T_cur < self.T_up: # Starting Warm up
            return [base_lr + ((self.eta_max - base_lr) * self.T_cur / self.T_up)
                    for base_lr in self.base_lrs]
        else: # After starting
            return [
                base_lr + 
                ((self.eta_max - base_lr) *
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """ Step could be called after every batch update """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
                self.cycle += 1
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.cycle = n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
