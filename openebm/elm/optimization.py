import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _get_value,
                        _fused_doc, _maximize_doc, _default_to_fused_or_foreach)
from typing import List, Optional, Tuple, Union

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
        epoch=0, 
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        self.epoch = epoch
        super().__init__(params, defaults)

    def update_epoch(self, epoch):
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if (g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1


class WarmUpLinearWarmdownLR(_LRScheduler):
    """
    Option 3: NanoChat 风格的 LR 调度
    - warmup: 线性从 0 (或 lr/divider) 增加到 peak_lr
    - constant: 保持 peak_lr
    - warmdown: 线性从 peak_lr 衰减到 final_lr_frac * peak_lr

    参考 base_train.py:
    - warmup_ratio=0.0 (无 warmup)
    - warmdown_ratio=0.5 (后 50% 衰减)
    - final_lr_frac=0.0 (衰减到 0)
    """
    def __init__(self, optimizer, warmup_ratio, warmdown_ratio, final_lr_frac, total_steps,
                 warm_up_finished_func=None, enable_wd_decay=False, resume_warmup_steps=0):
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.warmdown_steps = int(warmdown_ratio * total_steps)
        self.constant_steps = total_steps - self.warmup_steps - self.warmdown_steps
        self.final_lr_frac = final_lr_frac
        self.total_steps = total_steps
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.last_step = 0
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        # Resume warmup: 从 checkpoint 恢复后，在 N 步内将 LR 从 ckpt_lr 线性升到 schedule 值
        self.resume_warmup_steps = resume_warmup_steps
        self.resume_start_step = None  # 在 load_state_dict 中设置
        self.resume_base_lr = None  # ckpt 保存的 LR，作为 resume warmup 起点

        # Option 2 兼容: 动态 Weight Decay
        self.enable_wd_decay = enable_wd_decay
        self.initial_weight_decays = [group.get('weight_decay', 0) for group in optimizer.param_groups]

        self._last_lr = [0.0 for _ in self.highest_lr] if self.warmup_steps > 0 else self.highest_lr.copy()

        print(f"[Option 3] Linear Warmdown LR 调度已启用:")
        print(f"  - Warmup steps: {self.warmup_steps} ({warmup_ratio*100:.1f}%)")
        print(f"  - Constant steps: {self.constant_steps}")
        print(f"  - Warmdown steps: {self.warmdown_steps} ({warmdown_ratio*100:.1f}%)")
        print(f"  - Final LR fraction: {final_lr_frac}")

        super(WarmUpLinearWarmdownLR, self).__init__(optimizer)

    def step(self):
        self.last_step += 1
        super().step()

    def _compute_schedule_lr(self):
        """计算原始 LR schedule 值 (不含 resume warmup 调整)"""
        # Option 2 兼容: 动态更新 weight decay
        if self.enable_wd_decay and self.total_steps > 0:
            wd_multiplier = max(0.0, 1.0 - self.last_step / self.total_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                if self.initial_weight_decays[i] > 0:
                    group['weight_decay'] = self.initial_weight_decays[i] * wd_multiplier

        step = self.last_step

        if step <= self.warmup_steps:
            # Linear warmup: 0 -> peak_lr
            progress = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [lr * progress for lr in self.highest_lr]
        elif step <= self.warmup_steps + self.constant_steps:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            return self.highest_lr.copy()
        else:
            # Linear warmdown: peak_lr -> final_lr_frac * peak_lr
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            warmdown_progress = (step - self.warmup_steps - self.constant_steps) / self.warmdown_steps if self.warmdown_steps > 0 else 1.0
            warmdown_progress = min(1.0, warmdown_progress)
            return [lr * (1.0 - warmdown_progress * (1.0 - self.final_lr_frac)) for lr in self.highest_lr]

    def get_lr(self):
        target_lrs = self._compute_schedule_lr()

        # Resume warmup: 在恢复后的 N 步内，将 LR 从 ckpt_lr 线性插值到 schedule 值
        if self.resume_start_step is not None and self.resume_warmup_steps > 0:
            steps_since_resume = self.last_step - self.resume_start_step
            if steps_since_resume < self.resume_warmup_steps:
                progress = steps_since_resume / self.resume_warmup_steps
                base_lrs = self.resume_base_lr if self.resume_base_lr is not None else [0.0] * len(target_lrs)
                self._last_lr = [base + progress * (target - base)
                                 for base, target in zip(base_lrs, target_lrs)]
                return self._last_lr

        self._last_lr = target_lrs
        return self._last_lr

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'last_step': self.last_step,
            'last_lr': self._last_lr,
            'finished_warming_up': self.finished_warming_up,
            'resume_warmup_steps': self.resume_warmup_steps,
            'resume_start_step': self.resume_start_step,
            'resume_base_lr': self.resume_base_lr,
        }

    def load_state_dict(self, state_dict):
        self.last_step = state_dict['last_step']
        self._last_lr = state_dict['last_lr']
        self.finished_warming_up = state_dict['finished_warming_up']

        # Resume warmup: 若启用，记录恢复点，并保存 ckpt LR 作为 warmup 起点
        if self.resume_warmup_steps > 0:
            self.resume_start_step = self.last_step
            self.resume_base_lr = list(self._last_lr)  # ckpt 保存的 LR
            print(f"[Resume Warmup] 从 step={self.resume_start_step} 开始，"
                  f"在 {self.resume_warmup_steps} 步内将 LR 从 {self.resume_base_lr} 线性升到 schedule 值")
        else:
            # 从 checkpoint 恢复已有字段 (若存在)
            self.resume_start_step = state_dict.get('resume_start_step', None)
            self.resume_base_lr = state_dict.get('resume_base_lr', None)


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_steps, warm_up_base_lr_divider, cosine_scheduler,
                 warm_up_finished_func=None, total_steps=None, enable_wd_decay=False):
        self.warm_up_steps = warm_up_steps
        self.cosine_scheduler = cosine_scheduler
        self.last_step = 0
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        # Option 2: 动态 Weight Decay
        self.total_steps = total_steps
        self.enable_wd_decay = enable_wd_decay
        self.initial_weight_decays = [group.get('weight_decay', 0) for group in optimizer.param_groups]
        if enable_wd_decay:
            print(f"[Option 2] 动态 Weight Decay 已启用: 将从初始值线性衰减到 0")
            print(f"  - 初始 WD 值: {self.initial_weight_decays}")

        self.base_lr_divider = warm_up_base_lr_divider
        if self.base_lr_divider == -1:
            self._last_lr = [0.0 for _ in self.highest_lr]
        else:
            self._last_lr = [lr / self.base_lr_divider for lr in self.highest_lr]

        super(WarmUpCosineAnnealingLR, self).__init__(optimizer)
        
    def step(self):
        self.last_step += 1
        super().step()

    def get_lr(self):
        # Option 2: 动态更新 weight decay (线性衰减到 0)
        if self.enable_wd_decay and self.total_steps and self.total_steps > 0:
            wd_multiplier = max(0.0, 1.0 - self.last_step / self.total_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                if self.initial_weight_decays[i] > 0:
                    group['weight_decay'] = self.initial_weight_decays[i] * wd_multiplier

        if self.warm_up_steps != 0 and self.last_step <= self.warm_up_steps:
            if self.base_lr_divider == -1: # Warm-up from 0 to highest_lr
                warmup_lr = [
                    lr * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            else: # Warm-up from lr / base_lr_divider to highest_lr
                warmup_lr = [
                    (lr / self.base_lr_divider) +
                    (lr - (lr / self.base_lr_divider)) * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            self._last_lr = warmup_lr
            return warmup_lr
        else:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            # Proceed with the cosine scheduler after warm-up
            self.cosine_scheduler.step()
            self._last_lr = self.cosine_scheduler.get_last_lr()
            return self._last_lr
            
    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'warm_up_steps': self.warm_up_steps,
            'last_lr': self._last_lr,
            'last_step': self.last_step,
            'finished_warming_up': self.finished_warming_up,
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.warm_up_steps = state_dict['warm_up_steps']
        self._last_lr = state_dict['last_lr']
        self.last_step = state_dict['last_step']
        self.finished_warming_up = state_dict['finished_warming_up']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state_dict'])
            
class StableAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(StableAdamW, self).__init__(params, defaults)

    def update_epoch(self, epoch):
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Correct bias for moving averages
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute debiased moving averages
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                RMSt = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                eta_t = group['lr'] / max(1, RMSt.mean())  # Adjust learning rate dynamically

                # Update parameters with the dynamically adjusted learning rate
                # Adjusting the formula to fit typical implementation patterns
                p.data.mul_(1 - eta_t * group['weight_decay'])
                p.data.addcdiv_(corrected_exp_avg, RMSt, value=-eta_t)

class StableAdamWUnfused(torch.optim.Optimizer): # gotten from https://gist.github.com/mitchellnw/d42e22a0b9ec02ceaf4f7b4457f51423

    def __init__(self, params, lr=0.002, weight_decay=0.2, betas=(0.9, 0.99), eps=1e-6, clip_thresh=1., custom_scalar=65536):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamWUnfused, self).__init__(params, defaults)

        self.eps=eps
        self.d = clip_thresh

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = None
        self.custom_scaler = custom_scalar

        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using StableAdamWUnfused-v1')
    def update_epoch(self, epoch):
        # self.epoch = epoch
        pass

    def __setstate__(self, state):
        super(StableAdamWUnfused, self).__setstate__(state)
        
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                param_state = self.state[p]

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                    
                # StableAdamW = AdamW + update clipping (https://arxiv.org/abs/1804.04235) applied tensor-wise.
                rms = torch.div(
                    g.pow(2), 
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                ).mean().sqrt().item()
                
                new_lr = lr * (1. / max(1., rms / self.d ))

                theta = theta.mul_(1.0-new_lr*weight_decay).addcdiv_(
                    v, 
                    denominator, 
                    value=-new_lr
                )

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
            
            group['step'] = step + 1