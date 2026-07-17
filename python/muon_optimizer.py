"""Kimi-style Muon, adapted from HZY and LK's KataGo_Transformer implementation.

Source: https://github.com/hzyhhzy/KataGo_Transformer/blob/main/train/muon_kissin.py
Copyright (c) 2025 HZY. MIT license: see muon.LICENSE.
"""
import torch


def orthogonalize(matrix):
    x = matrix.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    return x


class Muon(torch.optim.Optimizer):
    """Muon for internal weight matrices; auxiliary Adam for heads and scalars."""
    def __init__(self, groups):
        super().__init__(groups, dict(lr=0.0, weight_decay=0.0))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            lr = group['lr']
            if group['group_name'] in ('normal', 'normal_attn'):
                by_shape = {}
                for p in params:
                    state = self.state[p]
                    if not state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    momentum = state['momentum_buffer']
                    momentum.lerp_(p.grad, 0.05)
                    update = p.grad.lerp_(momentum, 0.95).reshape(p.shape[0], -1)
                    transposed = update.shape[0] > update.shape[1]
                    if transposed:
                        update = update.mT
                    by_shape.setdefault(tuple(update.shape), []).append((p, update, transposed))
                for entries in by_shape.values():
                    for start in range(0, len(entries), 32):
                        chunk = entries[start:start + 32]
                        updates = orthogonalize(torch.stack([entry[1] for entry in chunk]))
                        for (p, _, transposed), update in zip(chunk, updates.unbind()):
                            if transposed:
                                update = update.mT
                            update *= max(p.shape) ** 0.5
                            p.mul_(1 - lr * group['weight_decay'])
                            p.add_(update.reshape_as(p), alpha=-lr)
            else:
                lr /= 8.0
                by_step = {}
                for p in params:
                    state = self.state[p]
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                    state['step'] += 1
                    by_step.setdefault(state['step'], []).append((p, p.grad, state['exp_avg'], state['exp_avg_sq']))
                for step, entries in by_step.items():
                    ps, grads, avgs, squares = map(list, zip(*entries))
                    torch._foreach_lerp_(avgs, grads, 0.05)
                    torch._foreach_mul_(squares, 0.995)
                    torch._foreach_addcmul_(squares, grads, grads, value=0.005)
                    denominators = torch._foreach_sqrt(squares)
                    torch._foreach_div_(denominators, (1 - 0.995 ** step) ** 0.5)
                    torch._foreach_add_(denominators, 1e-6)
                    updates = torch._foreach_div(avgs, denominators)
                    torch._foreach_mul_(ps, 1 - lr * group['weight_decay'])
                    torch._foreach_add_(ps, updates, alpha=-lr / (1 - 0.95 ** step))
