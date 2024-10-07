import torch
# OrthgonalNesterov optimizer

class OrthogonalNesterov(torch.optim.Optimizer):
    """
    Some warnings: This optimizer assumes that all parameters passed in are 2D.
    It shouldn't be used for the embedding layer, the final fully connected layer, or {0,1}-D
    parameters; those should be optimized by a standard method (e.g., AdamW).
    To use it with 4D convolutional filters, it works well to flatten their last 3 dimensions.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, zeropower_iters=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, zeropower_iters=zeropower_iters)
        super().__init__(params, defaults)

    # closure is here as a placeholder for compatibility with the PyTorch API
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = zeroth_power_via_newtonschulz5(g, steps=group['zeropower_iters'])
                scale = update.numel()**0.5 / update.norm()
                p.data.add_(update, alpha=-lr * scale)

@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. It turns out
    to be empirically effective to keep increasing the slope of the quintic at zero even beyond the
    point where it no longer converges to one everywhere after repeated application (so long as it
    stays relatively close to 1 across the interval). Our usage of a Newton-Schulz iteration as the
    orthogonalization method traces to Bernstein & Newhouse (2024) https://arxiv.org/abs/2409.20325
    who suggested its use for computing the preconditioners of Shampoo.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

def separate_params(param_groups):
    param_groups_2d     = []
    param_groups_non2d  = []
    total_param_2d_count      = 0
    total_param_non2d_count   = 0

    # Check if param_groups is a list of dicts or list of params
    if (isinstance(param_groups, list) and isinstance(param_groups[0], dict)) \
      or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        # param_groups is a list of dicts
        for group in param_groups:
            params_2d, params_non2d, param_2d_count, param_non2d_count = separate_params(group['params'])
            param_group_2d      = {'params': params_2d}
            param_group_non2d   = {'params': params_non2d}
            # Copy the group dict and replace the 'params' key with the separated params
            for k in group.keys():
                if k != 'params':
                    param_group_2d[k]    = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count    += param_2d_count
            total_param_non2d_count += param_non2d_count

        return param_groups_2d, param_groups_non2d, total_param_2d_count, total_param_non2d_count

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d    = []
        params_non2d = []
        param_group  = param_groups
        # param_group is a list of param tensors
        for param in param_group:
            if param.ndim == 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        breakpoint()

'''
# Note that CombinedOptimizer is not a torch.optim.Optimizer, but a wrapper around multiple optimizers.
# Example usage:
    optimizer = CombinedOptimizer([
        torch.optim.AdamW(self.lm_head.parameters(), lr=learning_rate, betas=betas, weight_decay=0, fused=True),
        OrthogonalNesterov(self.transformer.h.parameters(), lr=0.1*learning_rate, momentum=0.95)
    ])
'''
class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_types, configs):
        # Separate 2D and non-2D parameters.
        # param_groups_2d_non2d: (param_groups_2d, param_groups_non2d).
        # If params is a list of tensors, then each of param_groups_2d and param_groups_non2d 
        # will be a list of tensors.
        # If params is a list of dicts, then each of param_groups_2d and param_groups_non2d
        # will be a list of dicts.
        # If params is a dict, then each of param_groups_2d and param_groups_non2d will 
        # be a list of dicts containing only one dict.
        param_groups_2d, param_groups_non2d, total_param_2d_count, total_param_non2d_count \
            = separate_params(params)
        param_groups_2d_non2d = (param_groups_2d, param_groups_non2d)
        print(f"Total 2D params: {total_param_2d_count}, Total non-2D params: {total_param_non2d_count}")

        assert len(optimizer_types) == len(configs) == 2
        assert optimizer_types[0] == OrthogonalNesterov, "The first optimizer must be OrthogonalNesterov"
        self.optimizers = [ optimizer_types[i](param_groups_2d_non2d[i], **configs[i]) for i in range(2) ]
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]
        # Combine the state dicts of all opt in self.optimizers into a single dict
        self.state = {k: v for opt in self.optimizers for k, v in opt.state.items()}
        # Initially all states are empty. So no point to print their counts.
        # Only use the defaults of the OrthogonalNesterov optimizer
        self.defaults = self.optimizers[0].defaults

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]
