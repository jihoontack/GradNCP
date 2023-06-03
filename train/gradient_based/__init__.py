from collections import OrderedDict

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_grad_norm(grads, bs, detach=True):
    grad_norm_list = []
    for grad in grads:
        if grad is None:
            grad_norm = 0
        else:
            if detach:
                grad_norm = torch.norm(
                    grad.data.view(bs, -1), p=2, dim=1, keepdim=True
                )
            else:
                grad_norm = torch.norm(
                    grad.view(bs, -1), p=2, dim=1, keepdim=True
                )
        grad_norm_list.append(grad_norm)
    return torch.norm(torch.cat(grad_norm_list, dim=1), p=2, dim=1)


def inner_adapt(wrapper, task_data, step_size=1e-2, num_steps=3,
                first_order=False, params=None, sample_type='none'):

    loss = 0.
    batch_size = task_data[0].size(0)
    params = wrapper.get_batch_params(params, batch_size)

    """ inner gradient step """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            wrapper.sample(sample_type, task_data, params)

        params, loss = inner_loop_step(
            wrapper,
            params,
            task_data,
            step_size,
            first_order,
        )

    return params, loss


def inner_loop_step(
    wrapper,
    params,
    task_data,
    inner_lr=1e-2,
    first_order=False,
):
    """Performs a single inner loop step."""
    batch_size = len(task_data[0])
    wrapper.decoder.zero_grad()

    with torch.enable_grad():
        loss = wrapper(task_data, params=params)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            params.values(),
            create_graph=not first_order,
            allow_unused=True
        )
        updated_params = OrderedDict()

        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = 0.

            updated_params[name] = param - inner_lr * grad

    return updated_params, loss


def inner_adapt_test_scale(wrapper, task_data, step_size=1e-2, num_steps=3,
                           first_order=False, params=None, sample_type='none',
                           scale_type='loss'):

    loss = 0.
    batch_size = task_data[0].size(0)
    params = wrapper.get_batch_params(params, batch_size)

    """ inner gradient step """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            wrapper.sample(sample_type, task_data, params)

        params, loss = inner_test_gradscale_loop_step(
            wrapper,
            params,
            task_data,
            step_size,
            first_order,
            scale_type,
        )

    return params, loss


def inner_test_gradscale_loop_step(
    wrapper,
    params,
    task_data,
    inner_lr=1e-2,
    first_order=False,
    scale_type='grad',
):
    """Performs a single inner loop step."""
    batch_size = len(task_data[0])

    wrapper.decoder.zero_grad()
    with torch.enable_grad():
        subsample_loss = wrapper(task_data, params=params)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            params.values(),
            create_graph=False,
            allow_unused=True
        )

    wrapper.decoder.zero_grad()
    wrapper.coord_init()
    with torch.enable_grad():
        loss = wrapper(task_data, params=params)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            params.values(),
            create_graph=not first_order,
            allow_unused=True
        )
        updated_params = OrderedDict()

        if scale_type == 'grad':
            subsample_grad_norm = get_grad_norm(subsample_grad, batch_size, detach=True)
            grads_norm = get_grad_norm(grads, batch_size, detach=True)
            grads_scale = subsample_grad_norm / (grads_norm + 1e-16)
        else:
            raise NotImplementedError()

        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = 0.
            else:
                grads_scale_ = grads_scale.view(
                    (batch_size,) + (1,) * (len(grad.shape) - 1)
                ).detach()
            updated_params[name] = param - inner_lr * grads_scale_ * grad

    return updated_params, loss
