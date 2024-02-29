import torch
import torch.autograd as autograd


def hard_gradient_penalty(net, real_data, fake_data, device):

    mask = torch.FloatTensor(real_data.shape).to(device).uniform_() > 0.5
    inv_mask = ~mask
    mask, inv_mask = mask.float(), inv_mask.float()

    interpolates = mask * real_data + inv_mask * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    c_interpolates = net(interpolates)

    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gp