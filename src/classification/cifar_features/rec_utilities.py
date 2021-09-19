import torch
import sys

sys.path.extend(["./"])


class PoisonBatch(torch.nn.Module):
    def __init__(self, point):
        super(PoisonBatch, self).__init__()
        self.poison = torch.nn.Parameter(point.unsqueeze(0).clone())

    def forward(self):
        return self.poison


def rc_loss(model, x, fx_star):
    fx = model.get_features(x)
    return (fx - fx_star).norm(p=2) ** 2


def inv_transform(x):
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            )
        ]
    )
    return inv_transform(x)


import torchvision.transforms as T


def reconstruct(net, x_i, fx_p, lr=0.1, epochs=1000, device="cuda:0"):
    fx_star = fx_p.to(device)
    poison_batch = PoisonBatch(x_i).to(device)
    x_opt = x_i.to(device)

    params = poison_batch.parameters()  # parameters to optimize
    optimizer = torch.optim.Adam(params, lr=lr)
    net.to(device)
    net.eval()

    min_rec_error = float("inf")
    for i in range(epochs):
        poison_batch.zero_grad()
        x = poison_batch.poison

        reconstruction_error = rc_loss(net, x, fx_star)

        if i % 100 == 0:
            print("reconstruction error: ", reconstruction_error)
        if reconstruction_error < min_rec_error:
            min_rec_error = reconstruction_error
            x_opt = poison_batch.poison.detach().clone()
        # Do backward pass
        reconstruction_error.backward(retain_graph=True)
        # Clip data to input constraints
        poison_batch.poison.data = poison_batch.poison.data
        # Update data
        optimizer.step()
    x_p = inv_transform(x_opt.cpu().squeeze(0)).clamp(0, 1)
    return x_p, x_opt, min_rec_error
