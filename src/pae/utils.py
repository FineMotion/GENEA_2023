import torch


def normalize_vector(v):
    bs = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device(f'cuda:{gpu}'))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(bs, 1).expand(bs, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    bs = u.shape[0]
    i = u[:, 1]*v[:, 2] - u[:, 2]*v[:, 1]
    j = u[:, 2]*v[:, 0] - u[:, 0]*v[:, 2]
    k = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]

    out = torch.cat((i.view(bs, 1), j.view(bs, 1), k.view(bs, 1)), 1)
    return out


def rotmat_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, :3]
    y_raw = ortho6d[:, 3:]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)

    matrix = torch.cat((x, y, z), 2)
    return matrix
