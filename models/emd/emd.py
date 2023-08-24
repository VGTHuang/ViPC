'''
Use Sinkhorn distance to approximate EMD:
http://www.kernel-operations.io/geomloss/index.html
'''


import torch
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

sinkhorn_dist = SamplesLoss(loss="sinkhorn", backend="tensorized", p=2, blur=.05)

def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    costs = []
    for x1, x2 in zip(xyz1, xyz2):
        costs.append(sinkhorn_dist(x1, x2))
    print(costs)
    cost = sum(costs) / len(costs)
    return cost


# # Create some large point clouds in 3D
# x = torch.randn(4,3,10000, requires_grad=True).cuda()
# y = torch.randn(4,3,10000).cuda()

# # Define a Sinkhorn (~Wasserstein) loss between sampled measures

# L = earth_mover_distance(x, y)
# g_x, = torch.autograd.grad(L, [x])  # GeomLoss fully supports autograd!
# print(L, g_x)