
import torch

# from torch.utils.cpp_extension import load
# cd = load(name="cd",
#           sources=["models/cd_distance/chamfer_distance.cpp",
#                    "models/cd_distance/chamfer_distance.cu"])

# class ChamferDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         batchsize, n, _ = xyz1.size()
#         _, m, _ = xyz2.size()
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#         dist1 = torch.zeros(batchsize, n)
#         dist2 = torch.zeros(batchsize, m)

#         idx1 = torch.zeros(batchsize, n, dtype=torch.int)
#         idx2 = torch.zeros(batchsize, m, dtype=torch.int)

#         if not xyz1.is_cuda:
#             cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
#         else:
#             dist1 = dist1.cuda()
#             dist2 = dist2.cuda()
#             idx1 = idx1.cuda()
#             idx2 = idx2.cuda()
#             cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

#         return dist1, dist2

#     @staticmethod
#     def backward(ctx, graddist1, graddist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

#         graddist1 = graddist1.contiguous()
#         graddist2 = graddist2.contiguous()

#         gradxyz1 = torch.zeros(xyz1.size())
#         gradxyz2 = torch.zeros(xyz2.size())

#         if not graddist1.is_cuda:
#             cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
#         else:
#             gradxyz1 = gradxyz1.cuda()
#             gradxyz2 = gradxyz2.cuda()
#             cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

#         return gradxyz1, gradxyz2
    
from kaolin.metrics.pointcloud import sided_distance

def chamfer_distance(p1, p2, w1=1., w2=1., squared=True):
    r"""Computes the chamfer distance between two pointclouds, defined as following:

    :math:`\dfrac{w_1}{|P_1|}\sum\limits_{p_{1i} \in P_1}\min\limits_{p_{2j} \in P_2}(||p_{1i} - p_{2j}||_2^2) +
    \dfrac{w_2}{|P_2|}\sum\limits_{p_{2j} \in P_2}\min\limits_{p_{1i} \in P_1}(||p_{2j} - p_{1i}||_2^2)`

    Args:
        p1 (torch.Tensor): Pointclouds, of shape
                           :math:`(\text{batch_size}, \text{num_points1}, 3)`.
        p2 (torch.Tensor): Pointclouds, of shape
                           :math:`(\text{batch_size}, \text{num_points2}, 3)`.
        w1 (float, optional): Weighting of forward direction. Default: 1.
        w2 (float, optional): Weighting of backward direction. Default: 1.
        squared (bool, optional): Use the squared sided distance.
                                  Default: True.

    Returns:
        (torch.Tensor):
            Chamfer distance between two pointclouds p1 and p2,
            of shape :math:`(\text{batch_size})`.
    Example:
        >>> p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
        ...                     [8.5640, 7.7767, 9.4214]],
        ...                    [[0.5431, 6.4495, 11.4914],
        ...                     [3.2126, 8.0865, 3.1018]]], device='cuda', dtype=torch.float)
        >>> p2 = torch.tensor([[[6.9340, 6.1152, 3.4435],
        ...                     [0.1032, 9.8181, 11.3350]],
        ...                    [[11.4006, 2.2154, 7.9589],
        ...                     [4.2586, 1.4133, 7.2606]]], device='cuda', dtype=torch.float)
        >>> chamfer_distance(p1, p2)
        tensor([ 72.5838, 151.0809], device='cuda:0')
    """
    sdist1 = sided_distance(p1, p2)[0]
    sdist2 = sided_distance(p2, p1)[0]

    if not squared:
        sdist1 = torch.sqrt(sdist1)
        sdist2 = torch.sqrt(sdist2)

    # dist_to_p2 = sdist1.mean(dim=-1)
    # dist_to_p1 = sdist2.mean(dim=-1)

    # if (w1 == 1 and w2 == 1):
    #     distance = dist_to_p2 + dist_to_p1
    # else:
    #     distance = w1 * dist_to_p2 + w2 * dist_to_p1

    return sdist1, sdist2

# class ChamferDistance(torch.nn.Module):
#     def forward(self, xyz1, xyz2):
#         return ChamferDistanceFunction.apply(xyz1, xyz2)

if __name__ == '__main__':
    p1 = torch.rand(2, 100, 3).to('cuda')
    p2 = torch.rand(2, 120, 3).to('cuda')
    sdist1, sdist2 = chamfer_distance(p1, p2)
    print(p1.shape, p2.shape)