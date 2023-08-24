'''
Fps functions are modified from pointnet2:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
'''

import numpy as np
import torch
from torch.autograd import Function


def farthest_point_sampler(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point = point.detach().cpu().numpy()
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return torch.from_numpy(centroids)
    # point = point[centroids.astype(np.int32)]
    # return point

class FarthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, cnt: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param cnt: int, number of features in the sampled set
        :return:
             output: (B, npoint) index of sampling points
        """
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        B, N, _ = xyz.size()
        output: torch.Tensor = torch.cuda.IntTensor(B, cnt)

        output = [farthest_point_sampler(x, cnt) for x in xyz]
        output = torch.stack(output)
        return output.long().to(xyz.device)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = FarthestPointSampling.apply

if __name__ == '__main__':
    fps = farthest_point_sample
    ps = torch.rand((8,5000,3)).to('cuda')
    result = fps(ps, 2048)
    print(result)
    