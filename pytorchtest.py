import torch
import numpy as np
from util import util

t1 = np.concatenate((np.zeros((3, 6)), np.eye(6)))
print(t1)

t2 = torch.cat((torch.zeros((3, 6)), torch.eye(6)), axis = 0)

print(t2)

"""
print("d")
print(t1.shape[0])
print(t2.shape[0])
"""
t3 = np.copy(t1)
print(t3-t1)

t4 = torch.clone(t2).detach()
print(t4)


snp = np.dot(np.eye(9), 5*np.eye(9))[:, 6:]
storch = torch.matmul(torch.eye(9), 5*torch.eye(9))[:, 6:]

"""
print(snp)
print(storch)
"""

a = torch.randn(5, 5)

print(a.expand(3, -1, -1), a.expand(3, -1, -1).shape)

b = np.array([1,2,3])
b = torch.from_numpy(b).expand(3, -1)
print(b, b.shape)


rot = torch.randn(5, 3, 3)
rot2 = util.MakeHorizontalDirX(rot)

print("rot 1", rot)
"""
print("rot 2", rot2)
print(rot2[:, 0, :])
print(rot2[:, :, 0])
"""

rot3= util.rotationZ(10*torch.ones(5), rot2)
print("\n \n rot2", rot2)
print("\n rot 3", rot3 )