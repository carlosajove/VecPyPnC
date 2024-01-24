import torch
import numpy as np

t1 = np.concatenate((np.zeros((3, 6)), np.eye(6)))
print(t1)

t2 = torch.cat((torch.zeros((3, 6)), torch.eye(6)), axis = 0)

print(t2)

print("d")
print(t1.shape[0])
print(t2.shape[0])

t3 = np.copy(t1)
print(t3-t1)

t4 = torch.clone(t2).detach()
print(t4)


snp = np.dot(np.eye(9), 5*np.eye(9))[:, 6:]
storch = torch.matmul(torch.eye(9), 5*torch.eye(9))[:, 6:]


print(snp)
print(storch)