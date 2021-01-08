import torch 

a = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)
c = a * b
c *= c
# c=c*c
print(c)
# c.backward()