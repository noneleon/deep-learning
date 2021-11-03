import torch


# 类型推断

a = torch.randn(2,3)

a.type()

type(a)

isinstance(a,torch.FloatTensor)

isinstance(data,torch.cuda.DoubleTensor)

data = data.cuda()

isinstance(data,torch.cuda.DoubleTensor)


# 标量

torch.tensor(1.)

torch.tensor(1.3)

# shape

a = torch.tensor(2,2)

a.shape

torch.Size([])

len(a.shape)

a.size()

# tensor

torch.tensor([1.1])

torch.tensor([1.1,1.2])
torch.tensor([[[1.1,1.2]],[[1.1,1.2]]])
torch.FloatTensor(1)



