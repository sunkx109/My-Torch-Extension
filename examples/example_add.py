import time
import torch
from torch import nn
from torch.autograd import Function
import my_torch_ext


class AddModelFunction(Function):
    @staticmethod
    def forward(ctx,a,b,n):
        c = torch.empty(n).cuda()
        my_torch_ext.ops.add(c, a, b, n)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output, None)

class AddModel(nn.Module):
    def __init__(self,a,b,n):
        super(AddModel, self).__init__()
        self.a = a
        self.b = b
        self.n = torch.tensor(data=n,device="cuda:0")
        self.c = torch.empty(n).cuda()
        
    def forward(self):
        self.c = AddModelFunction.apply(self.a, self.b, self.n)
        return self.c


def performace(n,a,b,c):
    # our extension
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        my_torch_ext.ops.add(c,a,b,n)
    torch.cuda.synchronize()
    out_impl_time = (time.time() - start) / 10
    
    # pytorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c = torch.add(a,b)
    torch.cuda.synchronize()
    torch_impl_time = (time.time() - start) /10 
    
    print('Our Implementation: {:.3f} us | Torch Implementation {:.3f} us'.format(out_impl_time * 1e6/1e5, torch_impl_time * 1e6/1e5))


def accuracy(a,b,n):
    model = AddModel(a,b,n)
    our_res = model()
    print("Our Implementation 's res: ",our_res)
    pytorch_res = torch.add(a,b)
    print("Pytorch Implementation 's res: ",pytorch_res)
    if torch.allclose(pytorch_res,our_res):
        print("Test Pass !")
    else:
        print("Test Failed !")


if __name__ == "__main__":
    n = 1024 * 1024
    a = torch.rand(n, device="cuda:0")
    b = torch.rand(n, device="cuda:0")
    cuda_c = torch.rand(n, device="cuda:0")
    
    # performance comparison
    performace(n,a,b,cuda_c)
    
    # functional verification
    accuracy(a,b,n)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
    