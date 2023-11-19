# https://discuss.pytorch.org/t/gpu-speed-and-memory-difference-between-einsum-and-matmul/164493
import torch
import time

bs = 8
L = 4096
dim = 2048

tensor1 = torch.randn((bs, L,   L)).to('cuda')
tensor2 = torch.randn((bs, dim, L)).to('cuda')

# warmup the GPU
for _ in range(5):
    warump_tensor = torch.matmul(tensor1, tensor1.transpose(1, 2))

def test_einsum(tensor1, tensor2, n=100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):    
        output1 = torch.einsum("bdl,blx->bdx", tensor2, tensor1)

    torch.cuda.synchronize()
    end = time.time()
    print('einsum time:', end-start)
    print('einsum memory (GB):', torch.cuda.max_memory_allocated('cuda')/10**9)
    return output1

def test_matmul(tensor1, tensor2, n=100):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        output2 = torch.matmul(tensor2, tensor1)

    torch.cuda.synchronize()
    end = time.time()
    print('matmul time:', end-start)
    print('matmul memory (GB):', torch.cuda.max_memory_allocated('cuda')/10**9)
    return output2

output1 = test_einsum(tensor1, tensor2)
output2 = test_matmul(tensor1, tensor2)

print('same res?', torch.allclose(output1, output2, atol=1e-5)) # we are using float not double

'''
if we run this code as is, we get:
einsum time: 2.7761459350585938
einsum memory (GB): 1.879048192
matmul time: 2.7803025245666504
matmul memory (GB): 2.147483648

if we switch the order of the tests, we get:
matmul time: 2.775181531906128
matmul memory (GB): 1.879048192
einsum time: 2.785910129547119
einsum memory (GB): 2.147483648

So they are basically the same, and the second one always 
takes more memory (becase of the first one).
'''
