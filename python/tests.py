import torch

def test_sanity_check():

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    # assert ymg.data == ypt.data.item()
    # backward pass went well
    # assert xmg.grad == xpt.grad.item()
    print("Expected ypt: ")
    print(ypt)
    
    # print(ypt.grad.item())
    
    print("Expected xpt: ")
    
    print(xpt)

test_sanity_check()
