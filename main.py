#利用pytroch做简单的线性回归
import torch as t
from matplotlib import  pyplot as plt
from IPython import display

device = t.device('cpu')

t.manual_seed(100)

def init(batch_size= 8):
    '''
    随机产生数据 y = 2x + 1，加上噪声
    :param batch_size: 
    :return: 
    '''
    x = t.rand(batch_size,1,device=device)*5
    y = 2*x+1+t.randn(batch_size,1,device = device)
    return x,y

x,y = init(16)
plt.scatter(x.squeeze().cpu().numpy(),y.squeeze().cpu().numpy())

#initialize the parameter randomly
w = t.rand(1,1).to(device)
b = t.zeros(1,1).to(device)

#learning rate
lr = 0.02
for ii in range(500):
    x,y = init(batch_size = 4)
    yy = w*x+b
    loss = 0.5*(yy-y)**2
    loss = loss.mean()
    dyy = (yy - y)
    dw = x.t().mm(dyy)
    db = dyy.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)

    if ii % 1000 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 20).view(-1, 1).float()
        y = x * w + b
        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # predicted

        x2, y2 = init(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 20)
        plt.ylim(0, 45)
        plt.show()
        plt.pause(0.5)

print('w = ',w.item())
print('b = ',b.item())
