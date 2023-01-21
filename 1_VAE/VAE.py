# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:47:57 2022

@author: G.Lippen
"""
import time
import torch
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# 使用 visdom 前，在命令行输入 python -m visdom.server 请求一个客户端，然后打开所给的 http 网站

class VAE(nn.Module):    
    def __init__(self):
        super(VAE, self).__init__()     

        # [batch_size, 784] => [batch_size, 20]
        self.encoder = nn.Sequential(nn.Linear(784, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, 64), 
                                     nn.ReLU(),
                                     nn.Linear(64, 20),
                                     nn.ReLU())        
        
        # mu:    [batch_size, 10]
        # sigma: [batch_size, 10]

        # [batch_size, 10] => [batch_size, 784]
        self.decoder = nn.Sequential(nn.Linear(10, 64),
                                     nn.ReLU(), 
                                     nn.Linear(64, 256), 
                                     nn.ReLU(), 
                                     nn.Linear(256, 784),
                                     nn.Sigmoid())
        
    def forward(self, x):
        """
        para x: [batch_siez, 1, 28, 28]
        return:
        """
        batch_size = x.size(0)
        
        # encoder
        x = x.view(batch_size, 784)
        x = self.encoder(x)                         # => [bs, 20]

        # reparametrize trick, epsilon~N(0, 1)
        mu, sigma = x.chunk(2, dim=1)               # => [bs, 10] & [bs, 10]
        h = mu + sigma * torch.randn_like(sigma)    # => [bs, 10]
        
        # decoder & reshape
        x_hat = self.decoder(h)
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        
        # KL divergence
        KLd = 0.5 * torch.sum(torch.pow(mu, 2)
                              + torch.pow(sigma, 2)
                              - torch.log(1e-8 + torch.pow(sigma, 2))
                              - 1) / batch_size / 28 / 28
        
        return x_hat, KLd
        
def main():
    # MNIST datasets for train and test
    mnist_train = datasets.MNIST('mnist', 
                           train=True, 
                           transform=transforms.Compose([transforms.ToTensor()]), 
                           download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
    
    mnist_test = datasets.MNIST('mnist', 
                           train=False, 
                           transform=transforms.Compose([transforms.ToTensor()]), 
                           download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)
    
    x, _ = iter(mnist_train).next()
    print('Data shape is :\n', x.shape)
    
    # build model and optimizer
    device = torch.device('cpu')
    model = VAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('the VAE model is:\n ', model)
    
    # train
    viz = visdom.Visdom()    
    for epoch in range(1000):
        time_start = time.time()
        for _, (x, _) in enumerate(mnist_train):    # _ & _ => index & lable
            x = x.to(device)
            
            x_hat, KLd = model(x)
            loss = criteon(x_hat, x)
            
            if KLd is not None:
                ELBO = - loss - 1.0 * KLd
                loss = - ELBO
            
            # backward propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat, KLd = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))
        time_end = time.time()
        print(epoch, 'loss', loss.item(), KLd.item(), 'Time cost = %fs' % (time_end-time_start))

if __name__ == '__main__':
    main()