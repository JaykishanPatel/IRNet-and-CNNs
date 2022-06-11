import sys, os, time, math, argparse, numpy as np
from matplotlib import pyplot as plt
import torch, torch.nn as nn
from util import fullpath

parser = argparse.ArgumentParser()
parser.add_argument( '--imagedir',   type=str, default='images' )
parser.add_argument( '--lib',        type=str, default='../lib', help='directory holding additional python modules' )
parser.add_argument( '--modelfile',  type=str, default='model-geometric-1-.pt', help='file where model state is loaded and saved' )
parser.add_argument( '--device',     type=str, default='', help='device (cuda:0 or cpu)' )
parser.add_argument( '--plotdest',   type=int, default=1, help='1 = show plot, 2 = save plot as .pdf' )
args = parser.parse_args()

# add library folder to path
if len(args.lib) > 0: sys.path.append(fullpath(args.lib))

# import local modules
import irnet, torch_util as tu

# validation image directory
valdir = os.path.join(args.imagedir, 'val')

# create network
device = args.device if len(args.device)>0 else tu.device()
net = irnet.IRNet().to( device )
tu.load(model=net, device=device, filename=args.modelfile, require=False)
net.eval()

# choose loss criterion
criterion = nn.MSELoss()

# run model on validation images
losslist = []
vgen = tu.gen('images', batchsize=1, tensor=True, device=device)

for batchk, (lum_in, rg_label, filelist) in enumerate(vgen):

        rg_hat = net.forward(lum_in)
        loss = criterion(rg_hat, rg_label)
        losslist.append(loss.item())
        print(f'{loss:.4f}')

        plt.clf()

        iml = np.moveaxis( lum_in[0,].cpu().detach().numpy(), ( 0, 1, 2 ), ( 2, 0, 1 ) )
        plt.subplot(2,2,1), plt.imshow( iml, cmap='gray' )

        imr = np.moveaxis( rg_label[0,].cpu().detach().numpy(), ( 0, 1, 2 ), ( 2, 0, 1 ) )
        plt.subplot(2,2,3), plt.imshow( imr, cmap='gray', vmin=0, vmax=1 )

        imrhat = np.moveaxis( rg_hat[0,].cpu().detach().numpy(), ( 0, 1, 2 ), ( 2, 0, 1 ) )
        plt.subplot(2,2,4), plt.imshow( imrhat, cmap='gray', vmin=0, vmax=1 )

        k = np.random.randint(0,iml.size,1000)
        plt.subplot(2,2,2), plt.plot( imr.reshape(imr.size,1)[k], imrhat.reshape(imrhat.size,1)[k], 'r.' )
        plt.axis( [ 0, 1, 0, 1 ] ), plt.xlabel( 'actual reflectance' ), plt.ylabel( 'estimated reflectance' )
        plt.axis('square')

        if args.plotdest==1:
            plt.draw()
            plt.pause(0.001)
        elif args.plotdest==2:
            plt.savefig(f'plot{batchk:03d}.pdf')
