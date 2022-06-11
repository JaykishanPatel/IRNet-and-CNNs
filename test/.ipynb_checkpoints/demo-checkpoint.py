import numpy as np
import matplotlib.pyplot as plt
import torch_util as tu

device = tu.device()
tgen = tu.gen('images', batchsize=1, tensor=True, device=device)

for lum_in, rg_label, filelist in tgen:

        # pass lum_in to irnet, but in order to show lum_in and rg_label in figure windows,
        # you have to reshape them as follows
        iml = np.moveaxis( lum_in[0,].cpu().detach().numpy(), ( 0, 1, 2 ), ( 2, 0, 1 ) )
        imr = np.moveaxis( rg_label[0,].cpu().detach().numpy(), ( 0, 1, 2 ), ( 2, 0, 1 ) )

        print(filelist[0])
        plt.clf()
        plt.subplot(1,2,1), plt.imshow( iml, cmap='gray' )
        plt.subplot(1,2,2), plt.imshow( imr, cmap='gray', vmin=0, vmax=1 )
        plt.draw()
        plt.show()
        plt.pause(0.001)

