import plac
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import os
import models
import utils
import argparse


desc = "Pytorch implementation of 'GLO'"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dir', type=str, default='/data/LSUN_dataset/')
parser.add_argument('--prfx', type=str, default='glo')
parser.add_argument('--d', type=int, help='Dimensionality of latent representation space',default=128)
parser.add_argument('--lr_g', type=float, default=1)
parser.add_argument('--lr_z', type=float, default=10)
parser.add_argument('--e', type=int, default=25, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--init', type=str, default='pca', help='initialization')
parser.add_argument('--loss', type=str, default='lap_l1', help='loss')
parser.add_argument('--cl', type=str, default='church_outdoor', help='class of LSUN')
parser.add_argument('--gpu', type=bool, default=True,help='Use GPU?')
parser.add_argument('--n_pca', type=int, help='Number of samples to take for PCA',default=(64 * 64 * 3 * 2))
parser.add_argument('--n', type=int, help='Cap on the number of samples from the LSUN dataset',default=10000)






def main(args):

    def maybe_cuda(tensor):
        return tensor.cuda() if args.gpu else tensor

    Img_dir = 'Figs/'
    if not os.path.exists(Img_dir):
        os.makedirs(Img_dir)

    Model_dir = 'Models/'

    if not os.path.exists(Model_dir):
        os.makedirs(Model_dir)

    Data_dir = 'Data/'

    if not os.path.exists(Data_dir):
        os.makedirs(Data_dir)

    train_set = utils.IndexedDataset(
        LSUN(args.dir, classes=[args.cl+'_train'],
             transform=transforms.Compose([
                 transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ]))
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=args.gpu,
    )
    
    val_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=8*8)

    if args.n > 0:
        train_set.base.length = args.n
        train_set.base.indices = [args.n]

    # initialize representation space:
    if args.init == 'pca':
        print('Check if PCA is already calculated...')
        pca_path = 'Data/GLO_pca_init_{}_{}.pt'.format(
            args.cl, args.d)
        if os.path.isfile(pca_path):
            print(
                '[Latent Init] PCA already calculated before and saved at {}'.
                    format(pca_path))
            Z = torch.load(pca_path)
        else:
            from sklearn.decomposition import PCA

            # first, take a subset of train set to fit the PCA
            X_pca = np.vstack([
                X.cpu().numpy().reshape(len(X), -1)
                for i, (X, _, _)
                in zip(tqdm(range(args.n_pca // train_loader.batch_size), 'collect data for PCA'),
                       train_loader)
            ])
            print("perform PCA...")
            pca = PCA(n_components=args.d)
            pca.fit(X_pca)
            # then, initialize latent vectors to the pca projections of the complete dataset
            Z = np.empty((len(train_loader.dataset), args.d))
            for X, _, idx in tqdm(train_loader, 'pca projection'):
                Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))


    elif args.init == 'random':
        Z = np.random.randn(len(train_set), args.d)

    Z = utils.project_l2_ball(Z)

    model_generator = maybe_cuda(models.Generator(args.d))
    loss_fn = utils.LapLoss(max_levels=3) if args.loss == 'lap_l1' else nn.MSELoss()
    zi = maybe_cuda(torch.zeros((args.batch_size, args.d)))
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': model_generator.parameters(), 'lr': args.lr_g},
        {'params': zi, 'lr': args.lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))
    utils.imsave(Img_dir+'target_%s_%s.png' % (args.cl,args.prfx),
           make_grid(Xi_val.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

    for epoch in range(args.e):
        losses = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)

        for i, (Xi, yi, idx) in enumerate(train_loader):
            Xi = Variable(maybe_cuda(Xi))
            zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))

            optimizer.zero_grad()
            rec = model_generator(zi)
            loss = loss_fn(rec, Xi)
            loss.backward()
            optimizer.step()

            Z[idx.numpy()] = utils.project_l2_ball(zi.data.cpu().numpy())

            losses.append(loss.data[0])
            progress.set_postfix({'loss': np.mean(losses[-100:])})
            progress.update()


        progress.close()

        # visualize reconstructions
        rec = model_generator(Variable(maybe_cuda(torch.FloatTensor(Z[idx_val.numpy()]))))
        utils.imsave(Img_dir+'%s_%s_rec_epoch_%03d_%s.png' % (args.cl,args.prfx,epoch, args.init),
               make_grid(rec.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))
    print('Saving the model : epoch % 3d'%epoch)
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model_generator.state_dict(),
    },
        Model_dir + 'Glo_{}_z_{}_epch_{}_init_{}.pt'.format(args.cl,args.d, epoch, args.init))


if __name__ == "__main__":
    args= parser.parse_args()
    main(args)

