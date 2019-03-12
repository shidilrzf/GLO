from torch import nn

class Generator(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), # 2x2
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32
            nn.Tanh(),
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))

class Generator_cifar(nn.Module):
    def __init__(self, code_dim, n_filter=32, out_channels=3):
        super(Generator_cifar, self).__init__()
        self.code_dim = code_dim
        nf = 32
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 2, 1, bias=False), # 2x2
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32
            nn.Tanh(),
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Encoder_cifar(nn.Module):
    def __init__(self, code_dim, in_channels=3):
        super(Encoder_cifar, self).__init__()
        self.code_dim =code_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),  #
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),  #
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=False),  #
            nn.BatchNorm2d(16), nn.ReLU(True),

        )


        self.fc = nn.Linear(1024, code_dim)
        self.tanh = nn.Tanh()
        self.flatten = Flatten()



    def forward(self, x):
        x = self.enc_conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        z = self.tanh(x)


        return z



class Encoder_cifar_stc(nn.Module):
    def __init__(self,code_dim):
        super(Encoder_cifar_stc, self).__init__()
        self.code_dim =code_dim

        self.fc1 = nn.Linear(1024, 32 * 32)

        self.enc_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 7, stride=2, padding=3, bias=False),  #
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, bias=False),  #
            nn.BatchNorm2d(16), nn.ReLU(True),

        )

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512, code_dim)
        self.tanh = nn.Tanh()
        self.flatten = Flatten()



    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        z = self.fc4(x)


        return z

