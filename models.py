import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 30 #size of the latent space in the variational autoencoder

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()
        
        # layers for encoder
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32*7*9*7, LATENT_DIM)
        self.fc2 = nn.Linear(32*7*9*7, LATENT_DIM)
        
        # layers for decoder
        self.fc_decoder = nn.Linear(LATENT_DIM, 32*7*9*7)
        
        self.conv1_decoder = nn.Conv3d(32, 32, kernel_size=3, padding=1) 
        self.conv2_decoder = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3_decoder = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        
        self.conv4_decoder = nn.Conv3d(32, 1, kernel_size=3, padding=(3,1,3))
        
        
    def encode(self, x):
        x = F.relu(self.conv1(x)) #shape after conv: (8, 61, 73, 61)
        x = F.max_pool3d(x, kernel_size=2) #shape after pooling: (8, 30, 36, 30)

        x = F.relu(self.conv2(x)) #shape after conv: (16, 30, 36, 30)
        x = F.max_pool3d(x, kernel_size=2) #shape after pooling: (16, 15, 18, 15)
        
        x = F.relu(self.conv3(x)) #shape after conv: (32, 15, 18, 15)
        x = F.max_pool3d(x, kernel_size=2) #shape after pooling: (32, 7, 9, 7)

        x = x.view(-1, 7*9*7*32)
        return self.fc1(x), self.fc2(x)
    
    
    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    
    def decode(self, z):
        z = F.relu(self.fc_decoder(z))
        z = z.view(-1, 32,7,9,7) #reshape to (32, 7, 9, 7)
        
        z = F.relu(self.conv1_decoder(z)) #shape after conv (32, 7, 9, 7)
        z = F.upsample(z, scale_factor=2, mode='nearest') #shape after upsampling (32, 14, 18, 14)
        z = F.relu(self.conv2_decoder(z)) #shape after conv (32, 14, 18, 14)
        z = F.upsample(z, scale_factor=2, mode='nearest') #shape after upsampling (32, 28, 36, 28)
        z = F.relu(self.conv3_decoder(z)) #shape after conv (32,28,36,28)
        z = F.upsample(z, scale_factor=2, mode='nearest') #shape after conv (32, 56, 72, 56)
        z = self.conv4_decoder(z) #shape after conv (1, 60, 72, 60)
        z = F.pad(z, (0,1,0,1,0,1), "constant", -10) #after padding (1, 61, 73, 61) (to match the input size)
        return F.sigmoid(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    