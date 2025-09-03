class CNN_LSR(nn.Module):
    def __init__(self, latent_dim):  # Pianoroll input: 128 pitches, 96 time steps
        super(CNN_LSR, self).__init__()
        self.latent_dim = latent_dim
        self.input_size=128*96
        self.flattened_size = 1120
        #self.flatten_input = nn.Flatten()
        self.encoder_layers = nn.Sequential(
        nn.Conv2d(1, 512, kernel_size=3, stride=2, padding=2),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 32, kernel_size=3, stride=(2,1), padding=1),
        nn.Flatten(),
        )

        self.fc1 = nn.Linear(self.flattened_size, self.latent_dim)  # mean
        self.fc2 = nn.Linear(self.flattened_size, self.latent_dim)  # logvar

        self.decoder_fc = nn.Linear(in_features=self.latent_dim, out_features=self.flattened_size)

        self.decoder_layers = nn.Sequential(
          nn.ConvTranspose2d(32, 32, kernel_size=3, stride=(2,1), padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.ConvTranspose2d(256, 1, kernel_size=3, stride=2, padding=2, output_padding=1),
          nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_layers(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.reshape(-1, 32, 5, 7)
        z = self.decoder_layers(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar

# Instantiate the model with the specified latent dimensions
model = CNN_LSR(LATENT_DIM)

# Outputs a summary of the model for manual checks
summary(model, input_size=(98, 1, 128, 96))
