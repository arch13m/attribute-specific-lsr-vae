class LINEAR_LSR(nn.Module):
    def __init__(self, latent_dim):  # Pianoroll input: 128 pitches, 96 time steps
        super(LINEAR_LSR, self).__init__()
        self.input_dim=128*96
        self.flatten_input = nn.Flatten()
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)  # mean
        self.fc32 = nn.Linear(256, latent_dim)  # logvar

        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, self.input_dim)

    def encode(self, x):
        x = self.flatten_input(x)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        z = self.decode(z)
        z = z.reshape(-1, 128, 96)
        return z, mu, logvar

# Instantiate the model with the specified latent dimensions
model = LINEAR_LSR(LATENT_DIM)

# Outputs a summary of the model for manual checks
#summary(model, input_size=(1, 1, 128, 96))
