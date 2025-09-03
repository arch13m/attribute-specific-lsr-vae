def loss_fn(recon_x, x, mu, logvar, attr):
  # Standard VAE loss function computes the sum of the reconstruction loss (binary cross-entropy) and the KL divergence.
  BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  z = model.reparameterise(mu, logvar)
  # Calculates the MSE between the value of a certain element's 0th dimension of latent code, and its rhythmic complexity.
  LSR = F.mse_loss(z[:, 0], attr)

  LSR_coef = 10000
  KLD_coef = 0.1

  return BCE + KLD_coef * KLD + LSR_coef * LSR

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, device, train_loader, optimiser, epoch):
  model.to(device)
  model.train()
  train_loss = 0
  for batch_idx, data in enumerate(train_loader):
      data = data.to(device)
      optimiser.zero_grad()
      recon_batch, mu, logvar = model(data)

      # Get the actual batch size for the current batch
      current_batch_size = data.size(0)

      # Calculate the rhythmic complexity for each sample in the batch
      rhythmic_complexities = []
      for i in range(current_batch_size):
          rhythmic_complexities.append(get_rhythmic_complexity(data[i]))

      # Create a tensor of the rhythmic complexity for each sample
      attr = torch.tensor(rhythmic_complexities, device=device, dtype=torch.float32)

      loss = loss_fn(recon_batch, data, mu, logvar, attr)
      loss.backward()
      train_loss += loss.item()
      optimiser.step()

  avg_loss = train_loss / len(train_loader.dataset)
  print(f'Epoch {epoch}, Loss: {avg_loss}')
  return avg_loss

def train_loop(epochs):
  losses = []
  for epoch in range(1, epochs + 1):
    avg_loss = train(model, device, data_loader, optimiser, epoch)
    losses.append(avg_loss)
    scheduler.step()

  plt.figure(figsize=(10, 5))
  plt.plot(range(1, epochs+1), losses, marker='o', label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Over Epochs')
  plt.legend()
  plt.show()

train_loop(epochs=20)

torch.save(model.state_dict(), './RHY_LINEAR_LSR_XXXX2025.pth')
