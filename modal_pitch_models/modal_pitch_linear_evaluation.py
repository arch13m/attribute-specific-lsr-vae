# Load model
model = LINEAR_LSR(LATENT_DIM)
model.load_state_dict(torch.load('./MOD_LINEAR_LSR.pth'))

model.eval()
modal_pitches = []
alpha_all     = []

for alpha in np.arange(-20, 20, 4):
  # Vary 0th latent dimension for abritrary sample
  modal = linear_vary_constrained_latent_dim(1, alpha, attribute="mod")
  modal_pitches.append(modal)
  alpha_all.append(alpha)

# Plot changes in modal pitch as 0th latent dimension varies
plt.plot(alpha_all, modal_pitches, marker = ".")
current_values = plt.gca().get_yticks()
plt.gca().set_yticks(current_values)
plt.gca().set_yticklabels(pitch_num_to_name(int(x)) for x in current_values)
plt.xlabel('Alpha')
plt.ylabel('Pitch')
plt.show()

# Run interpretability metric for given sample
linear_interpretability_metric(attribute="mod")
