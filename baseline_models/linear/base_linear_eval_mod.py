model.eval()
modal_pitch_all = []
alpha_all       = []
for alpha in np.arange(-20, 20, 4):
  modal_pitch = cnn_vary_constrained_latent_dim(1, alpha, attribute="mod")
  modal_pitch_all.append(modal_pitch)
  alpha_all.append(alpha)

plt.plot(alpha_all, modal_pitch_all, marker = ".")
current_values = plt.gca().get_yticks()
plt.gca().set_yticks(current_values)
plt.gca().set_yticklabels(pitch_num_to_name(int(x)) for x in current_values)
plt.xlabel('Alpha')
plt.ylabel('Modal pitch')
plt.show()

linear_interpretability_metric(attribute="mod")
