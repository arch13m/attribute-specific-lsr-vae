model.eval()
rhythmic_complexity_all = []
alpha_all               = []
for alpha in np.arange(-20, 20, 4):
  rhythmic_complexity = linear_vary_constrained_latent_dim(1, alpha, attribute="rhy")
  rhythmic_complexity_all.append(rhythmic_complexity)
  alpha_all.append(alpha)

plt.plot(alpha_all, rhythmic_complexity_all, 'red', marker = ".")
current_values = plt.gca().get_yticks()
plt.gca().set_yticks(current_values)
plt.gca().set_yticklabels(round(x,2) for x in current_values)
plt.xlabel('Alpha')
plt.ylabel('Rhythmic complexity')
plt.show()

linear_interpretability_metric(attribute="rhy")
