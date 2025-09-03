# attribute-specific-lsr-vae
Submitted as a Master's Project.

Attribute-specific latent space regularisation for MIDI VAE models. Achieved by altering the standard VAE loss function to include LSR term that maps specified musical attribute to 0th latent dimension.
Attributes used for regularisation: modal pitch; rythmic complexity.
Model architectures used: linear; CNN.
Baseline models included (no LSR term in loss function).

Evaluation: midpoint checks; plots for specified musical attribute as latent dimension varies; regression score for dimension that best fits specified musical attribute.
