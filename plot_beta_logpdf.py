import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the beta distribution
alpha = 203
beta_param = 133

# Create x values from 0 to 1
x = np.linspace(0.001, 0.999, 1000)

# Calculate log probability density function
logpdf_values = beta.logpdf(x, alpha, beta_param)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, logpdf_values, 'b-', linewidth=2, label=f'Beta logpdf (α={alpha}, β={beta_param})')
plt.xlabel('x')
plt.ylabel('Log Probability Density')
plt.title(f'Beta Distribution Log PDF (α={alpha}, β={beta_param})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('plots/beta_logpdf.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Beta distribution with α={alpha}, β={beta_param}")
print(f"Mean: {alpha / (alpha + beta_param):.4f}")
print(f"Mode: {(alpha - 1) / (alpha + beta_param - 2):.4f}")
print(f"Maximum logpdf value: {np.max(logpdf_values):.4f}")
print(f"x value at maximum: {x[np.argmax(logpdf_values)]:.4f}") 