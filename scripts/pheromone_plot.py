import numpy as np
import matplotlib.pyplot as plt
import seaborn


# Init seaborn
seaborn.set()

# Create data which fills the plot
xn_n1_0_82 = np.arange(-1.25, -1, 0.001)
yn_n1_0_82 = 1/np.exp(abs(xn_n1_0_82))

xn_0_82_0_6 = np.arange(-1, -0.5, 0.001)
yn_0_82_0_6 = 1/np.exp(abs(xn_0_82_0_6))

xn_0_6_037 = np.arange(-0.5, -0.2, 0.001)
yn_0_6_037 = 1/np.exp(abs(xn_0_6_037))

x037 = np.arange(-0.2, 0.2, 0.001)
y037 = 1/np.exp(abs(x037))

# Plot the data
x_all = np.arange(-2, 2, 0.001)
y2 = 1/np.exp(abs(x_all))

# build the plot
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_all, y2, color='r', linewidth=2, label='pheromone level')

# add the filled areas for discretized pheromone levels
ax.fill_between(xn_n1_0_82,yn_n1_0_82,0, alpha=0.1, color='b', label='0')
ax.fill_between(xn_0_82_0_6,yn_0_82_0_6,0, alpha=0.3, color='b', label='1')
ax.fill_between(xn_0_6_037,yn_0_6_037,0, alpha=0.5, color='b', label='2')
ax.fill_between(x037,y037,0, alpha=0.8, color='b', label='3')
ax.fill_between(-xn_0_6_037,yn_0_6_037,0, alpha=0.5, color='b')
ax.fill_between(-xn_0_82_0_6,yn_0_82_0_6,0, alpha=0.3, color='b')
ax.fill_between(-xn_n1_0_82,yn_n1_0_82,0, alpha=0.1, color='b')


ax.set_xlim([-1.25, 1.25])
ax.set_ylim([0, 1.2])
ax.set_ylabel('pheromone concentration')
ax.set_xlabel('distance from target direction')
# ax.set_title('Pheromones Curve')

plt.legend()
plt.savefig('pheromone.png', dpi=72, bbox_inches='tight')
plt.show()