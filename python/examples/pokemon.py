import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

# plots handicaps (using a multiplicative handicap function)
# versus Nash equilibrium of the original game

data = dataset.nonsymmetric.pokemon_type_chart_6
names = dataset.nonsymmetric.pokemon_type_names_6
colors = [dataset.nonsymmetric.pokemon_type_colors[name] for name in names]

row_nash, col_nash = zerosum.nash.nash(data)
balance = zerosum.balance.MultiplicativeBalance(data).optimize()

dpi = 240
marker_size = 64
text_size = 15
loc = matplotlib.ticker.MultipleLocator(base=0.1)

# plot attacker and defender on separate subplots
fig = plt.figure(figsize = (12, 7))
gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2, 3])
ax = plt.subplot(gs[0])

# draw a line indicating where the uniform distribution would be
uniform_x = 1.0 / len(row_nash.strategy)
ax.plot([uniform_x, uniform_x], [-0.2, 0.25],
        color='#bfbfbf', linestyle='-', zorder=0)
plt.text(uniform_x, -0.203, 'Uniform',
         fontsize = 12,
         rotation = 0,
         ha = 'center', va = 'top')

# scatter plot of handicaps vs. nash
x = row_nash.strategy
y = balance.row_log_handicaps

ax.scatter(x, y, s = marker_size, c = colors)

# label each scatter plot point with the type name
for pointx, pointy, name, color in zip(x, y, names, colors):
    ha = 'left'
    # manual adjustment
    if name in ['Flying']:
        pointy -= 0.005
    if name in ['Dark']:
        pointy -= 0.01
    if name in ['Fairy']:
        pointx -= 0.005
        pointy -= 0.015
    if name in ['Ground']:
        ha = 'right'
    
    name = ' ' + name + ' '
    plt.text(pointx, pointy, name,
             fontsize = text_size,
             rotation = 0,
             ha = ha, va = 'center')

ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.set_title('Attacker')

ax.set_xlabel('Nash probability of initial game', fontsize = text_size)
ax.set_ylabel('Log (handicap) producing uniform Nash', fontsize = text_size)

ax.set_xlim(left=0.0, right = 0.2)
ax.set_ylim(bottom=-0.2, top = 0.25)

# now for the defender plot
ax = plt.subplot(gs[1])

uniform_x = 1.0 / len(col_nash.strategy)
ax.plot([uniform_x, uniform_x], [-0.2, 0.25],
        color='#bfbfbf', linestyle='-', zorder=0)
plt.text(uniform_x, -0.203, 'Uniform',
         fontsize = 12,
         rotation = 0,
         ha = 'center', va = 'top')

x = col_nash.strategy
y = balance.col_log_handicaps

ax.scatter(x, y, s = marker_size, c = colors)

for pointx, pointy, name, color in zip(x, y, names, colors):
    ha = 'left'
    # manual adjustment
    if name in ['Dark']:
        pointx -= 0.002
        pointy += 0.012
    if name in ['Dragon', 'Fighting']:
        pointy -= 0.01
    if name in ['Normal']:
        ha = 'right'
    
    name = ' ' + name + ' '
    plt.text(pointx, pointy, name,
             fontsize = text_size,
             rotation = 0,
             ha = ha, va = 'center')

ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.set_title('Defender')

ax.set_xlabel('Nash probability of initial game', fontsize = text_size)
ax.yaxis.set_ticklabels([])

ax.set_xlim(left=0.0, right = 0.3)
ax.set_ylim(bottom=-0.2, top = 0.25)

plt.tight_layout()

plt.savefig("out/pokemon_scatter.png", dpi = dpi, bbox_inches = "tight")
plt.show()
