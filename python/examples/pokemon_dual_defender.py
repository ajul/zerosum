import _initpath

import numpy
import dataset.nonsymmetric
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

# Balances the Pokemon type chart using a multiplicative handicap function.
# This one includes all possible dual-type defenses.
# Plots the handicaps versus Nash equilibrium of the original game.

# Initial payoff matrix.
data = dataset.nonsymmetric.pokemon_6_dual_defender.data
# Names of the Pokemon types.
row_names = dataset.nonsymmetric.pokemon_6_dual_defender.row_names
# Vector of color codes for the types.
row_colors = [dataset.nonsymmetric.pokemon_type_colors[name.split('/')[0]] for name in row_names]

# Names of the Pokemon types.
col_names = dataset.nonsymmetric.pokemon_6_dual_defender.col_names
# Vector of color codes for the types.
col_colors = [dataset.nonsymmetric.pokemon_type_colors[name.split('/')[0]] for name in col_names]

# Nash equilibrium of the initial game.
row_nash, col_nash = zerosum.nash.nash(data)
# Handicaps producing a uniform Nash equilibrium.
balance = zerosum.balance.MultiplicativeBalance(data).optimize()

# Now to plot.
dpi = 240
marker_size = 64
text_size = 15
bottom = -0.4
top = 0.4
loc = matplotlib.ticker.MultipleLocator(base=0.1)

# Plot attacker and defender on separate subplots. Attacker first.
fig = plt.figure(figsize = (18, 6))
gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax = plt.subplot(gs[0])

# Draw a line indicating where the uniform distribution would be.
uniform_x = 1.0 / len(row_nash.strategy)
ax.plot([uniform_x, uniform_x], [bottom, top],
        color='#bfbfbf', linestyle='-', zorder=0)
plt.text(uniform_x, bottom - 0.003, 'Uniform',
         fontsize = 12,
         rotation = 0,
         ha = 'center', va = 'top')

# Scatter plot of handicaps vs. Nash of initial game.
x = row_nash.strategy
y = numpy.log(balance.row_handicaps)
y -= numpy.mean(y)

ax.scatter(x, y, s = marker_size, c = row_colors)

# Label each scatter plot point with the type name.
for pointx, pointy, name, color in zip(x, y, row_names, row_colors):
    ha = 'left'
    # manual adjustment
    if name == 'Steel':
        pointy += 0.01
    if name == 'Ghost':
        pointy -= 0.02
    
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

ax.set_xlim(left=0.0, right = 0.25)
ax.set_ylim(bottom=bottom, top = top)

# Now for the defender plot.
ax = plt.subplot(gs[1])

uniform_x = 1.0 / len(col_nash.strategy)
ax.plot([uniform_x, uniform_x], [bottom, top],
        color='#bfbfbf', linestyle='-', zorder=0)
plt.text(uniform_x, bottom - 0.003, 'Uniform',
         fontsize = 12,
         rotation = 0,
         ha = 'left', va = 'top')

x = col_nash.strategy
y = numpy.log(balance.col_handicaps)
y -= numpy.mean(y)

sel = x >= 1e-6

ax.scatter(x[sel], y[sel], s = marker_size, c = [c for i, c in enumerate(col_colors) if sel[i]])

for pointx, pointy, name, color in zip(x, y, col_names, col_colors):
    ha = 'left'
    if pointx < 1e-6: continue
    # manual adjustment
    if name == 'Fairy':
        pointy += 0.01
    
    name = ' ' + name + ' '
    plt.text(pointx, pointy, name,
             fontsize = text_size,
             rotation = 0,
             ha = ha, va = 'center')

ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.set_title('Defender (nonzero Nash only)')

ax.set_xlabel('Nash probability of initial game', fontsize = text_size)
ax.yaxis.set_ticklabels([])

ax.set_xlim(left=0.0, right = 0.25)
ax.set_ylim(bottom=bottom, top = top)

plt.tight_layout()

plt.savefig("out/pokemon_dual_defender_scatter.png", dpi = dpi, bbox_inches = "tight")
plt.show()

