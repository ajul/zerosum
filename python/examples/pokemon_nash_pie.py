import _initpath

import numpy
import dataset.pokemon
import zerosum.balance
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

type_chart = dataset.pokemon.pokemon_6
# Vector of color codes for the types.
colors = [dataset.pokemon.pokemon_type_colors[name] for name in type_chart.row_names]

row_nash, col_nash = zerosum.nash.nash(type_chart.data)

dpi = 120
figsize = (12, 6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize, dpi = dpi)

labels = [(name if weight > 0.0 else '') for (name, weight)
          in zip(type_chart.row_names, row_nash.strategy)]
ax1.pie(row_nash.strategy, labels = labels, colors = colors)
ax1.axis('equal')
ax1.set_title('Attacker')

labels = [(name if weight > 0.0 else '') for (name, weight)
          in zip(type_chart.col_names, col_nash.strategy)]
ax2.pie(col_nash.strategy, labels = labels, colors = colors)
ax2.axis('equal')
ax2.set_title('Defender')

plt.suptitle('Pokémon Nash equilibrium')
fig.subplots_adjust(wspace=0.5)
plt.savefig("out/pokemon_nash_pie.png", dpi = dpi, bbox_inches = "tight")
plt.show()
