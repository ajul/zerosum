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

balance_result = zerosum.balance.MultiplicativeBalance(type_chart.data).optimize()


dpi = 120
figsize = (12, 6)
bar_width = 0.8
bar_offset = 0.5 * (1.0 - bar_width)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = figsize, dpi = dpi)
x = numpy.arange(len(balance_result.row_handicaps))

ax1.bar(x + bar_offset, numpy.log(balance_result.row_handicaps), bar_width, color = colors)
ax1.set_xticks([])
ax1.set_xticks(x + 0.5)
ax1.set_xticklabels(['' for _ in x])
ax1.set_title('Attacker')

ax2.bar(x + bar_offset, numpy.log(balance_result.col_handicaps), bar_width, color = colors)
ax2.set_xticks(x + 0.5)
ax2.set_xticklabels(type_chart.col_names, rotation='vertical')
ax2.set_title('Defender')

plt.suptitle('Pokémon log handicaps')
#fig.subplots_adjust(hspace=0.25)
plt.savefig("out/pokemon_handicap_bar.png", dpi = dpi, bbox_inches = "tight")
plt.show()
