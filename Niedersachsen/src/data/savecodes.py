# %% 
import pandas as pd
###################################################################
# Visualizing
 # %% to obtain x and y limits for sample year data
ax = df[df['year'] == 2023].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2023')
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
plt.show()
print(f"2023 X limits: {x_lim}, Y limits: {y_lim}")
# %% to set x and y limits for sample year data
ax = df[df['year'] == 2023].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2023')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()