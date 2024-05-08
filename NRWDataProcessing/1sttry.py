# %%
import os
import geopandas as gpd
import shapely as sh
import matplotlib
# %%
os.getcwd()

# %%
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
# %%
#calculate the distance between shapely objects, such as two points
Point(0,0)
a = Point(0, 0)
b = Point(1, 0)
a.distance(b)

# %%
# Place multiple points into a single object:
MultiPoint([(0,0), (0,1), (1,1), (1,0)])
# %%
#form a line object
line = LineString([(0,0),(1,2), (0,1)])
line

# %%
#define length and bounds attributes:
print(f'Length of line {line.length}')
print(f'Bounds of line {line.bounds}')

# %%
pol = Polygon([(0,0), (0,1), (1,1), (1,0)]) #define polygon
pol
# but how does the it work with creating irregukar shaped polygons?
# %%
#define polygon attributes
pol.area
# %%
world_gdf = gpd.read_file(
    gpd.datasets.get_path('naturalearth_lowres')
)
world_gdf

# %%
world_gdf['pop_density'] = world_gdf.pop_est / world_gdf.area * 10**6

world_gdf.sort_values(by='pop_density', ascending=False)

# %%
figsize = (20, 11)

world_gdf.plot('pop_density', legend=True, figsize=figsize);

# %%
norm = matplotlib.colors.LogNorm(vmin=world_gdf.pop_density.min(), vmax=world_gdf.pop_density.max())

world_gdf.to_crs('epsg:4326').plot("pop_density", 
                                   figsize=figsize, 
                                   legend=True,  
                                   norm=norm);


