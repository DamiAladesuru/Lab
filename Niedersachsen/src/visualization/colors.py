# %%
import pandas as pd

# Group hue colors dictionary
hue_colors = {
    'ackerfutter': '#8c564b',
    'andere handelsgewächse': '#ffbb78',
    'aukm': '#98df8a',
    'aus der produktion genommen': '#ff9896',
    'dauergrünland': '#006D5B',
    'dauerkulturen': '#2ca02c',
    'eiweißpflanzen': '#c5b0d5',
    'energiepflanzen': '#c49c94',
    'gemüse': '#d62728',
    'getreide': '#bcbd22',
    'hackfrüchte': '#9467bd',
    'kräuter': '#f7b6d2',
    'leguminosen': '#ff7f0e',
    'mischkultur': '#17becf',
    'sonstige flächen': '#7f7f7f',
    'stilllegung/aufforstung': '#dbdb8d',
    'zierpflanzen': '#9edae5',
    'ölsaaten': '#e377c2'
}

# Translation dictionary
translation_dict = {
    'ackerfutter': 'forage crops',
    'andere handelsgewächse': 'other commercial crops',
    'aukm': 'aukm',
    'aus der produktion genommen': 'taken out of production',
    'dauergrünland': 'permanent grassland',
    'dauerkulturen': 'perennial crops',
    'eiweißpflanzen': 'protein plants',
    'energiepflanzen': 'energy crops',
    'gemüse': 'vegetables',
    'getreide': 'cereals',
    'hackfrüchte': 'root crops',
    'kräuter': 'herbs',
    'leguminosen': 'legumes',
    'mischkultur': 'mixed culture',
    'sonstige flächen': 'other areas',
    'stilllegung/aufforstung': 'set-aside/afforestation',
    'zierpflanzen': 'ornamental plants',
    'ölsaaten': 'oilseeds'
}

# Create a DataFrame from the dictionaries
df = pd.DataFrame({
    'Gruppe': hue_colors.keys(),
    'Hue': hue_colors.values(),
    'Translation': [translation_dict[key] for key in hue_colors.keys()]
})

# Save the DataFrame to a spreadsheet
df.to_excel('reports/figures/grp_hue_translation.xlsx', index=False)

print("Spreadsheet saved as 'hue_translation.xlsx'")
# %%
# Category2 dictionary color mapping
hue_colors = {
    'getreide': '#bcbd22',
    'ackerfutter': '#8c564b',
    'dauergrünland': '#006D5B',
    'gemüse': '#d62728',
    'hackfrüchte': '#9467bd',
    'ölsaaten': '#e377c2',
    'leguminosen': '#ff7f0e',
    'mischkultur': '#17becf',
    'dauerkulturen': '#2ca02c',
    'sonstige flächen': '#7f7f7f',
    'environmental': '#1f77b4'
}    

# Translation dictionary
translation_dict = {
    'ackerfutter': 'forage crops',
    'gemüse': 'vegetables',
    'environmental': 'environmental areas',
    'leguminosen': 'legumes',
    'mischkultur': 'mixed culture',
    'hackfrüchte': 'root crops',
    'dauergrünland': 'permanent grassland',
    'ölsaaten': 'oilseeds',
    'dauerkulturen': 'perennial crops',
    'getreide': 'cereals',
    'sonstige flächen': 'other areas'
}

# Create a DataFrame from the dictionaries
df = pd.DataFrame({
    'Category': hue_colors.keys(),
    'Hue': hue_colors.values(),
    'Translation': [translation_dict[key] for key in hue_colors.keys()]
})

# Save the DataFrame to a spreadsheet
df.to_excel('reports/figures/cat2_hue_translation.xlsx', index=False)

print("Spreadsheet saved as 'cat2_hue_translation.xlsx'")
# %%
