
# %% ########################################################
# check if data contains ecological area codes
#############################################################
# check for all years in all_years the min and max value of 'kulturcode' column
# %%
print(['year'], all_years.groupby('year')['kulturcode'].max())
print(['year'], all_years.groupby('year')['kulturcode'].min())

# %%
# Extract unique values from 'kulturcode' column for future use
unique_kulturcodes = all_years['kulturcode'].unique()
# Convert to DataFrame for easy CSV export
unique_kulturcodes_df = pd.DataFrame(unique_kulturcodes, columns=['UniqueKulturcodes'])
unique_kulturcodes_df.to_csv('reports/unique_kulturcodes.csv', index=False)

# %%
def stille_count(data, year):
    a = all_years[(all_years['kulturcode'] >= 545) & (all_years['kulturcode'] <= 587)]
    b = all_years[(all_years['kulturcode'] >= 52) & (all_years['kulturcode'] <= 66)]
    acount = a.groupby('year')['kulturcode'].value_counts()
    bcount = b.groupby('year')['kulturcode'].value_counts()
    joined = pd.concat([acount, bcount], axis=1)
    sorted = joined.sort_index()
    return sorted
print(stille_count(all_years, years))
# to csv
stille_count(all_years, years).to_csv('reports/statistics/stille_count.csv') #save to csv