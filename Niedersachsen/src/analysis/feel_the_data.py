# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.data import dataload as dl

gld = dl.load_data(loadExistingData = True)

# %% get a feel of the data ###
############################
# 1. print info to learn about the total number of entries (len(gld)), data types and presence/ number of missing values
gld.info()

# 2. create a copy removing FLIK, area_m2, cpar, shp_index, fract
# and making kulturcode, CELLCODE and LANDKREIS nominal
def copy_data(data):
    data = data.drop(columns=['FLIK', 'area_m2', 'cpar','shp_index', 'fract'])
    data['kulturcode'] = data['kulturcode'].astype('category')
    data['CELLCODE'] = data['CELLCODE'].astype('category')
    data['LANDKREIS'] = data['LANDKREIS'].astype('category')
    return data
data = copy_data(gld)

# 3. examine data distribution (perhaps clean outliers)
# 3.1. take two representative years, 1st and last and create distribution plots for
# numeric columns
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def distribution_plots(df):
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Group by year and create distribution plots for numeric columns in a FacetGrid
    for year, group in df.groupby('year'):
        plot_file = f'reports/distribution_plots_{year}.png'
        
        if os.path.exists(plot_file):
            # Load and display the plot from the file
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.title(f'Distribution of Numeric Columns in {year}')
            plt.show()
        else:
            # Create a FacetGrid for the current year
            g = sns.FacetGrid(pd.melt(group, id_vars=['year'], value_vars=numeric_columns), col='variable', col_wrap=3, sharex=False, sharey=False)
            g.map(sns.histplot, 'value', kde=True)
            
            # Set titles and labels
            g.fig.suptitle(f'Distribution of Numeric Columns in {year}', y=1.02)
            g.set_axis_labels('Value', 'Frequency')
            
            # Save the FacetGrid plot
            g.savefig(plot_file)
            plt.show()
# Filter the DataFrame to include only the years 2012 and 2023
filtered_data = data[data['year'].isin([2012, 2023])]
distribution_plots(filtered_data)
#  the data has a long right tail which suggests the presence of outliers

# %% 3.2 plot box plot with and without outliers so as to get the outlier threshold
# we use area_ha as the control column
def box_plot(df, column):
    sns.boxplot(df[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
    
box_plot(data, 'area_ha')

def datatrim_box_plot(df, column, threshold):
    data_trim = df[df[column] <= threshold]

    sns.boxplot(data_trim[column])
    plt.title(f'Box Plot without Outliers of {column}')
    plt.show()
    
    # Calculate quartile
    Q1 = data_trim['area_ha'].quantile(0.25)
    Q2 = data_trim['area_ha'].quantile(0.50)
    Q3 = data_trim['area_ha'].quantile(0.75)
    #Q4 = data_trim['area_ha'].quantile(0.935)
    
    # print the quartiles
    print(f"Q1: {Q1}, Q2: {Q2}, Q3: {Q3}") #, Q4: {Q4}")
    return data_trim

threshold_value = 20
data_trim = datatrim_box_plot(data, 'area_ha', threshold_value)
# from the box plot, the threshold value can be set at 20 and
# we can use 0, 1, 2, 4, 6, 8, 10, 20, as bins for understanding area distribution

# %% 3.3 create a bar plot of the percentage distribution of area_ha in specified ranges
# to get a better understanding of the data distribution with and without outliers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bar_plot(data, column, bins, labels):
    # Bin the data
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=labels, right=False)

    # Calculate percentage frequency using the dynamically named binned column
    percentage_frequency = data[f'{column}_binned'].value_counts(normalize=True) * 100
    percentage_frequency = percentage_frequency.reindex(labels)  # Ensure the order matches the labels
    
    # Drop empty bins if any
    percentage_frequency = percentage_frequency[percentage_frequency > 0]

    # Convert labels to strings to avoid the warning
    percentage_frequency.index = percentage_frequency.index.astype(str)

    # Create the percentage frequency bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=percentage_frequency.index, y=percentage_frequency.values)
    plt.title(f'Percentage Distribution of {column} in Specified Ranges')
    plt.xlabel('Range')
    plt.ylabel('Percentage Frequency')
    plt.grid(False)
    plt.xticks(rotation=45)

    # Remove the top and right spines
    sns.despine(left=True, bottom=True)

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.show()

# Usage for the field level data
bins1 = [0, 1, 2, 4, 6, 8, 10, 20, float('inf')]
labels1 = ['0-1', '1-2', '2-4', '4-6', '6-8', '8-10', '10-20', '>20']
bar_plot(data, 'area_ha', bins1, labels1)
bar_plot(data_trim, 'area_ha', bins1, labels1)
#data_trim.info()

# %% mean of data_trim 'area_ha' column
data_trim['area_ha'].mean()

# %% 4.  yearly exploration of the data without outliers (i.e., use data_trim)
# 4.1. group data by year and count the unique values in the non-numeric columns
# Identify non-numeric columns
def count_unique(data):
    category_columns = data.select_dtypes(include=['category']).columns
    # Exclude 'area_ha_binned'
    category_columns = category_columns.drop('area_ha_binned', errors='ignore')
    # Group by year and count unique values in non-numeric columns
    unique_counts_by_year = gld.groupby('year')[category_columns].nunique()
    # Print the unique counts
    print("Unique counts in non-numeric columns by year:")
    print(unique_counts_by_year)
count_unique(data_trim)

# %% 4.2. for numeric columns, get the range of of values: the min, max, mean, median a standard deviation
def get_numstats(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    range_stats_by_year = gld.groupby('year')[numeric_columns].agg(['min', 'max', 'mean', 'median', 'std'])
    output_excel_path = 'reports/trim_numcolstats_by_year.xlsx'
    range_stats_by_year.to_excel(output_excel_path, sheet_name='NumericColumnStats')
    print(f"Range statistics saved to {output_excel_path}")
get_numstats(data_trim)

# %% 4.3. replot the distribution plots for the numeric columns
def distribution_plots(df):
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Group by year and create distribution plots for numeric columns in a FacetGrid
    for year, group in df.groupby('year'):
        plot_file = f'reports/trimmed_distribution_plots_{year}.png'
        
        if os.path.exists(plot_file):
            # Load and display the plot from the file
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.title(f'Distribution of Numeric Columns in {year} without Outliers')
            plt.show()
        else:
            # Create a FacetGrid for the current year
            g = sns.FacetGrid(pd.melt(group, id_vars=['year'], value_vars=numeric_columns), col='variable', col_wrap=3, sharex=False, sharey=False)
            g.map(sns.histplot, 'value', kde=True)
            
            # Set titles and labels
            g.fig.suptitle(f'Distribution of Numeric Columns in {year} without Outliers', y=1.02)
            g.set_axis_labels('Value', 'Frequency')
            
            # Save the FacetGrid plot
            g.savefig(plot_file)
            plt.show()
# Filter the DataFrame to include only the years 2012 and 2023
filtered_data = data_trim[data_trim['year'].isin([2012, 2023])]
distribution_plots(filtered_data)

# %% 5. Test for normal distribution
from scipy.stats import shapiro

# Group by year and perform normality tests for numeric columns
normality_results = {}
numeric_columns = data_trim.select_dtypes(include=['number']).columns
for year, group in data_trim.groupby('year'):
    normality_results[year] = {}
    for column in numeric_columns:
        stat, p_value = shapiro(group[column].dropna())
        normality_results[year][column] = {'statistic': stat, 'p_value': p_value}

# Print the normality test results
print("Normality test results (Shapiro-Wilk) for numeric columns by year:")
for year, results in normality_results.items():
    print(f"\nYear: {year}")
    for column, result in results.items():
        print(f"  Column: {column}, Statistic: {result['statistic']:.4f}, P-value: {result['p_value']:.4f}")
        
# %% 6. Test for correlation between numeric columns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def test_correlation(df):
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    
    # Print the correlation matrix
    print("Correlation matrix:")
    print(correlation_matrix)
    
    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numeric Columns')
    plt.show()

# Example usage
test_correlation(data_trim)

# %%
# general data descriptive statistics grouped by year
def yearly_gen_statistics(gld):
    yearlygen_stats = gld.groupby('year').agg(
        fields = ('area_ha', 'count'),
        
        area_ha_sum=('area_ha', 'sum'),
        area_ha_mean=('area_ha', 'mean'),
        area_ha_median=('area_ha', 'median'),

        peri_m_sum=('peri_m', 'sum'),
        peri_m_mean=('peri_m', 'mean'),
        peri_m_median=('peri_m', 'median'),
                    
        par_sum=('par', 'sum'),
        par_mean=('par', 'mean'),
        par_median=('par', 'median'),
                                

    ).reset_index()
    
    yearlygen_stats['fields_ha'] = yearlygen_stats['fields'] / yearlygen_stats['area_ha_sum']

    return yearlygen_stats

ygs = yearly_gen_statistics(data)
#ygs1 = yearly_gen_statistics(data_trim)

# %%
# Create line plot of yearly sum of all field areas or any other ygs/ygs1 column
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("colorblind")

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=ygs, x='year', y='area_ha_sum', color='purple', marker='o', ax=ax)
ax.set_title('Yearly sum of all field areas (Trimmed)')
ax.set_ylabel('Area (ha)')
ax.set_xlabel('Year')
plt.show()

# %%