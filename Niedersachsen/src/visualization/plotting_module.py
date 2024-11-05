# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# %%
# Global dictionary to store colors for each label after first plot
label_color_dict = {}

'''Plotting functions'''

#############################################
# multiline plots for all data and color initialization
#############################################
def initialize_plotting(df, title, ylabel, metrics, color_dict_path):
    global label_color_dict

    if os.path.exists(color_dict_path):
        with open(color_dict_path, 'rb') as f:
            label_color_dict = pickle.load(f)
    else:
        label_color_dict = {}

    multimetric_plot(df, title, ylabel, metrics)
    
    with open(color_dict_path, 'wb') as f:
        pickle.dump(label_color_dict, f)


def multimetric_plot(df, title, ylabel, metrics):
    global label_color_dict
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, column in metrics.items():
        if label in label_color_dict:
            color = label_color_dict[label]
            sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
        else:
            line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
            color = line.get_lines()[-1].get_color()
            label_color_dict[label] = color

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(title='Metrics')
    plt.show()
# %%
#############################################
# Correlation
#############################################
# 1. Correlation test
def test_correlation(df, target_columns, new_column_names):
    # Calculate correlation matrix
    correlation_matrix = df[target_columns].corr()
    
    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = new_column_names
    correlation_matrix.index = new_column_names
    
    return correlation_matrix

# 2. Single matrix plot
def plot_correlation_matrix(df, title, target_columns, new_column_names):
    # Get the correlation matrix
    corr_matrix = test_correlation(df, target_columns, new_column_names)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
# 3. Correlation matrix for metrics with and without outlier
''' requires both data with and without outlier to be loaded'''

def plot_correlation_matrices(df1, df2, title1, title2):
    # Get the correlation matrices
    corr_matrix1 = test_correlation(df1)
    corr_matrix2 = test_correlation(df2)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first correlation matrix
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[0])
    axes[0].set_title(title1)
    
    # Plot the second correlation matrix
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[1])
    axes[1].set_title(title2)
    
    plt.tight_layout()
    plt.show()

############################################################
# multimetric extended for facet plot for subsamples of data
############################################################
def multimetric_ss_plot(dict, title, ylabel, metrics):
    global label_color_dict  # Access the global color dictionary
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Determine the number of subplots based on the number of Gruppe values
    n_subplots = len(dict)
    n_cols = min(3, n_subplots)  # Maximum 3 columns
    n_rows = (n_subplots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    for idx, (gruppe, df) in enumerate(dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot each metric for this Gruppe
        for label, column in metrics.items():
            if label in label_color_dict:
                color = label_color_dict[label]
                sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
            else:
                line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
                color = line.get_lines()[-1].get_color()
                label_color_dict[label] = color
        
        ax.set_title(f'Gruppe: {gruppe}')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.legend(title='Metrics', loc='best')
    
    # Remove any unused subplots
    for idx in range(n_subplots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()



