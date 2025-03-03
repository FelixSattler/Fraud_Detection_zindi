import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt


def explore(df): 
    """ 
    Perform basic exploratory data analysis (EDA) on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to explore.

    Returns:
    None (prints dataset info, statistics, and missing values overview).
    """

    #info
    print('General information:')
    df.info()

    #first 5 rows
    print(f'\n First 5 rows: \n {df.head()}')

    #describe
    print(f'\n Descriptive statistics: \n {df.describe()}')

    #shape
    print(f'\n DF shape:   {df.shape}')

    #unique values
    print(f'\n Number of unique values: \n {df.nunique()}')
    
    #duplicates
    print(f'\n Sum of duplicates: {df.duplicated().sum()}\n') 

    #missing values
    sum_missing = df.isna().sum()
    perc_missing = (sum_missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': sum_missing.index,
        'Sum of NaNs': sum_missing.values,
        'Perc of NaNs': perc_missing.values
    })

    print(f'Missing values: \n {missing_df.to_string(index=False, float_format="%.2f")}')

    # Visualize missing values only if there are missing values
    if sum_missing.sum() > 0:
        msno.matrix(df)
        plt.title("Missing data by column", fontsize=20, fontweight="bold")
        plt.show()

