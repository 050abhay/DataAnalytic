import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define the text content as a list of sections
text_content = [
    {"cell_type": "markdown", "content": "# 1. What is a DataFrame?\nA DataFrame is a two-dimensional, size-mutable, and heterogeneous tabular data structure with labeled axes (rows and columns). It’s similar to a spreadsheet or SQL table and is one of the most versatile structures in Pandas."},
    
    {"cell_type": "markdown", "content": "## 2. Creating a DataFrame\nDataFrames can be created in several ways:"},
    
    {"cell_type": "markdown", "content": "### From a Dictionary:\nYou can create a DataFrame using a dictionary where the keys are column names and the values are lists or arrays."},
    
    {"cell_type": "code", "content": """import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)"""},

    {"cell_type": "markdown", "content": "### From a List of Dictionaries:\nAnother common way to create a DataFrame is from a list of dictionaries."},

    {"cell_type": "code", "content": """data = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Chicago'}
]

df = pd.DataFrame(data)"""},

    {"cell_type": "markdown", "content": "### From a CSV or Excel File:\nYou can load data directly from files into a DataFrame using `pd.read_csv()` or `pd.read_excel()`."},

    {"cell_type": "code", "content": "df = pd.read_csv('file.csv')"},

    {"cell_type": "markdown", "content": "### From NumPy Arrays:\nDataFrames can also be created from NumPy arrays."},

    {"cell_type": "code", "content": """import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])"""},

    {"cell_type": "markdown", "content": "## 3. Indexing and Selecting Data\nPandas provides powerful ways to access and manipulate data:"},

    {"cell_type": "markdown", "content": "### .loc[]: Access rows and columns by labels."},

    {"cell_type": "code", "content": """df.loc[0]  # Accesses the first row
df.loc[:, 'Age']  # Accesses the 'Age' column
df.loc[0, 'Name']  # Accesses the value at the first row of 'Name' column"""},

    {"cell_type": "markdown", "content": "### .iloc[]: Access rows and columns by integer positions."},

    {"cell_type": "code", "content": """df.iloc[0]  # Accesses the first row
df.iloc[:, 1]  # Accesses the second column
df.iloc[0, 1]  # Accesses the value at the first row of the second column"""},

    {"cell_type": "markdown", "content": "### Boolean Indexing: Filtering data based on conditions."},

    {"cell_type": "code", "content": "df[df['Age'] > 30]  # Returns rows where 'Age' is greater than 30"},

    {"cell_type": "markdown", "content": "## 4. DataFrame Operations"},

    {"cell_type": "markdown", "content": "### Add/Delete Columns:"},

    {"cell_type": "code", "content": """df['Salary'] = [50000, 60000, 70000]  # Adding a new column
df.drop('City', axis=1, inplace=True)  # Deleting a column"""},

    {"cell_type": "markdown", "content": "### Add/Delete Rows:"},

    {"cell_type": "code", "content": """df = df.append({'Name': 'David', 'Age': 40, 'City': 'Miami'}, ignore_index=True)  # Adding a row
df.drop(0, axis=0, inplace=True)  # Deleting a row by index"""},

    {"cell_type": "markdown", "content": "### DataFrame Transpose:\nYou can transpose a DataFrame (swap rows and columns)."},

    {"cell_type": "code", "content": "df.T"},

    {"cell_type": "markdown", "content": "## 5. Data Cleaning"},

    {"cell_type": "markdown", "content": "### Handling Missing Data:"},

    {"cell_type": "code", "content": """df.dropna()  # Drops rows with missing values
df.fillna(0)  # Fills missing values with 0
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Fills missing values with the mean"""},

    {"cell_type": "markdown", "content": "### Data Type Conversion:"},

    {"cell_type": "code", "content": "df['Age'] = df['Age'].astype(float)"},

    {"cell_type": "markdown", "content": "### Removing Duplicates:"},

    {"cell_type": "code", "content": "df.drop_duplicates(inplace=True)"},

    {"cell_type": "markdown", "content": "## 6. Merging and Joining DataFrames"},

    {"cell_type": "markdown", "content": "### Concat: Concatenate DataFrames either along rows or columns."},

    {"cell_type": "code", "content": """df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
pd.concat([df1, df2])"""},

    {"cell_type": "markdown", "content": "### Merge: Merge DataFrames based on keys (similar to SQL JOIN)."},

    {"cell_type": "code", "content": """df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
pd.merge(df1, df2, on='key', how='inner')"""},

    {"cell_type": "markdown", "content": "## 7. GroupBy Operations"},

    {"cell_type": "markdown", "content": "### GroupBy: You can group data by one or more columns and then apply aggregate functions."},

    {"cell_type": "code", "content": """df.groupby('City').mean()  # Group by 'City' and calculate the mean
df.groupby('City').agg({'Age': 'mean', 'Salary': 'sum'})  # Apply different aggregations"""},

    {"cell_type": "markdown", "content": "## 8. Pivot Tables\nPivot tables in Pandas are similar to pivot tables in Excel. They allow you to rearrange and summarize data."},

    {"cell_type": "code", "content": "df.pivot_table(values='Age', index='City', columns='Name', aggfunc='mean')"},

    {"cell_type": "markdown", "content": "## 9. Handling Dates"},

    {"cell_type": "markdown", "content": "### Datetime Conversion: Convert a column to datetime."},

    {"cell_type": "code", "content": "df['Date'] = pd.to_datetime(df['Date'])"},

    {"cell_type": "markdown", "content": "### DateTimeIndex: You can use date/time as an index."},

    {"cell_type": "code", "content": "df.set_index('Date', inplace=True)"},

    {"cell_type": "markdown", "content": "### Resampling: Resample time-series data to a different frequency."},

    {"cell_type": "code", "content": "df.resample('M').mean()  # Resample by month and calculate the mean"},

    {"cell_type": "markdown", "content": "## 10. Advanced Indexing"},

    {"cell_type": "markdown", "content": "### MultiIndex: Pandas supports hierarchical indexing to work with high-dimensional data."},

    {"cell_type": "code", "content": """arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
df = pd.DataFrame(np.random.randn(4, 2), index=arrays, columns=['Data1', 'Data2'])"""},

    {"cell_type": "markdown", "content": "### Index Slicing: You can slice DataFrames with MultiIndex."},

    {"cell_type": "code", "content": """df.loc['A']
df.loc[('A', 1):('B', 1)]"""},

    {"cell_type": "markdown", "content": "## 11. Visualization\nPandas integrates well with Matplotlib, allowing for quick visualization."},

    {"cell_type": "markdown", "content": "### Plotting: Simple plots can be created directly from DataFrames."},

    {"cell_type": "code", "content": "df.plot(x='Age', y='Salary', kind='scatter')"},

    {"cell_type": "markdown", "content": "## 12. Performance Optimization"},

    {"cell_type": "markdown", "content": "### Vectorization:\nPandas operations are vectorized, meaning they are applied element-wise, making them very efficient."},

    {"cell_type": "markdown", "content": "### Apply/Map:\nFor custom operations, apply and map are often used, but they can be slower than vectorized operations."},

    {"cell_type": "code", "content": "df['Age'].apply(lambda x: x + 1)"},

    {"cell_type": "markdown", "content": "### Memory Usage:\nYou can check and optimize memory usage by downcasting data types."},

    {"cell_type": "code", "content": "df.memory_usage(deep=True)"},

    {"cell_type": "markdown", "content": "## 14. Common Issues"},

    {"cell_type": "markdown", "content": "### SettingWithCopyWarning:\nThis warning occurs when you're trying to modify a copy of a slice from a DataFrame. It's essential to understand when you're working with views vs. copies."},

    {"cell_type": "code", "content": "df.loc[0, 'Age'] = 26  # Direct assignment avoids this warning"}
]

# Convert each section into markdown or code cells
cells = []
for section in text_content:
    if section["cell_type"] == "code":
        cells.append(nbf.v4.new_code_cell(section["content"]))
    else:
        cells.append(nbf.v4.new_markdown_cell(section["content"]))

# Assign the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('pandas_tutorial3.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
