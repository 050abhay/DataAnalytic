import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define the text content with mixed markdown and code
text_content = [
    ("markdown", "# 1. What is a DataFrame?\nA DataFrame is a two-dimensional, size-mutable, and heterogeneous tabular data structure with labeled axes (rows and columns). It’s similar to a spreadsheet or SQL table and is one of the most versatile structures in Pandas."),
    
    ("markdown", "## 2. Creating a DataFrame\nDataFrames can be created in several ways:\n"),
    
    ("markdown", "### From a Dictionary:\nYou can create a DataFrame using a dictionary where the keys are column names and the values are lists or arrays."),
    ("code", """import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
"""),
    
    ("markdown", "### From a List of Dictionaries:\nAnother common way to create a DataFrame is from a list of dictionaries."),
    ("code", """data = [
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'Los Angeles'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Chicago'}
]

df = pd.DataFrame(data)
"""),
    
    ("markdown", "### From a CSV or Excel File:\nYou can load data directly from files into a DataFrame using `pd.read_csv()` or `pd.read_excel()`."),
    ("code", "df = pd.read_csv('file.csv')"),
    
    ("markdown", "### From NumPy Arrays:\nDataFrames can also be created from NumPy arrays."),
    ("code", """import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
"""),
    
    ("markdown", "## 3. Indexing and Selecting Data\nPandas provides powerful ways to access and manipulate data:\n"),
    
    ("markdown", "### .loc[]: Access rows and columns by labels."),
    ("code", """df.loc[0]  # Accesses the first row
df.loc[:, 'Age']  # Accesses the 'Age' column
df.loc[0, 'Name']  # Accesses the value at the first row of 'Name' column
"""),
    
    ("markdown", "### .iloc[]: Access rows and columns by integer positions."),
    ("code", """df.iloc[0]  # Accesses the first row
df.iloc[:, 1]  # Accesses the second column
df.iloc[0, 1]  # Accesses the value at the first row of the second column
"""),
    
    ("markdown", "### Boolean Indexing: Filtering data based on conditions."),
    ("code", "df[df['Age'] > 30]  # Returns rows where 'Age' is greater than 30"),
    
    ("markdown", "## 4. DataFrame Operations"),
    
    ("markdown", "### Add/Delete Columns:"),
    ("code", """df['Salary'] = [50000, 60000, 70000]  # Adding a new column
df.drop('City', axis=1, inplace=True)  # Deleting a column
"""),
    
    ("markdown", "### Add/Delete Rows:"),
    ("code", """df = df.append({'Name': 'David', 'Age': 40, 'City': 'Miami'}, ignore_index=True)  # Adding a row
df.drop(0, axis=0, inplace=True)  # Deleting a row by index
"""),
    
    ("markdown", "### DataFrame Transpose:\nYou can transpose a DataFrame (swap rows and columns)."),
    ("code", "df.T"),
    
    ("markdown", "## 5. Data Cleaning"),
    
    ("markdown", "### Handling Missing Data:"),
    ("code", """df.dropna()  # Drops rows with missing values
df.fillna(0)  # Fills missing values with 0
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Fills missing values with the mean
"""),
    
    ("markdown", "### Data Type Conversion:"),
    ("code", "df['Age'] = df['Age'].astype(float)"),
    
    ("markdown", "### Removing Duplicates:"),
    ("code", "df.drop_duplicates(inplace=True)"),
    
    ("markdown", "## 6. Merging and Joining DataFrames"),
    
    ("markdown", "### Concat: Concatenate DataFrames either along rows or columns."),
    ("code", """df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
pd.concat([df1, df2])
"""),
    
    ("markdown", "### Merge: Merge DataFrames based on keys (similar to SQL JOIN)."),
    ("code", """df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
pd.merge(df1, df2, on='key', how='inner')
"""),
    
    ("markdown", "## 7. GroupBy Operations"),
    
    ("markdown", "### GroupBy: You can group data by one or more columns and then apply aggregate functions."),
    ("code", """df.groupby('City').mean()  # Group by 'City' and calculate the mean
df.groupby('City').agg({'Age': 'mean', 'Salary': 'sum'})  # Apply different aggregations
"""),
    
    ("markdown", "## 8. Pivot Tables\nPivot tables in Pandas are similar to pivot tables in Excel. They allow you to rearrange and summarize data."),
    ("code", "df.pivot_table(values='Age', index='City', columns='Name', aggfunc='mean')"),
    
    ("markdown", "## 9. Handling Dates"),
    
    ("markdown", "### Datetime Conversion: Convert a column to datetime."),
    ("code", "df['Date'] = pd.to_datetime(df['Date'])"),
    
    ("markdown", "### DateTimeIndex: You can use date/time as an index."),
    ("code", "df.set_index('Date', inplace=True)"),
    
    ("markdown", "### Resampling: Resample time-series data to a different frequency."),
    ("code", "df.resample('M').mean()  # Resample by month and calculate the mean"),
    
    ("markdown", "## 10. Advanced Indexing"),
    
    ("markdown", "### MultiIndex: Pandas supports hierarchical indexing to work with high-dimensional data."),
    ("code", """arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
df = pd.DataFrame(np.random.randn(4, 2), index=arrays, columns=['Data1', 'Data2'])
"""),
    
    ("markdown", "### Index Slicing: You can slice DataFrames with MultiIndex."),
    ("code", """df.loc['A']
df.loc[('A', 1):('B', 1)]
"""),
    
    ("markdown", "## 11. Visualization"),
    
    ("markdown", "### Plotting: Simple plots can be created directly from DataFrames."),
    ("code", "df.plot(x='Age', y='Salary', kind='scatter')"),
    
    ("markdown", "## 12. Performance Optimization"),
    
    ("markdown", "### Vectorization:\nPandas operations are vectorized, meaning they are applied element-wise, making them very efficient."),
    
    ("markdown", "### Apply/Map:\nFor custom operations, apply and map are often used, but they can be slower than vectorized operations."),
    ("code", "df['Age'].apply(lambda x: x + 1)"),
    
    ("markdown", "### Memory Usage:\nYou can check and optimize memory usage by downcasting data types."),
    ("code", "df.memory_usage(deep=True)"),
    
    ("markdown", "## 14. Common Issues"),
    
    ("markdown", "### SettingWithCopyWarning:\nThis warning occurs when you're trying to modify a copy of a slice from a DataFrame. It's essential to understand when you're working with views vs. copies."),
    ("code", "df.loc[0, 'Age'] = 26  # Direct assignment avoids this warning")
]

# Convert each section into markdown or code cells
cells = []
for cell_type, content in text_content:
    if cell_type == "code":
        # Add a code cell
        cells.append(nbf.v4.new_code_cell(content.strip()))
    else:
        # Add a markdown cell
        cells.append(nbf.v4.new_markdown_cell(content))

# Assign the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file with UTF-8 encoding
with open('pandas_tutorial4.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
