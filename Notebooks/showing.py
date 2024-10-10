import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Show summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Plot the distribution of the 'age' feature
plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the relationship between 'age' and 'fare'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', data=df)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()