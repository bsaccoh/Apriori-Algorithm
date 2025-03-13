import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess the dataset
transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Visualize frequent itemsets
plt.figure(figsize=(10, 6))
sns.barplot(x='support', y='itemsets', data=frequent_itemsets, hue='itemsets', palette='viridis', legend=False)
plt.title('Frequent Itemsets (Support >= 0.2)')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()

# Visualize association rules (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules, hue='lift', palette='viridis', sizes=(20, 200))
plt.title('Association Rules (Support vs Confidence)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Visualize association rules (Heatmap)
pivot_table = rules.pivot(index='antecedents', columns='consequents', values='confidence')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f')
plt.title('Association Rules Heatmap (Confidence)')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.show()