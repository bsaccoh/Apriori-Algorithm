#Import Required Libraries
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Dataset
df = pd.read_csv('data.csv')

# Convert the dataset into a list of transactions
transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# Function to generate candidate itemsets of size k
def generate_candidates(itemsets, k):
    candidates = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            union = itemset1.union(itemset2)
            if len(union) == k:
                candidates.add(union)
    return candidates

# Function to prune infrequent itemsets
def prune_itemsets(candidates, transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                item_counts[candidate] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {
        itemset: support / num_transactions
        for itemset, support in item_counts.items()
        if support / num_transactions >= min_support
    }
    return frequent_itemsets

# Function to find all frequent itemsets
def apriori(transactions, min_support):
    # Generate frequent 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {
        itemset: support / num_transactions
        for itemset, support in item_counts.items()
        if support / num_transactions >= min_support
    }

    # Generate frequent k-itemsets (k > 1)
    k = 2
    while True:
        candidates = generate_candidates(frequent_itemsets.keys(), k)
        if not candidates:
            break

        new_frequent_itemsets = prune_itemsets(candidates, transactions, min_support)
        if not new_frequent_itemsets:
            break

        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    return frequent_itemsets

# Find Frequent Itemsets
min_support = 0.2
frequent_itemsets = apriori(transactions, min_support)

# Convert frequent itemsets to a DataFrame for better visualization
frequent_itemsets_df = pd.DataFrame(
    [(itemset, support) for itemset, support in frequent_itemsets.items()],
    columns=['Itemset', 'Support']
)

# Generate Association Rules
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            for antecedent in itertools.combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules

min_confidence = 0.7
rules = generate_association_rules(frequent_itemsets, min_confidence)

# Convert rules to a DataFrame for better visualization
rules_df = pd.DataFrame(
    rules,
    columns=['Antecedent', 'Consequent', 'Confidence']
)

# Bar plot for frequent itemsets
plt.figure(figsize=(10, 6))
sns.barplot(x='Support', y='Itemset', data=frequent_itemsets_df, hue='Itemset', palette='viridis', legend=False)
plt.title(f'Frequent Itemsets (Support >= {min_support})')
plt.xlabel('Support')
plt.ylabel('Itemset')
plt.show()

# Scatter plot for association rules
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=[frequent_itemsets[rule[0]] for rule in rules],  # Support of antecedent
    y=[rule[2] for rule in rules],  # Confidence
    size=[frequent_itemsets[rule[0].union(rule[1])] for rule in rules],  # Support of rule
    hue=[frequent_itemsets[rule[0].union(rule[1])] for rule in rules],  # Lift (approximated)
    palette='viridis',
    sizes=(20, 200)
)
plt.title(f'Association Rules (Confidence >= {min_confidence})')
plt.xlabel('Support of Antecedent')
plt.ylabel('Confidence')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 8: Display Results
print("Frequent Itemsets:")
print(frequent_itemsets_df)

print("\nAssociation Rules:")
print(rules_df)