
# Base Model Decision Tree Classification - Steps to Follow

## Step-by-Step Explanation

1. **Choose a Target Attribute**
   - In decision trees, we start by selecting a **target attribute**, which is the attribute (feature) that the tree will use to decide the best splits for classification. 
   - For example, if we’re predicting whether an email is spam or not, the target attribute would be the final classification (spam or not spam).

2. **Calculate Information Gain (IG)**
   - **Information Gain** helps in determining which attribute to split on at each step in the tree. It measures the reduction in entropy (or impurity) when a dataset is split based on a particular attribute.
   - The formula for **Information Gain** of the target attribute is:
$$
     [
     -\frac{P}{P+N} \cdot \log_2 \left( \frac{P}{P+N} \right) - \frac{N}{N+P} \cdot \log_2 \left( \frac{N}{N+P} \right)
     ]
$$
     - Here:
       - \( P \): The number of positive instances (e.g., genuine orders).
       - \( N \): The number of negative instances (e.g., fake orders).
     - This formula calculates the entropy of the target attribute, which measures the uncertainty or impurity in the target. Lower entropy values indicate more pure splits.

3. **Calculate Entropy of Other Attributes**
   - For each attribute (feature) in the dataset, calculate the entropy to determine its contribution to classification.
   - The formula given is:
$$
     [
     E(A) = \sum_{i=1}^{n} \frac{P_i + N_i}{P + N} \cdot I(P_i N_i)
     ]
$$
     - Here:
       - $ P_i $ and $ N_i $ represent the number of positive and negative examples for each unique value of attribute \( A \).
       - The sum iterates over all values of the attribute \( A \).
       - $ I(P_i, N_i)$ calculates the entropy of each subset created by splitting on \( A \).
   - This gives the average entropy across all subsets of the attribute, helping us understand how well this attribute contributes to reducing uncertainty in the target.

4. **Calculate the Gain**
   - To determine if an attribute is a good candidate for a split, calculate its **Gain** by subtracting the entropy of the attribute \( E(A) \) from the Information Gain of the target attribute.
   - The attribute with the highest Gain is chosen as the split point since it provides the most information about the target and reduces entropy the most.

## Summary
This process of calculating Information Gain and Entropy helps build the decision tree by selecting the attribute at each node that provides the best split (i.e., the most information) for classifying the target attribute. This way, the decision tree grows by iteratively reducing uncertainty (entropy) and maximizing the clarity (purity) of each split.
