import numpy as np
import pandas as pd

class DecisionTree:
    def _init_(self, max_depth=None):
        self.max_depth = max_depth
        
    def entropy(self, labels):
        """Calculate the entropy of a set of labels."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def information_gain(self, data, feature_name, target_name):
        """Calculate the information gain for a given feature."""
        total_entropy = self.entropy(data[target_name])
        
        # Calculate the weighted entropy after splitting on the feature
        weighted_entropy = 0
        unique_values = data[feature_name].unique()
        for value in unique_values:
            subset = data[data[feature_name] == value]
            subset_entropy = self.entropy(subset[target_name])
            weight = len(subset) / len(data)
            weighted_entropy += weight * subset_entropy
        
        # Calculate information gain
        information_gain = total_entropy - weighted_entropy
        return information_gain
    
    def find_best_split(self, data, target_name):
        """Find the best feature to split on."""
        best_gain = 0
        best_feature = None
        for feature in data.columns:
            if feature != target_name:
                gain = self.information_gain(data, feature, target_name)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
        return best_feature
    
    def split_data(self, data, feature_name, value):
        """Split data based on a given feature and value."""
        return data[data[feature_name] == value]
    
    def majority_vote(self, labels):
        """Return the majority class label."""
        return labels.value_counts().idxmax()
    
    def build_tree(self, data, target_name, depth=0):
        """Recursively build the decision tree."""
        # Check for stopping conditions
        if self.max_depth is not None and depth == self.max_depth:
            return self.majority_vote(data[target_name])
        
        if len(data[target_name].unique()) == 1:
            return data[target_name].iloc[0]
        
        # Find the best feature to split on
        best_feature = self.find_best_split(data, target_name)
        
        # Create sub-trees for each unique value of the best feature
        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            subset = self.split_data(data, best_feature, value)
            subtree = self.build_tree(subset, target_name, depth + 1)
            tree[best_feature][value] = subtree
        return tree
    
    def fit(self, data, target_name):
        """Fit the Decision Tree to the training data."""
        self.tree = self.build_tree(data, target_name)
    
    def predict_instance(self, instance, tree):
        """Predict the class label for a single instance."""
        if isinstance(tree, str):
            return tree
        elif isinstance(tree, np.int64):  # Handle leaf nodes
            return str(tree)
        
        feature = next(iter(tree))
        value = instance[feature]
        
        if value not in tree[feature]:
            return "unknown"
        
        subtree = tree[feature][value]
        return self.predict_instance(instance, subtree)
    
    def predict(self, data):
        """Predict class labels for a dataset."""
        predictions = []
        for _, instance in data.iterrows():
            prediction = self.predict_instance(instance, self.tree)
            predictions.append(prediction)
        return pd.Series(predictions)

# Load data from CSV
file_path = r'A:\sem4\ML\Unemployment_in_India.csv'
data = pd.read_csv(file_path)

# Initialize and train the Decision Tree model
tree_model = DecisionTree()
target_name = 'status'
tree_model.fit(data, target_name)

# Make predictions
predictions = tree_model.predict(data)
print(predictions)