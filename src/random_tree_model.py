"""
Random Tree & Random Forest Machine Learning Classifier implementation
Includes pure Python Decision Tree & Random Forest Classifier algorithms
with Gini Impurity, feature sampling, tree depth visualizer export, and inference methods.
"""

import math
import random
import json
import numpy as np

class DecisionTreeNode:
    """Represents a single decision node or leaf node in a Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gini=0.0, samples=0):
        self.feature = feature        # Feature name or index to split on
        self.threshold = threshold    # Value threshold for continuous or target level for categorical
        self.left = left              # Left subtree (True/<= condition)
        self.right = right            # Right subtree (False/> condition)
        self.value = value            # Class probability / prediction if leaf node
        self.gini = gini              # Gini impurity score at node
        self.samples = samples        # Number of samples reaching node

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifierCustom:
    """Custom Decision Tree Classifier implementing Gini Impurity splits."""
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.feature_importances_ = {}

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return 1.0 - (p0**2 + p1**2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_gini = self._gini(y)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Sample thresholds if too many to keep training fast
            if len(thresholds) > 50:
                thresholds = np.random.choice(thresholds, size=50, replace=False)

            for thresh in thresholds:
                left_idxs, right_idxs = self._split(X_column, thresh)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate Gini Information Gain
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
                child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

                gain = current_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_val = np.mean(y) if n_samples > 0 else 0.0
            return DecisionTreeNode(value=leaf_val, gini=self._gini(y), samples=n_samples)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find best split
        best_feat, best_thresh, gain = self._best_split(X, y, feat_idxs)

        if best_gain_is_invalid := (best_feat is None or gain <= 0):
            leaf_val = np.mean(y) if n_samples > 0 else 0.0
            return DecisionTreeNode(value=leaf_val, gini=self._gini(y), samples=n_samples)

        # Track feature importance
        if best_feat not in self.feature_importances_:
            self.feature_importances_[best_feat] = 0.0
        self.feature_importances_[best_feat] += gain * n_samples

        # Split left and right
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return DecisionTreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left,
            right=right,
            gini=self._gini(y),
            samples=n_samples
        )

    def fit(self, X, y):
        n_feats = X.shape[1]
        self.n_features = n_feats if not self.n_features else min(n_feats, self.n_features)
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

class RandomTreeForestClassifier:
    """Random Forest (Ensemble of Random Decision Trees) Classifier."""
    def __init__(self, n_trees=15, min_samples_split=2, max_depth=8, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        n_feats = X.shape[1]
        max_f = int(math.sqrt(n_feats)) if not self.n_features else self.n_features

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifierCustom(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=max_f
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

def export_tree_to_json(node, feature_names):
    """Recursively export decision tree node to JSON dict structure for frontend visualization."""
    if node.is_leaf_node():
        churn_risk = "High" if node.value >= 0.5 else "Low"
        return {
            "name": f"Leaf: {churn_risk} Risk ({node.value*100:.1f}% Churn)",
            "type": "leaf",
            "probability": round(float(node.value), 4),
            "samples": int(node.samples),
            "gini": round(float(node.gini), 3)
        }

    feat_name = feature_names[node.feature] if feature_names and node.feature < len(feature_names) else f"Feature_{node.feature}"
    return {
        "name": f"Split: {feat_name} <= {node.threshold:.2f}",
        "feature": feat_name,
        "threshold": float(node.threshold),
        "type": "split",
        "gini": round(float(node.gini), 3),
        "samples": int(node.samples),
        "children": [
            export_tree_to_json(node.left, feature_names),
            export_tree_to_json(node.right, feature_names)
        ]
    }
