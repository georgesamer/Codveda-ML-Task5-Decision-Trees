import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, f1_score, classification_report)
import matplotlib.pyplot as plt
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

class DecisionTreeAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, filepath, target_column, test_size=0.2, stratify=True):
        try:
            # Load data
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File {filepath} not found")
                
            self.df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"Target distribution:\n{self.df[target_column].value_counts()}")
            
            # Prepare
            self.X = self.df.drop(target_column, axis=1)
            self.y = self.df[target_column]
            self.feature_names = self.X.columns.tolist()
            self.class_names = sorted(self.y.unique())
            
            # Split
            stratify_param = self.y if stratify else None
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_param
            )
            
            print(f"Training set: {self.X_train.shape}")
            print(f"Test set: {self.X_test.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def train_basic_model(self, model_name="basic"):
        """Train a basic decision tree without pruning."""
        clf = DecisionTreeClassifier(random_state=self.random_state)
        clf.fit(self.X_train, self.y_train)
        self.models[model_name] = clf
        
        # Cross-validation
        cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"{model_name.title()} Model - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return clf
    
    def train_pruned_model(self, model_name="pruned", **params):
        """Train a pruned decision tree with specified parameters."""
        default_params = {'max_depth': 3, 'min_samples_split': 5, 'min_samples_leaf': 2}
        default_params.update(params)
        
        clf = DecisionTreeClassifier(random_state=self.random_state, **default_params)
        clf.fit(self.X_train, self.y_train)
        self.models[model_name] = clf
        
        # Cross-validation
        cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"{model_name.title()} Model - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return clf
    
#evaluation_before_pruning
    def evaluate_model(self, model_name):
        """evaluate the model on the test set"""
        clf = self.models[model_name]   # call the model
        y_pred = clf.predict(self.X_test)   # using the model to predict the test set
        
        print(f"\n=== Evaluation: {model_name.title()} ===")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("F1 Score (macro):", f1_score(self.y_test, y_pred, average="macro"))
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))

    def plot_tree_visualization(self, model_name):
        """draw the decision tree"""
        clf = self.models[model_name]
        plt.figure(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=self.feature_names, class_names=self.class_names)
        plt.title(f"Decision Tree - {model_name}")
        plt.show()

def main():
    analyzer = DecisionTreeAnalyzer(random_state=42)
    
    try:
        # Load data
        analyzer.load_data("iris.csv", target_column="species", test_size=0.2)
        
        # Train models
        analyzer.train_basic_model("basic")
        analyzer.train_pruned_model("pruned", max_depth=3)
        
        # Evaluate
        analyzer.evaluate_model("basic")
        analyzer.evaluate_model("pruned")
        
        # Visualize
        analyzer.plot_tree_visualization("basic")
        analyzer.plot_tree_visualization("pruned")
        
    except FileNotFoundError:
        print("iris.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
