"""Traditional machine learning models"""

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class TraditionalMLModels:
    """Traditional ML models for classification"""
    
    @staticmethod
    def get_svm(kernel='linear', C=1.0):
        """Get SVM classifier"""
        if kernel == 'linear':
            return LinearSVC(C=C, max_iter=1000, random_state=42)
        else:
            return SVC(kernel=kernel, C=C, random_state=42)
    
    @staticmethod
    def get_random_forest(n_estimators=100, max_depth=None):
        """Get Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def get_decision_tree(max_depth=None):
        """Get Decision Tree classifier"""
        return DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
    
    @staticmethod
    def get_gradient_boosting(n_estimators=100, learning_rate=0.1):
        """Get Gradient Boosting classifier"""
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )


if __name__ == '__main__':
    # Test model creation
    svm_model = TraditionalMLModels.get_svm()
    rf_model = TraditionalMLModels.get_random_forest()
    print("Traditional ML models created successfully")
