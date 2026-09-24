
import numpy as np
from sklearn.base import BaseEstimator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConformalPolicy(ABC):
    """Abstract base class for all conformal policies"""
    @abstractmethod
    def calibrate(self, scores_calib, X_calib=None):
        """Calibrate the policy on calibration scores"""
        pass
    
    @abstractmethod
    def get_threshold(self, x_test, calibration_summary):
        """Get prediction threshold for a test point"""
        pass
    
    @abstractmethod
    def predict_set(self, base_model, x_test, threshold):
        """Generate prediction set given threshold"""
        pass

class FixedAlphaPolicy(ConformalPolicy):
    """Standard conformal prediction with fixed alpha"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.threshold = None
    
    def calibrate(self, scores_calib, X_calib=None):
        n = len(scores_calib)
        self.threshold = np.quantile(
            scores_calib, 
            np.ceil((n + 1) * (1 - self.alpha)) / n
        )
        return self
    
    def get_threshold(self, x_test, calibration_summary):
        # Fixed threshold for all test points
        return self.threshold
    
    def predict_set(self, base_model, x_test, threshold):
        # Implementation depends on score type
        # For regression with absolute error:
        prediction = base_model.predict(x_test.reshape(1, -1))[0]
        return [prediction - threshold, prediction + threshold]


class AlphaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.initialize_to_one()
    
    def initialize_to_one(self):
        with torch.no_grad():
            self.fc2.weight.data.normal_(0, 0.01)
            self.fc2.bias.data.fill_(5.0)  # sigmoid(5) ≈ 0.993
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x)).squeeze(-1)


class ConformalPredictor(BaseEstimator, RegressorMixin):
    """Main sklearn-compatible class with pluggable policies"""
    
    def __init__(self, 
                 base_model,
                 score_func=None,
                 policy="fixed_alpha",
                 **policy_kwargs):
        """
        Args:
            base_model: sklearn-compatible model
            score_func: function computing nonconformity scores
            policy: one of ["fixed_alpha", "epicscore", "adaptive", "combined"]
            policy_kwargs: passed to policy constructor
        """
        self.base_model = base_model
        self.score_func = score_func or self._default_score_func
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs
        
        # Initialize selected policy
        self.policy = self._create_policy(policy, policy_kwargs)
        
        # Calibration data storage
        self.X_calib_ = None
        self.scores_calib_ = None

    def _create_policy(self, policy_name, kwargs):
        """Factory method for creating policies"""
        policies = {
            "fixed_alpha": FixedAlphaPolicy,
           # "epicscore": EPICSCOREPolicy,
            #"adaptive": AdaptivePolicy,
            #"combined": CombinedPolicy,
        }
        
        if policy_name not in policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        return policies[policy_name](**kwargs)
    
    def _default_score_func(self, y_true, y_pred):
        """Default absolute error score for regression"""
        return np.abs(y_true - y_pred)
    
    def fit(self, X_train, y_train):
        """Fit the base model only"""
        self.base_model.fit(X_train, y_train)
        return self
    
    def calibrate(self, X_calib, y_calib):
        """Calibrate the conformal policy"""
        self.X_calib_ = X_calib
        
        # Compute calibration scores
        y_pred = self.base_model.predict(X_calib)
        self.scores_calib_ = self.score_func(y_calib, y_pred)
        
        # Calibrate the selected policy
        self.policy.calibrate(self.scores_calib_, X_calib)
        return self
    
    def predict(self, X_test):
        """Point predictions (delegate to base model)"""
        return self.base_model.predict(X_test)
    
    def predict_interval(self, X_test):
        """Get prediction intervals with conformal guarantee"""
        intervals = []
        
        calibration_summary = {
            'scores': self.scores_calib_,
            'X_calib': self.X_calib_,
        }
        
        for x in X_test:
            # Get dynamic threshold from policy
            threshold = self.policy.get_threshold(x, calibration_summary)
            
            # Generate prediction set
            interval = self.policy.predict_set(self.base_model, x, threshold)
            intervals.append(interval)
        
        return np.array(intervals)
    
    def score(self, X, y):
        """Can define custom scoring for conformal prediction"""
        intervals = self.predict_interval(X)
        coverage = np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1]))
        avg_width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        # Return tuple or custom score object
        return {"coverage": coverage, "avg_width": avg_width}
    
    def set_policy(self, policy_name, **kwargs):
        """Dynamically change policy"""
        self.policy_type = policy_name
        self.policy_kwargs.update(kwargs)
        self.policy = self._create_policy(policy_name, self.policy_kwargs)
        
        # Re-calibrate if we already have calibration data
        if self.X_calib_ is not None:
            self.calibrate(self.X_calib_, self._get_y_calib())
        return self
