import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class IndianVolatilityRegimeDetector:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _initialize_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
    def engineer_features(self, prices: pd.Series, returns: pd.Series = None) -> pd.DataFrame:
        if returns is None:
            returns = prices.pct_change().dropna()
        
        features = pd.DataFrame(index=prices.index)
        
        features['vol_5d'] = returns.rolling(5).std()
        features['vol_10d'] = returns.rolling(10).std()
        features['vol_20d'] = returns.rolling(20).std()
        features['vol_60d'] = returns.rolling(60).std()
        
        features['vol_ratio_5_20'] = features['vol_5d'] / features['vol_20d']
        features['vol_ratio_10_60'] = features['vol_10d'] / features['vol_60d']
        
        features['return_mean_20d'] = returns.rolling(20).mean()
        features['return_skew_20d'] = returns.rolling(20).skew()
        features['return_kurt_20d'] = returns.rolling(20).kurt()
        
        features['momentum_5d'] = prices.pct_change(5)
        features['momentum_20d'] = prices.pct_change(20)
        
        features['autocorr_1d'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        features['autocorr_5d'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        features['price_ma_20'] = prices.rolling(20).mean()
        features['price_deviation'] = (prices - features['price_ma_20']) / features['price_ma_20']
        
        features = features.dropna()
        
        return features
    
    def create_regime_labels(self, returns: pd.Series, method: str = 'quantile', 
                            threshold: float = 0.25) -> pd.Series:
        volatility = returns.rolling(20).std()
        
        if method == 'quantile':
            high_vol_threshold = volatility.quantile(1 - threshold)
            low_vol_threshold = volatility.quantile(threshold)
            
            labels = pd.Series(index=volatility.index, dtype=float)
            labels[volatility >= high_vol_threshold] = 1
            labels[volatility <= low_vol_threshold] = 0
            
        elif method == 'threshold':
            median_vol = volatility.median()
            labels = (volatility > median_vol).astype(int)
        
        return labels.dropna()
    
    def train(self, features: pd.DataFrame, labels: pd.Series, 
             test_size: float = 0.2) -> Dict:
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self._initialize_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'feature_importance': feature_importance,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features[self.feature_names])
        return self.model.predict_proba(X_scaled)
    
    def get_trading_signal(self, current_features: pd.DataFrame) -> Dict:
        if not self.is_fitted:
            raise ValueError("Model must be trained before generating signals")
        
        prediction = self.predict(current_features)[0]
        probabilities = self.predict_proba(current_features)[0]
        
        if prediction == 1:
            recommendation = {
                'regime': 'HIGH VOLATILITY',
                'confidence': probabilities[1],
                'strategy': 'Hedge Vega exposure, favor Gamma-neutral spreads',
                'suggested_actions': [
                    'Reduce net Vega exposure',
                    'Implement protective puts',
                    'Consider short volatility spreads',
                    'Tighten stop losses'
                ]
            }
        else:
            recommendation = {
                'regime': 'LOW VOLATILITY',
                'confidence': probabilities[0],
                'strategy': 'Sell volatility, short Vega positions',
                'suggested_actions': [
                    'Sell straddles/strangles',
                    'Increase directional exposure',
                    'Collect theta premium',
                    'Implement calendar spreads'
                ]
            }
        
        return recommendation
    
    def explain_with_shap(self, X_test: np.ndarray):
        if not SHAP_AVAILABLE:
            return None, None
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before explaining")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        
        return explainer, shap_values

if __name__ == "__main__":
    print("Volatility Regime Detector - Example")
    print("=" * 60)
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    returns = np.random.normal(0, 0.01, len(dates))
    high_vol_periods = (dates >= '2020-03-01') & (dates <= '2020-06-01')
    returns[high_vol_periods] = np.random.normal(0, 0.03, high_vol_periods.sum())
    
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    returns_series = pd.Series(returns, index=dates)
    
    print(f"\nData generated: {len(prices)} days of price data")
    print(f"Price range: ₹{prices.min():.2f} - ₹{prices.max():.2f}")
    
    detector = IndianVolatilityRegimeDetector(model_type='random_forest')
    
    print("\nEngineering features...")
    features = detector.engineer_features(prices, returns_series)
    print(f"Created {len(features.columns)} features")
    
    labels = detector.create_regime_labels(returns_series)
    print(f"Created labels: {(labels == 1).sum()} high-vol days, {(labels == 0).sum()} low-vol days")
    
    print("\nTraining model...")
    metrics = detector.train(features, labels)
    
    print(f"\nTraining Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Cross-validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\nTop 5 Important Features:")
    print(metrics['feature_importance'].head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Model training complete!")
