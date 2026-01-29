import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional

class IndianBlackScholesModel:
    def __init__(self, S: float, K: float, days_to_expiry: int, r: float = 0.065, 
                 sigma: float = 0.25, option_type: str = 'call'):
        self.S = S
        self.K = K
        self.days_to_expiry = days_to_expiry
        self.T = days_to_expiry / 365.0
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.days = days_to_expiry
        self._validate_inputs()
        
    def _validate_inputs(self):
        if self.S <= 0:
            raise ValueError("Price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.days <= 0:
            raise ValueError("Days to expiry must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def _calculate_d1_d2(self) -> Tuple[float, float]:
        if self.T < 0.001:
            return (np.inf if self.S > self.K else -np.inf, 
                    np.inf if self.S > self.K else -np.inf)
        
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def price(self) -> float:
        if self.T < 0.001:
            if self.option_type == 'call':
                return max(self.S - self.K, 0)
            else:
                return max(self.K - self.S, 0)
        
        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return price
    
    def delta(self) -> float:
        if self.T < 0.001:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1, _ = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self) -> float:
        if self.T < 0.001:
            return 0.0
        
        d1, _ = self._calculate_d1_d2()
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def vega(self) -> float:
        if self.T < 0.001:
            return 0.0
        
        d1, _ = self._calculate_d1_d2()
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T) / 100
        return vega
    
    def theta(self) -> float:
        if self.T < 0.001:
            return 0.0
        
        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return theta / 365
    
    def rho(self) -> float:
        if self.T < 0.001:
            return 0.0
        
        _, d2 = self._calculate_d1_d2()
        
        if self.option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        
        return rho
    
    def all_greeks(self) -> Dict[str, float]:
        return {
            'price': self.price(),
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho(),
            'days': self.days
        }
    
    def validate_greeks_numerical(self, epsilon: float = 1e-4) -> Dict[str, Dict[str, float]]:
        results = {}
        
        dS = 0.01 * self.S
        model_up = IndianBlackScholesModel(self.S + dS, self.K, self.days, self.r, self.sigma, self.option_type)
        model_down = IndianBlackScholesModel(self.S - dS, self.K, self.days, self.r, self.sigma, self.option_type)
        delta_numerical = (model_up.price() - model_down.price()) / (2 * dS)
        delta_analytical = self.delta()
        
        results['delta'] = {
            'analytical': delta_analytical,
            'numerical': delta_numerical,
            'error': abs(delta_analytical - delta_numerical),
            'pass': abs(delta_analytical - delta_numerical) < epsilon
        }
        
        gamma_numerical = (model_up.delta() - model_down.delta()) / (2 * dS)
        gamma_analytical = self.gamma()
        
        results['gamma'] = {
            'analytical': gamma_analytical,
            'numerical': gamma_numerical,
            'error': abs(gamma_analytical - gamma_numerical),
            'pass': abs(gamma_analytical - gamma_numerical) < epsilon
        }
        
        dsigma = 0.001
        model_vol_up = IndianBlackScholesModel(self.S, self.K, self.days, self.r, self.sigma + dsigma, self.option_type)
        model_vol_down = IndianBlackScholesModel(self.S, self.K, self.days, self.r, self.sigma - dsigma, self.option_type)
        vega_numerical = (model_vol_up.price() - model_vol_down.price()) / (2 * dsigma) / 100
        vega_analytical = self.vega()
        
        results['vega'] = {
            'analytical': vega_analytical,
            'numerical': vega_numerical,
            'error': abs(vega_analytical - vega_numerical),
            'pass': abs(vega_analytical - vega_numerical) < epsilon
        }
        
        return results

INDIAN_INDICES = {
    'NIFTY': {'symbol': '^NSEI', 'name': 'Nifty 50', 'expiry': 'Weekly & Monthly'},
    'BANKNIFTY': {'symbol': '^NSEBANK', 'name': 'Bank Nifty', 'expiry': 'Weekly & Monthly'},
    'SENSEX': {'symbol': '^BSESN', 'name': 'BSE Sensex', 'expiry': 'Weekly'},
    'FINNIFTY': {'symbol': 'FINNIFTY.NS', 'name': 'Finnifty', 'expiry': 'Weekly & Monthly'},
    'MIDCPNIFTY': {'symbol': 'MIDCPNIFTY.NS', 'name': 'Midcap Nifty', 'expiry': 'Monthly'},
}

def get_next_expiry(expiry_type: str) -> int:
    if expiry_type == 'weekly':
        return 7
    elif expiry_type == 'monthly':
        return 28
    else:
        return 1

if __name__ == "__main__":
    print("=" * 70)
    print("Indian Option Greeks Calculator - NSE Optimized")
    print("=" * 70)
    
    spot_price = 23000
    strike = 23000
    days_to_expiry = 7
    
    call = IndianBlackScholesModel(spot_price, strike, days_to_expiry, 
                                   r=0.065, sigma=0.20, option_type='call')
    
    greeks = call.all_greeks()
    
    print(f"\nSpot Price: ₹{spot_price}")
    print(f"Strike Price: ₹{strike}")
    print(f"Days to Expiry: {days_to_expiry} days")
    print(f"Volatility: 20%")
    print(f"\n{'Metric':<15} {'Value':<20} {'Interpretation':<30}")
    print("-" * 70)
    print(f"{'Price':<15} ₹{greeks['price']:<18.2f} {'Premium to pay/receive':<30}")
    print(f"{'Delta':<15} {greeks['delta']:<20.4f} {'Move ₹1 = Δ ₹{:.2f}'.format(greeks['delta']):<30}")
    print(f"{'Gamma':<15} {greeks['gamma']:<20.6f} {'Delta change per ₹1':<30}")
    print(f"{'Vega':<15} {greeks['vega']:<20.4f} {'Per 1% vol change':<30}")
    print(f"{'Theta':<15} ₹{greeks['theta']:<18.4f} {'Daily time decay (loss)':<30}")
    print(f"{'Rho':<15} {greeks['rho']:<20.4f} {'Per 1% rate change':<30}")
