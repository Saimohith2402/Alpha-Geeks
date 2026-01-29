import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from indian_bs_model import IndianBlackScholesModel

class IndianOptionLeg:
    def __init__(self, S: float, K: float, days_to_expiry: int, r: float, sigma: float, 
                 option_type: str, position: str, quantity: int = 1):
        self.model = IndianBlackScholesModel(S, K, days_to_expiry, r, sigma, option_type)
        self.position = position.lower()
        self.quantity = quantity
        self.multiplier = 1 if position.lower() == 'long' else -1
        
    def get_greeks(self) -> Dict[str, float]:
        greeks = self.model.all_greeks()
        return {k: v * self.multiplier * self.quantity for k, v in greeks.items()}

class IndianOptionPortfolio:
    def __init__(self, lot_size: int = 75):
        self.legs = []
        self.lot_size = lot_size
        
    def add_leg(self, leg: IndianOptionLeg):
        self.legs.append(leg)
        
    def clear(self):
        self.legs = []
        
    def net_greeks(self) -> Dict[str, float]:
        if not self.legs:
            return {'price': 0, 'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0, 'days': 0}
        
        net = {'price': 0, 'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0, 'days': 0}
        
        for leg in self.legs:
            leg_greeks = leg.get_greeks()
            for key in net:
                net[key] += leg_greeks[key]
        
        return net
    
    def greeks_by_leg(self) -> pd.DataFrame:
        data = []
        for i, leg in enumerate(self.legs):
            greeks = leg.get_greeks()
            row = {
                'Leg': i + 1,
                'Type': leg.model.option_type.upper(),
                'Strike': f"₹{leg.model.K}",
                'Position': leg.position.upper(),
                'Qty': leg.quantity,
                'Days': leg.model.days,
                'Premium': f"₹{greeks['price']:.2f}",
                'Delta': f"{greeks['delta']:.4f}",
                'Gamma': f"{greeks['gamma']:.6f}",
                'Vega': f"{greeks['vega']:.4f}",
                'Theta/Day': f"₹{greeks['theta']:.2f}",
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        net = self.net_greeks()
        net_row = {
            'Leg': 'NET', 'Type': '-', 'Strike': '-', 'Position': '-', 'Qty': '-', 'Days': '-',
            'Premium': f"₹{net['price']:.2f}",
            'Delta': f"{net['delta']:.4f}",
            'Gamma': f"{net['gamma']:.6f}",
            'Vega': f"{net['vega']:.4f}",
            'Theta/Day': f"₹{net['theta']:.2f}",
        }
        df = pd.concat([df, pd.DataFrame([net_row])], ignore_index=True)
        
        return df
    
    def pnl_simulation(self, spot_range: np.ndarray, vol_change: float = 0) -> Tuple[np.ndarray, pd.DataFrame]:
        initial_value = self.net_greeks()['price']
        pnl = np.zeros_like(spot_range)
        
        details = []
        
        for idx, S_new in enumerate(spot_range):
            portfolio_value = 0
            
            for leg in self.legs:
                new_sigma = leg.model.sigma + vol_change
                new_model = IndianBlackScholesModel(
                    S_new, leg.model.K, leg.model.days, leg.model.r, 
                    new_sigma, leg.model.option_type
                )
                leg_value = new_model.price() * leg.multiplier * leg.quantity
                portfolio_value += leg_value
            
            pnl[idx] = portfolio_value - initial_value
            
            if idx % max(1, len(spot_range) // 10) == 0:
                details.append({
                    'Spot': f"₹{S_new:.0f}",
                    'P&L': f"₹{pnl[idx]:.2f}"
                })
        
        return pnl, pd.DataFrame(details)
    
    def risk_metrics(self) -> Dict[str, any]:
        net = self.net_greeks()
        
        metrics = {
            'net_delta': net['delta'],
            'net_gamma': net['gamma'],
            'net_vega': net['vega'],
            'net_theta': net['theta'],
            'directional_bias': 'Bullish' if net['delta'] > 0.1 else 'Bearish' if net['delta'] < -0.1 else 'Neutral',
            'volatility_exposure': 'Long Vol' if net['vega'] > 0 else 'Short Vol' if net['vega'] < 0 else 'Vol Neutral',
            'daily_pnl_decay': net['theta'],
            'max_profit_1pct': net['delta'] * 100,
            'initial_cost': net['price'],
            'days_to_expiry': net['days']
        }
        
        return metrics

class IndianStrategyBuilder:
    @staticmethod
    def long_straddle(S: float, K: float, days_to_expiry: int, r: float = 0.065, 
                     sigma: float = 0.20, quantity: int = 1) -> IndianOptionPortfolio:
        portfolio = IndianOptionPortfolio()
        portfolio.add_leg(IndianOptionLeg(S, K, days_to_expiry, r, sigma, 'call', 'long', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K, days_to_expiry, r, sigma, 'put', 'long', quantity))
        return portfolio
    
    @staticmethod
    def short_straddle(S: float, K: float, days_to_expiry: int, r: float = 0.065, 
                      sigma: float = 0.20, quantity: int = 1) -> IndianOptionPortfolio:
        portfolio = IndianOptionPortfolio()
        portfolio.add_leg(IndianOptionLeg(S, K, days_to_expiry, r, sigma, 'call', 'short', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K, days_to_expiry, r, sigma, 'put', 'short', quantity))
        return portfolio
    
    @staticmethod
    def bull_call_spread(S: float, K_long: float, K_short: float, days_to_expiry: int, 
                        r: float = 0.065, sigma: float = 0.20, quantity: int = 1) -> IndianOptionPortfolio:
        portfolio = IndianOptionPortfolio()
        portfolio.add_leg(IndianOptionLeg(S, K_long, days_to_expiry, r, sigma, 'call', 'long', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K_short, days_to_expiry, r, sigma, 'call', 'short', quantity))
        return portfolio
    
    @staticmethod
    def bear_put_spread(S: float, K_long: float, K_short: float, days_to_expiry: int, 
                       r: float = 0.065, sigma: float = 0.20, quantity: int = 1) -> IndianOptionPortfolio:
        portfolio = IndianOptionPortfolio()
        portfolio.add_leg(IndianOptionLeg(S, K_short, days_to_expiry, r, sigma, 'put', 'short', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K_long, days_to_expiry, r, sigma, 'put', 'long', quantity))
        return portfolio
    
    @staticmethod
    def iron_condor(S: float, K_put_short: float, K_put_long: float, 
                   K_call_short: float, K_call_long: float, days_to_expiry: int, 
                   r: float = 0.065, sigma: float = 0.20, quantity: int = 1) -> IndianOptionPortfolio:
        portfolio = IndianOptionPortfolio()
        portfolio.add_leg(IndianOptionLeg(S, K_put_short, days_to_expiry, r, sigma, 'put', 'short', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K_put_long, days_to_expiry, r, sigma, 'put', 'long', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K_call_short, days_to_expiry, r, sigma, 'call', 'short', quantity))
        portfolio.add_leg(IndianOptionLeg(S, K_call_long, days_to_expiry, r, sigma, 'call', 'long', quantity))
        return portfolio

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INDIAN PORTFOLIO SIMULATOR - Multi-Leg Strategies for NSE")
    print("=" * 80)
    
    nifty_spot = 23000
    
    strategy = IndianStrategyBuilder.iron_condor(
        S=nifty_spot,
        K_put_short=23000,
        K_put_long=22800,
        K_call_short=23200,
        K_call_long=23400,
        days_to_expiry=7,
        sigma=0.20,
        quantity=1
    )
    
    print(f"\nNifty Spot: ₹{nifty_spot}")
    print(f"Days to Expiry: 7 days (Weekly)")
    print(f"Expected Profit: From time decay & premium")
    
    print("\nPosition Details:")
    print(strategy.greeks_by_leg().to_string(index=False))
    
    print("\n\nRisk Metrics:")
    metrics = strategy.risk_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:>12.4f}")
        else:
            print(f"{key:.<30} {value:>12}")
    
    print("\n\nP&L at Different Nifty Levels:")
    spot_range = np.linspace(22600, 23400, 21)
    pnl, details = strategy.pnl_simulation(spot_range)
    print(details.to_string(index=False))
    
    print("\n" + "=" * 80)
