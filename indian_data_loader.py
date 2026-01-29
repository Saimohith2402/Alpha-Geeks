import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class IndianMarketDataLoader:
    INDICES_TICKERS = {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK',
        'SENSEX': '^BSESN',
        'FINNIFTY': 'FINNIFTY.NS',
        'MIDCPNIFTY': 'MIDCPNIFTY.NS',
    }
    
    EXPIRY_INFO = {
        'NIFTY': {'weekly': 7, 'monthly': 28},
        'BANKNIFTY': {'weekly': 7, 'monthly': 28},
        'SENSEX': {'weekly': 7, 'monthly': 28},
        'FINNIFTY': {'monthly': 28},
        'MIDCPNIFTY': {'monthly': 28},
    }
    
    def __init__(self):
        self.last_updated = None
        self.data_cache = {}
    
    def get_live_price(self, symbol: str) -> Dict:
        try:
            if symbol.upper() in self.INDICES_TICKERS:
                ticker = self.INDICES_TICKERS[symbol.upper()]
            else:
                ticker = symbol.upper() if symbol.endswith('.NS') or symbol.endswith('.BO') else f"{symbol.upper()}.NS"
            
            data = yf.download(ticker, period='1d', progress=False, timeout=10)
            
            if data.empty:
                return {'error': f'Could not fetch data for {symbol}'}
            
            close = data['Close'].iloc[-1]
            open_price = data['Open'].iloc[-1]
            change = close - open_price
            change_pct = (change / open_price) * 100 if open_price != 0 else 0
            
            return {
                'symbol': symbol,
                'price': round(close, 2),
                'open': round(open_price, 2),
                'high': round(data['High'].iloc[-1], 2),
                'low': round(data['Low'].iloc[-1], 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            return {'error': str(e), 'symbol': symbol}
    
    def get_historical_volatility(self, symbol: str, days: int = 20) -> float:
        try:
            if symbol.upper() in self.INDICES_TICKERS:
                ticker = self.INDICES_TICKERS[symbol.upper()]
            else:
                ticker = symbol.upper() if symbol.endswith('.NS') or symbol.endswith('.BO') else f"{symbol.upper()}.NS"
            
            data = yf.download(ticker, period=f'{days+5}d', progress=False, timeout=10)
            
            if data.empty or len(data) < 2:
                return 0.20
            
            returns = data['Close'].pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            
            return round(vol, 4)
        
        except Exception as e:
            return 0.20
    
    def get_strike_chain(self, symbol: str, spot_price: float) -> List[float]:
        if spot_price < 1000:
            interval = 10
        elif spot_price < 5000:
            interval = 50
        elif spot_price < 10000:
            interval = 100
        else:
            interval = 500
        
        base_strike = (spot_price // interval) * interval
        strikes = [base_strike + (i * interval) for i in range(-10, 11)]
        
        return sorted([s for s in strikes if s > 0])
    
    def get_expiry_days_list(self, symbol: str) -> Dict[str, int]:
        symbol_upper = symbol.upper()
        
        if symbol_upper in self.EXPIRY_INFO:
            expiries = self.EXPIRY_INFO[symbol_upper]
            result = {}
            
            for expiry_type, days in expiries.items():
                result[f'{expiry_type.capitalize()} ({days} days)'] = days
            
            return result
        else:
            return {'Monthly (28 days)': 28}

class IndianDataHelper:
    @staticmethod
    def rupee_format(amount: float) -> str:
        if amount >= 1_00_00_000:
            return f"₹{amount/1_00_00_000:.2f}Cr"
        elif amount >= 1_00_000:
            return f"₹{amount/1_00_000:.2f}L"
        elif amount >= 1_000:
            return f"₹{amount/1_000:.2f}K"
        else:
            return f"₹{amount:.2f}"
    
    @staticmethod
    def get_strike_difference(spot: float) -> int:
        if spot < 1000:
            return 10
        elif spot < 5000:
            return 50
        elif spot < 10000:
            return 100
        else:
            return 500
    
    @staticmethod
    def get_lot_size(symbol: str) -> int:
        symbol = symbol.upper()
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
            return 75
        elif symbol == 'SENSEX':
            return 10
        else:
            return 1
    
    @staticmethod
    def interpret_greeks(greek_name: str, value: float, symbol: str) -> str:
        interpretations = {
            'delta': {
                'description': 'Move 1 point, option price moves',
                'example': f'If Nifty moves ₹1, option price moves ₹{abs(value):.2f}'
            },
            'gamma': {
                'description': 'Delta change per point move',
                'example': f'Per ₹1 move, Delta changes by {value:.4f}'
            },
            'vega': {
                'description': '1% volatility change impact',
                'example': f'If volatility increases 1%, P&L changes ₹{value:.2f}'
            },
            'theta': {
                'description': 'Daily time decay (daily P&L)',
                'example': f'Daily loss: ₹{value:.2f} (if held till expiry)'
            },
            'rho': {
                'description': 'Interest rate sensitivity',
                'example': f'1% rate change = ₹{value:.2f} P&L change'
            }
        }
        
        if greek_name.lower() in interpretations:
            return interpretations[greek_name.lower()]['example']
        return str(value)

if __name__ == "__main__":
    print("=" * 80)
    print("Indian Market Data Loader - NSE Live Data")
    print("=" * 80)
    
    loader = IndianMarketDataLoader()
    
    print("\nLive Prices:")
    print("-" * 80)
    
    symbols = ['NIFTY', 'BANKNIFTY', 'SENSEX']
    
    for symbol in symbols:
        data = loader.get_live_price(symbol)
        if 'error' not in data:
            print(f"\n{symbol}")
            print(f"  Price:     ₹{data['price']}")
            print(f"  Change:    {data['change']:+.2f} ({data['change_pct']:+.2f}%)")
            print(f"  High/Low:  ₹{data['high']} / ₹{data['low']}")
        else:
            print(f"\n{symbol}: {data['error']}")
    
    print("\n\nHistorical Volatility (20-day):")
    print("-" * 80)
    
    for symbol in symbols:
        vol = loader.get_historical_volatility(symbol)
        print(f"{symbol:.<30} {vol*100:>6.2f}%")
    
    print("\n\nExpiry Information:")
    print("-" * 80)
    
    for symbol in symbols:
        expiries = loader.get_expiry_days_list(symbol)
        print(f"\n{symbol}:")
        for exp_type, days in expiries.items():
            print(f"  {exp_type}")
    
    print("\n" + "=" * 80)
