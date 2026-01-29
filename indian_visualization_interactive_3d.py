import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple
from indian_bs_model import IndianBlackScholesModel

class IndianGreeksInteractiveVisualizer:
    
    @staticmethod
    def plot_greeks_surface_interactive(S_range, K_range, days_to_expiry, r, sigma, option_type='call', greek='delta'):
        """Create interactive 3D surface plot with hover values"""
        
        S_grid, K_grid = np.meshgrid(S_range, K_range)
        greek_surface = np.zeros_like(S_grid)
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                model = IndianBlackScholesModel(S_grid[i, j], K_grid[i, j], days_to_expiry, r, sigma, option_type)
                greeks = model.all_greeks()
                greek_surface[i, j] = greeks[greek]
        
        fig = go.Figure(data=[go.Surface(
            x=S_grid[0],
            y=K_grid[:, 0],
            z=greek_surface,
            colorscale='Viridis',
            hovertemplate='<b>Spot: ₹%{x:.0f}</b><br><b>Strike: ₹%{y:.0f}</b><br><b>' + greek.upper() + ': %{z:.4f}</b><extra></extra>',
            colorbar=dict(title=greek.upper())
        )])
        
        fig.update_layout(
            title=f'<b>{greek.upper()} Surface - {option_type.upper()} Option</b><br>{days_to_expiry} days to expiry',
            scene=dict(
                xaxis_title='Spot Price (₹)',
                yaxis_title='Strike Price (₹)',
                zaxis_title=f'{greek.upper()} Value',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=700,
            hovermode='closest',
            font=dict(size=11)
        )
        
        return fig
    
    @staticmethod
    def plot_all_greeks_interactive(spot_price, days_to_expiry, r=0.065, sigma=0.20):
        """Create 4 interactive 3D surfaces in one view"""
        
        S_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 30)
        K_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 30)
        
        greeks_to_plot = ['delta', 'gamma', 'vega', 'theta']
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=[g.upper() for g in greeks_to_plot]
        )
        
        for idx, greek in enumerate(greeks_to_plot, 1):
            S_grid, K_grid = np.meshgrid(S_range, K_range)
            greek_surface = np.zeros_like(S_grid)
            
            for i in range(S_grid.shape[0]):
                for j in range(S_grid.shape[1]):
                    model = IndianBlackScholesModel(S_grid[i, j], K_grid[i, j], days_to_expiry, r, sigma, 'call')
                    greeks = model.all_greeks()
                    greek_surface[i, j] = greeks[greek]
            
            row = (idx - 1) // 2 + 1
            col = (idx - 1) % 2 + 1
            
            fig.add_trace(
                go.Surface(
                    x=S_grid[0],
                    y=K_grid[:, 0],
                    z=greek_surface,
                    colorscale='Viridis',
                    hovertemplate='<b>Spot: ₹%{x:.0f}</b><br><b>Strike: ₹%{y:.0f}</b><br><b>' + greek.upper() + ': %{z:.4f}</b><extra></extra>',
                    colorbar=dict(title=greek.upper(), len=0.4),
                    showscale=True
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text=f'<b>All Greeks Surfaces - {days_to_expiry} days to expiry</b>',
            width=1400,
            height=1000,
            showlegend=False,
            font=dict(size=10)
        )
        
        for i in range(1, 5):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.update_scenes(
                xaxis_title='Spot (₹)',
                yaxis_title='Strike (₹)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                row=row, col=col
            )
        
        return fig
    
    @staticmethod
    def plot_greek_sensitivity_interactive(spot_price, strike, days_to_expiry, r=0.065, sigma=0.20, greek='delta'):
        """Interactive sensitivity curve"""
        
        S_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
        greek_values = []
        
        for S in S_range:
            model = IndianBlackScholesModel(S, strike, days_to_expiry, r, sigma, 'call')
            greeks = model.all_greeks()
            greek_values.append(greeks[greek])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=S_range,
            y=greek_values,
            mode='lines+markers',
            name=greek.upper(),
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=5),
            hovertemplate='<b>Spot: ₹%{x:.0f}</b><br><b>' + greek.upper() + ': %{y:.4f}</b><extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.add_vline(
            x=spot_price,
            line_dash='dash',
            line_color='red',
            annotation_text='Current Spot',
            annotation_position='top right'
        )
        
        fig.update_layout(
            title=f'<b>{greek.upper()} Sensitivity vs Spot Price</b><br>{days_to_expiry} days to expiry',
            xaxis_title='Spot Price (₹)',
            yaxis_title=f'{greek.upper()} Value',
            hovermode='x unified',
            width=1000,
            height=600,
            font=dict(size=12),
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_volatility_impact_interactive(spot_price, strike, days_to_expiry, r=0.065):
        """Interactive volatility impact with 3 curves"""
        
        vol_range = np.linspace(0.05, 0.60, 50)
        price_values = []
        delta_values = []
        vega_values = []
        
        for vol in vol_range:
            model = IndianBlackScholesModel(spot_price, strike, days_to_expiry, r, vol, 'call')
            greeks = model.all_greeks()
            price_values.append(greeks['price'])
            delta_values.append(greeks['delta'])
            vega_values.append(greeks['vega'])
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Price vs Volatility', 'Delta vs Volatility', 'Vega vs Volatility')
        )
        
        fig.add_trace(
            go.Scatter(
                x=vol_range * 100,
                y=price_values,
                mode='lines+markers',
                name='Price',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Vol: %{x:.1f}%</b><br><b>Price: ₹%{y:.2f}</b><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.2)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=vol_range * 100,
                y=delta_values,
                mode='lines+markers',
                name='Delta',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Vol: %{x:.1f}%</b><br><b>Delta: %{y:.4f}</b><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.2)'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=vol_range * 100,
                y=vega_values,
                mode='lines+markers',
                name='Vega',
                line=dict(color='#d62728', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Vol: %{x:.1f}%</b><br><b>Vega: %{y:.4f}</b><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)'
            ),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text='Volatility (%)', row=1, col=1)
        fig.update_xaxes(title_text='Volatility (%)', row=1, col=2)
        fig.update_xaxes(title_text='Volatility (%)', row=1, col=3)
        
        fig.update_yaxes(title_text='Price (₹)', row=1, col=1)
        fig.update_yaxes(title_text='Delta', row=1, col=2)
        fig.update_yaxes(title_text='Vega', row=1, col=3)
        
        fig.update_layout(
            title_text='<b>Volatility Impact Analysis</b>',
            width=1400,
            height=500,
            hovermode='x unified',
            font=dict(size=11),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_time_decay_interactive(spot_price, strike, r=0.065, sigma=0.20):
        """Interactive time decay analysis"""
        
        days_range = np.arange(60, 0, -1)
        price_values = []
        theta_values = []
        
        for days in days_range:
            model = IndianBlackScholesModel(spot_price, strike, days, r, sigma, 'call')
            greeks = model.all_greeks()
            price_values.append(greeks['price'])
            theta_values.append(greeks['theta'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Price vs Days to Expiry', 'Theta (Daily Decay) vs Days to Expiry')
        )
        
        fig.add_trace(
            go.Scatter(
                x=days_range,
                y=price_values,
                mode='lines+markers',
                name='Price',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Days: %{x}</b><br><b>Price: ₹%{y:.2f}</b><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=days_range,
                y=theta_values,
                mode='lines+markers',
                name='Theta',
                line=dict(color='#d62728', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Days: %{x}</b><br><b>Theta: ₹%{y:.4f}/day</b><extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.2)'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Days to Expiry', row=1, col=1)
        fig.update_xaxes(title_text='Days to Expiry', row=1, col=2)
        
        fig.update_yaxes(title_text='Option Price (₹)', row=1, col=1)
        fig.update_yaxes(title_text='Theta (₹/day)', row=1, col=2)
        
        fig.update_layout(
            title_text='<b>Time Decay Analysis</b>',
            width=1400,
            height=500,
            hovermode='x unified',
            font=dict(size=11),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_implied_volatility_smile_interactive(spot_price, days_to_expiry, r=0.065, sigma=0.20):
        """Interactive IV smile visualization"""
        
        K_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
        call_prices = []
        put_prices = []
        
        for K in K_range:
            model_call = IndianBlackScholesModel(spot_price, K, days_to_expiry, r, sigma, 'call')
            model_put = IndianBlackScholesModel(spot_price, K, days_to_expiry, r, sigma, 'put')
            call_prices.append(model_call.price())
            put_prices.append(model_put.price())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=K_range,
            y=call_prices,
            mode='lines+markers',
            name='Call Price',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Strike: ₹%{x:.0f}</b><br><b>Call: ₹%{y:.2f}</b><extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=K_range,
            y=put_prices,
            mode='lines+markers',
            name='Put Price',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Strike: ₹%{x:.0f}</b><br><b>Put: ₹%{y:.2f}</b><extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        fig.add_vline(
            x=spot_price,
            line_dash='dash',
            line_color='red',
            line_width=2,
            annotation_text='Current Spot',
            annotation_position='top right'
        )
        
        fig.update_layout(
            title=f'<b>Call-Put Price Smile (IV Smile)</b><br>{days_to_expiry} days to expiry',
            xaxis_title='Strike Price (₹)',
            yaxis_title='Option Price (₹)',
            hovermode='x unified',
            width=1000,
            height=600,
            font=dict(size=12),
            template='plotly_white',
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    @staticmethod
    def plot_greeks_heatmap(spot_price, days_to_expiry, r=0.065, sigma=0.20):
        """Interactive heatmap showing all Greeks at different spots and strikes"""
        
        S_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 25)
        K_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 25)
        
        S_grid, K_grid = np.meshgrid(S_range, K_range)
        
        greeks_data = {
            'delta': np.zeros_like(S_grid),
            'gamma': np.zeros_like(S_grid),
            'vega': np.zeros_like(S_grid),
            'theta': np.zeros_like(S_grid)
        }
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                model = IndianBlackScholesModel(S_grid[i, j], K_grid[i, j], days_to_expiry, r, sigma, 'call')
                greeks = model.all_greeks()
                for greek_name in greeks_data:
                    greeks_data[greek_name][i, j] = greeks[greek_name]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta Heatmap', 'Gamma Heatmap', 'Vega Heatmap', 'Theta Heatmap'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        for idx, (greek_name, data) in enumerate(greeks_data.items(), 1):
            row = (idx - 1) // 2 + 1
            col = (idx - 1) % 2 + 1
            
            fig.add_trace(
                go.Heatmap(
                    x=S_range,
                    y=K_range,
                    z=data,
                    colorscale='RdBu',
                    hovertemplate='<b>Spot: ₹%{x:.0f}</b><br><b>Strike: ₹%{y:.0f}</b><br><b>' + greek_name.upper() + ': %{z:.4f}</b><extra></extra>',
                    colorbar=dict(title=greek_name.upper(), len=0.4),
                    showscale=True
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text='Spot Price (₹)')
        fig.update_yaxes(title_text='Strike Price (₹)')
        
        fig.update_layout(
            title_text=f'<b>Greeks Heatmaps - {days_to_expiry} days to expiry</b>',
            width=1400,
            height=1000,
            font=dict(size=10)
        )
        
        return fig

if __name__ == "__main__":
    print("Interactive 3D Visualization Examples")
    print("=" * 60)
    
    spot = 23000
    strike = 23000
    days = 7
    
    print("Generating interactive 3D Delta Surface...")
    fig1 = IndianGreeksInteractiveVisualizer.plot_greeks_surface_interactive(
        np.linspace(22000, 24000, 25),
        np.linspace(22000, 24000, 25),
        days, 0.065, 0.20, 'call', 'delta'
    )
    print("✅ Interactive Delta Surface ready!")
    
    print("Generating all Greeks interactive surfaces...")
    fig2 = IndianGreeksInteractiveVisualizer.plot_all_greeks_interactive(spot, days)
    print("✅ All Greeks surfaces ready!")