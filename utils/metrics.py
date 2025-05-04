"""
Performance metrics calculation for options trading strategies.
"""
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0, annualize=True, trading_days_per_year=252):
    """
    Calculate the Sharpe ratio from a series of portfolio values
    
    Args:
        portfolio_values (list): List of portfolio values over time
        risk_free_rate (float): Annual risk-free rate (default: 0.0)
        annualize (bool): Whether to annualize the Sharpe ratio (default: True)
        trading_days_per_year (int): Number of trading days per year (default: 252)
        
    Returns:
        float: Sharpe ratio
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(values) / values[:-1]
    
    # Calculate mean return and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Use ddof=1 for sample std
    
    if std_return == 0:
        return 0.0
    
    # Daily risk-free rate
    daily_risk_free = risk_free_rate / trading_days_per_year if annualize else risk_free_rate
    
    # Calculate Sharpe
    sharpe = (mean_return - daily_risk_free) / std_return
    
    # Annualize if requested
    if annualize:
        sharpe *= np.sqrt(trading_days_per_year)
    
    return sharpe

def calculate_sortino_ratio(portfolio_values, risk_free_rate=0.0, annualize=True, trading_days_per_year=252):
    """
    Calculate the Sortino ratio from a series of portfolio values
    
    Args:
        portfolio_values (list): List of portfolio values over time
        risk_free_rate (float): Annual risk-free rate (default: 0.0)
        annualize (bool): Whether to annualize the ratio (default: True)
        trading_days_per_year (int): Number of trading days per year (default: 252)
        
    Returns:
        float: Sortino ratio
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate returns
    returns = np.diff(values) / values[:-1]
    
    # Calculate mean return
    mean_return = np.mean(returns)
    
    # Daily risk-free rate
    daily_risk_free = risk_free_rate / trading_days_per_year if annualize else risk_free_rate
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = returns[returns < daily_risk_free]
    
    if len(negative_returns) == 0:
        return np.inf  # No negative returns
    
    downside_deviation = np.std(negative_returns, ddof=1)
    
    if downside_deviation == 0:
        return 0.0
    
    # Calculate Sortino
    sortino = (mean_return - daily_risk_free) / downside_deviation
    
    # Annualize if requested
    if annualize:
        sortino *= np.sqrt(trading_days_per_year)
    
    return sortino

def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from a series of portfolio values
    
    Args:
        portfolio_values (list): List of portfolio values over time
        
    Returns:
        float: Maximum drawdown as a percentage (0 to 1)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Convert to numpy array
    values = np.array(portfolio_values)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdowns
    drawdowns = (running_max - values) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdowns)
    
    return max_drawdown

def calculate_win_rate(trades):
    """
    Calculate win rate from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'profit' keys
        
    Returns:
        float: Win rate as a percentage (0 to 1)
    """
    if not trades:
        return 0.0
    
    wins = sum(1 for trade in trades if trade.get('profit', 0) > 0)
    return wins / len(trades)

def calculate_profit_factor(trades):
    """
    Calculate profit factor (gross profit / gross loss)
    
    Args:
        trades (list): List of trade dictionaries with 'profit' keys
        
    Returns:
        float: Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
    gross_loss = sum(abs(trade.get('profit', 0)) for trade in trades if trade.get('profit', 0) < 0)
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss

def calculate_strategy_diversity(portfolio, strategies):
    """
    Calculate strategy diversity in a portfolio
    
    Args:
        portfolio (list): List of position dictionaries with 'strategy' keys
        strategies (list): List of all available strategies
        
    Returns:
        float: Strategy diversity score (0 to 1)
    """
    if not portfolio or not strategies:
        return 0.0
    
    # Count positions by strategy
    strategy_counts = {}
    for position in portfolio:
        strategy = position.get('strategy', '')
        if strategy:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    # Calculate diversity score (number of strategies used / total strategies available)
    return len(strategy_counts) / len(strategies)

def calculate_strategy_metrics(strategy_results):
    """
    Calculate comprehensive metrics for strategies
    
    Args:
        strategy_results (dict): Dictionary of strategy results
        
    Returns:
        dict: Dictionary of metrics for each strategy
    """
    metrics = {}
    
    for strategy, results in strategy_results.items():
        # Extract portfolio values
        portfolio_values = results.get('portfolio_values', [])
        
        # Extract trades
        trades = results.get('trades', [])
        
        # Calculate metrics
        sharpe = calculate_sharpe_ratio(portfolio_values)
        sortino = calculate_sortino_ratio(portfolio_values)
        max_dd = calculate_max_drawdown(portfolio_values)
        win_rate = calculate_win_rate(trades)
        profit_factor = calculate_profit_factor(trades)
        
        metrics[strategy] = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trades)
        }
    
    return metrics
