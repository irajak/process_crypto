import pandas as pd
import numpy as np

def backtest_strategy(data, initial_capital=10000):
    """
    Function to backtest a trading strategy.
    :param data: DataFrame with trading signals and prices.
    :param initial_capital: Initial amount of capital for backtesting.
    :return: DataFrame with portfolio value.
    """
    positions = initial_capital * data['Signal'] / data['Close']
    data['Portfolio Value'] = positions * data['Close']
    return data

# Example usage
if __name__ == "__main__":
    # Assuming 'data' is already defined with necessary columns
    backtested_data = backtest_strategy(data)
    roi = (backtested_data['Portfolio Value'].iloc[-1] - initial_capital) / initial_capital * 100
    print(f"Return on Investment (ROI) from backtest: {roi:.2f}%")