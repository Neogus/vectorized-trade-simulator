import pandas as pd
import numpy as np

# Define exit conditions for long trades (take profit or stop loss)
def long_exits(df, trade_col, take_p, stop_l, ex):
    df.loc[
        ((df.High.shift(1) < df.TL.shift(1) * (1 + take_p)) & (df.High > df.TL * (1 + take_p))) |
        ((df.Low.shift(1) > df.TL.shift(1) * (1 - stop_l)) & (df.Low < df.TL * (1 - stop_l))), trade_col] = ex
    return df

# Define exit conditions for short trades (take profit or stop loss)
def short_exits(df, trade_col, take_p, stop_l, ex):
    df.loc[
        ((df.Low.shift(1) > df.TL.shift(1) * (1 - take_p)) & (df.Low < df.TL * (1 - take_p))) |
        ((df.High.shift(1) < df.TL.shift(1) * (1 + stop_l)) & (df.High > df.TL * (1 + stop_l))), trade_col] = ex
    return df

# Dictionary to manage entry/exit logic and dependencies between long and short trades
aux_dic = {
    'Long_Trade': [long_exits, [-1, 1, -2, 2], 'Short_Trade'],
    'Short_Trade': [short_exits, [-2, 2, -1, 1], 'Long_Trade']
}

def exit_loop(df, trade_col, take_p, stop_l, second=False):
    # Decompose key elements from aux_dic
    en, ex, sen, sex = aux_dic[trade_col][1]

    # Store entry points
    df['entries'] = df[df[trade_col] == en][trade_col]

    # Initialize loop tracking variables
    exit_n = 2
    len_t = 0

    # Iteratively identify exit points until no more new exits are found
    while len_t != exit_n:
        len_t = exit_n

        # Create/clean trailing entry price column
        df['TL'] = np.nan
        df.loc[df[trade_col] == en, 'TL'] = df['Close']
        df.TL = df.TL.fillna(method='ffill')

        # Store previous number of exits
        exit_n = len(df[df[trade_col] > 0][trade_col])

        # Remove existing exit signals to recalculate cleanly
        df[trade_col] = df[df[trade_col] != ex][trade_col]

        # Apply exit logic (stop loss / take profit)
        df = aux_dic[trade_col][0](df, trade_col, take_p, stop_l, ex)

        # Reduce consecutive duplicate signals
        trade = df[trade_col].dropna()
        trade = trade[trade.shift(1) != trade]
        df[trade_col] = trade

        # Refill missed exit values due to forward fill logic
        df['trade_aux'] = df[trade_col].fillna(method='ffill')
        df.loc[(df['trade_aux'] == ex) & (df['trade_aux'].shift(1) != en), trade_col] = df['entries']

    # Clean up temporary columns
    df.drop(columns=['trade_aux', 'TL', 'entries'], inplace=True)

    # Initialize unified Trade column for merged trade signals
    df[trade_col] = trade
    df['Trade'] = np.nan

    # Skip merging if this is the first pass
    if not second:
        df['Trade'] = df[trade_col]
    else:
        primary = aux_dic[trade_col][2]

        # Prepare DataFrame with both trade types
        df2 = df[[primary, trade_col]].dropna()
        df2.loc[(df2[primary].shift(1) == sen) & (df2[trade_col].shift(1) == en), trade_col] = np.nan
        df2.loc[(df2[primary] == sen) & (df2[trade_col] == en), trade_col] = np.nan

        t_aux = df2[trade_col]
        df.loc[t_aux.index, trade_col] = t_aux

        # Assign temporary trade markers for both entry and exit directions
        df.loc[df[primary] < 0, 'ta1'] = df[primary]
        df.loc[df[trade_col] < 0, 'ta1'] = df[trade_col]
        df.loc[df[primary] > 0, 'ta2'] = df[primary]
        df.loc[df[trade_col] > 0, 'ta2'] = df[trade_col]

        # Fill gaps with forward-filled auxiliary columns
        df['ta1'] = df['ta1'].fillna(method='ffill')
        df['ta2'] = df['ta2'].fillna(method='ffill')
        df['ta3'] = df[primary].fillna(method='ffill')
        df['ta4'] = df[trade_col].fillna(method='ffill')

        # Use logic to derive merged trade signals
        df.loc[(df[trade_col] == en), 'Trade'] = en
        df.loc[(df[primary] == sen) & (df.Trade.isna()), 'Trade'] = sen
        df.loc[(df[primary] == sen) & (df.ta4 == en), 'Trade'] = np.nan

        # Final clean-up loop to remove overlapping/conflicting signals
        for _ in range(3):
            df['t_aux'] = np.nan
            t_aux = df[df['Trade'] < 0]['Trade']
            df.loc[t_aux.index, 't_aux'] = t_aux
            df['t_aux'] = df['t_aux'].fillna(method='ffill')
            df.loc[df.Trade > 0, 'Trade'] = np.nan
            df.loc[df[primary].abs() == df.t_aux.abs(), 'Trade'] = df[primary]
            df.loc[df[trade_col].abs() == df.t_aux.abs(), 'Trade'] = df[trade_col]

            # Drop consecutive duplicate signals
            t_aux = df['Trade'].dropna()
            t_aux = t_aux[np.sign(t_aux.shift(1)) != np.sign(t_aux)]
            df['Trade'] = t_aux

    return df

def get_ret(df, stop_l, take_p, fee, t_min=0, delay=0):
    """
    Calculates return series from long and short trades based on take profit and stop loss targets.

    Parameters:
    - df: pandas DataFrame with 'High', 'Low', 'Close', 'Long_Trade', and 'Short_Trade'
    - stop_l: stop loss percentage (e.g., 0.02 for 2%)
    - take_p: take profit percentage
    - fee: transaction fee per trade
    - t_min: minimum number of trades required
    - delay: slippage or delay cost
    """

    # Count entries
    len_lt = len(df[df.Long_Trade == -1])
    len_st = len(df[df.Short_Trade == -2])
    t_len = len_lt + len_st

    # Exit early if not enough trades
    if t_len < t_min or t_len < 2:
        return pd.Series([])

    # Trim unnecessary data
    df = df[['Low', 'High', 'Close', 'Long_Trade', 'Short_Trade']]

    # Conditions to allow calculating trades
    long_condition = len_lt > 2 and len_lt > t_min * len_lt / t_len
    short_condition = len_st > 2 and len_st > t_min * len_st / t_len

    if long_condition and short_condition:
        trade = df.loc[df['Long_Trade'] == -1, 'Long_Trade']
        trade2 = df.loc[df['Short_Trade'] == -2, 'Short_Trade']

        # Choose primary signal based on chronological order
        if trade.index[0] <= trade2.index[0]:
            primary, secondary = 'Long_Trade', 'Short_Trade'
        else:
            primary, secondary = 'Short_Trade', 'Long_Trade'

        # Apply both trade exit loops
        df = exit_loop(df, primary, take_p, stop_l, second=False)
        df = exit_loop(df, secondary, take_p, stop_l, second=True)

    elif long_condition:
        df = exit_loop(df, 'Long_Trade', take_p, stop_l, second=False)
    elif short_condition:
        df = exit_loop(df, 'Short_Trade', take_p, stop_l, second=False)
    else:
        return pd.Series([])

    # Final processing to calculate return
    df = df[['Low', 'High', 'Close', 'Trade']].dropna(subset=['Trade'])

    # Calculate returns based on achieved exit
    df.loc[(df.Trade == 1) & (df.High > df.Close.shift(1) * (1 + take_p)), 'Ret'] = take_p - fee + delay
    df.loc[(df.Trade == 1) & (df.Low < df.Close.shift(1) * (1 - stop_l)), 'Ret'] = -stop_l - fee - delay
    df.loc[(df.Trade == 2) & (df.High > df.Close.shift(1) * (1 + stop_l)), 'Ret'] = stop_l - fee + delay
    df.loc[(df.Trade == 2) & (df.Low < df.Close.shift(1) * (1 - take_p)), 'Ret'] = -take_p - fee - delay

    # Return only finalized results for closing trades
    ret = df.Ret[1::2].dropna()
    return ret
