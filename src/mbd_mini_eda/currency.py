"""Project currency flags: code 11 is primary; other known codes are foreign."""
import pandas as pd


def currency_flags(currency: pd.Series) -> pd.DataFrame:
    """Keep missing codes unknown, rather than classifying them as foreign.

    'Foreign' is the user's project convention; it does not recover ISO currencies
    from the anonymized codes. The input is not modified.
    """
    numeric = pd.to_numeric(currency, errors='raise')
    known = numeric.notna()
    if (numeric[known] % 1 != 0).any():
        raise ValueError('Currency codes must be integers or missing.')
    main = numeric.eq(11).astype('boolean').mask(~known)
    return pd.DataFrame({
        'currency_is_main': main,
        'currency_is_foreign': ~main,
        'currency_missing': ~known,
    }, index=currency.index)


def currency_analysis_group(currency: pd.Series) -> pd.Series:
    """Fixed user-requested EDA groups; preserve every other code and missing.

    Codes 5, 9, 12 and 16 have 2--7 observed rows each in this snapshot. This
    grouping does not pool their amounts into a common monetary unit.
    """
    currency_flags(currency)  # Validate integer codes without changing the input.
    numeric = pd.to_numeric(currency, errors='raise')
    return numeric.astype('Int64').astype('string').mask(
        numeric.isin([5, 9, 12, 16]), 'other_currency'
    ).fillna('missing_currency').rename('currency_group')
