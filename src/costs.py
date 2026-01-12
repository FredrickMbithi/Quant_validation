"""
Transaction Cost Models
- Bid-ask spreads, commissions, slippage
"""

def estimate_spread(asset, volume):
    """
    Estimate bid-ask spread for given asset and volume.
    
    Args:
        asset: Asset identifier
        volume: Trade volume
        
    Returns:
        Spread as percentage
    """
    pass


def estimate_commission(notional):
    """
    Estimate commissions for a trade.
    
    Args:
        notional: Trade notional value
        
    Returns:
        Commission cost
    """
    pass


def estimate_slippage(volume, asset):
    """
    Estimate slippage for given volume.
    
    Args:
        volume: Trade volume
        asset: Asset identifier
        
    Returns:
        Slippage cost
    """
    pass
