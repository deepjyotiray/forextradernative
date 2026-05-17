"""
Canonical SMC strategy entrypoint.

`SMC_CONFLUENCE` now uses the Strategy 1 model:
HTF liquidity sweep -> HTF BOS -> order-block retest -> LTF IFVG confirmation.
"""
from .liquidity_sweep_ob_strategy import LiquiditySweepOrderBlockStrategy


class SMCStrategy(LiquiditySweepOrderBlockStrategy):
    name = "SMC_CONFLUENCE"
