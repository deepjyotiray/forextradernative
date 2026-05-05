import unittest

import pandas as pd

from engine.mtf_bias import compute_bias


def _df_from_closes(closes):
    rows = []
    for idx, close in enumerate(closes):
        open_price = closes[idx - 1] if idx > 0 else close
        high = max(open_price, close) + 0.4
        low = min(open_price, close) - 0.4
        rows.append(
            {
                "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=idx),
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 1000.0 + idx,
            }
        )
    return pd.DataFrame(rows)


class MultiTimeframeBiasTests(unittest.TestCase):
    def test_m15_has_more_weight_than_h1(self):
        h1_df = _df_from_closes([
            100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0,
            104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0,
            109.5, 111.0, 110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0,
            114.5, 116.0, 115.5, 117.0, 116.5, 118.0,
        ])
        m15_df = _df_from_closes([
            120.0, 119.0, 119.6, 118.3, 118.9, 117.5, 118.0, 116.8, 117.2, 115.9,
            116.4, 115.0, 115.5, 114.1, 114.6, 113.2, 113.7, 112.4, 112.8, 111.6,
            112.0, 110.7, 111.1, 109.8, 110.2, 108.9, 109.3, 108.0, 108.4, 107.1,
            107.6, 106.3, 106.7, 105.4, 105.8, 104.5,
        ])

        bias = compute_bias(None, h1_df, m15_df)

        self.assertEqual(bias["direction"], "SHORT")
        self.assertGreater(bias["scores"]["SHORT"], bias["scores"]["LONG"])

    def test_h4_bias_removed_from_payload(self):
        h1_df = _df_from_closes([100 + i * 0.5 for i in range(40)])
        m15_df = _df_from_closes([100 + i * 0.4 for i in range(40)])

        bias = compute_bias(None, h1_df, m15_df)

        self.assertNotIn("h4_bias", bias)
        self.assertIn("m15_structure", bias)


if __name__ == "__main__":
    unittest.main()
