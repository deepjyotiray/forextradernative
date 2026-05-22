# Example Before / After Scenario

## Before

- Strategy: `M15_ZONE_SCALP`
- Setup: `SUPPLY` SELL
- Behaviour: immediate SELL on first touch
- Result: price sweeps higher first, then later turns down after bearish reclaim

## Forensic lesson

- Classification: `CORRECT_IDEA_WRONG_TIMING`
- Meaning: the trade direction was not the problem, but the entry was too early

## After live correction

- New runtime behaviour for the same strategy:
  wait for bearish candle close
  require EMA / VWAP alignment
  require bearish microflow confirmation

## Outcome

- Similar future setup is not disabled outright
- It is converted from immediate entry into pending-confirmation entry
- If confirmation never appears before expiry, the service skips the trade and logs why

