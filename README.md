# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sh1tfxxk/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sh1tfxxk/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC as five quantiles. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

An earn rate is a ratio, `rate = 100 * earned / tracked`, and the program models the two halves separately: log(tracked players) and log(earned count per trophy) are each fitted against `log(1 + hours since release)` and extrapolated to the target, then recombined. A second estimate fits the logit of the rate itself on the same time axis; the median is the average of the two. Identical consecutive reads (Exophase only changes the page when it refreshes) are collapsed before fitting, noisy per-trophy slopes are shrunk toward the median slope across trophies, and a long flat history is treated as an overdue refresh that dilutes rates. The spread combines a random-walk term, fit residuals, slope uncertainty, and one refresh step of uncertainty, so the five quantiles are honest about what is unknown. Standard library only, offline, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=70705sxz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
