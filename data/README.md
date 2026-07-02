# data/

Local data storage. **Nothing in this directory is committed to git** except this README.

| Path | Contents |
|---|---|
| `raw/` | As-downloaded vendor data (one parquet per symbol) |
| `cache/` | Download cache keyed by (source, symbol, range, interval) |
| `csv/` | User-provided CSVs (`SYMBOL.csv` with date, open, high, low, close, volume) |
| `processed/` | Feature/label panels produced by `make build-features` |

All data sources are public (yfinance, user CSVs, or the built-in synthetic
generator). Do not place proprietary, licensed, or restricted data here.
