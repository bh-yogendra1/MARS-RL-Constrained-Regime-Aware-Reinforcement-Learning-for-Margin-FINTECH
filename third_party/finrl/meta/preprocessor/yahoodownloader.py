"""Offline stub. The real FinRL YahooDownloader needs Yahoo Finance (blocked in the execution sandbox).
MARS-RL never calls it: raw prices are supplied as CSV and hashed in data/manifests/."""
class YahooDownloader:  # noqa
    def __init__(self, *a, **k):
        raise RuntimeError("YahooDownloader stub: provide data via data/raw/*.csv")
