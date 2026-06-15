from datetime import datetime, timezone, timedelta

UTC_8 = timezone(timedelta(hours=8))


def now_utc8():
    return datetime.now(UTC_8)
