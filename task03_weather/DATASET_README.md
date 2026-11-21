# task03_noaa_rain_tomorrow

Temporal 60/20/20 split by date (row date = the day features are observed, day t). Train is
the oldest 60% of days, dev the next 20%, test the latest 20%. Tasks 2-5 share these exact
date cutoffs. Rows whose 3-day target window crosses a split boundary are dropped, as are
rows with a missing target. Train and dev rows are in date order; test rows are shuffled.

| Split | First day | Last day |
| --- | --- | --- |
| train | 2004-10-21 | 2018-01-02 |
| dev | 2018-01-06 | 2022-05-29 |

| Split | Rows | Share |
| --- | --- | --- |
| train | 4820 | 60.2% |
| dev | 1596 | 19.9% |

| Class | train | dev |
| --- | --- | --- |
| 0 = no rain (<= 0.1 in) | 75.7% | 75.1% |
| 1 = rain (> 0.1 in = 2.54 mm) | 24.3% | 24.9% |

**Files:** `{train,dev}.X` and `.CT`. `feature_names.txt` names each `.X` column.

**Features:** `<var>_lag<k>` is the value k days before day t (lag0 = day t). Values are as
observed: days missing from the station records are written as `nan` and are not filled in.
Units (converted from GHCN's tenths): precipitation mm, temperature °C,
wind speed m/s, direction degrees, discharge cfs, gage height ft. Weather is GHCN-Daily station
USW00024217 (Bellingham Intl Airport). Columns starting `uil_` are from GHCN-Daily station USW00094240 (Quillayute,
on the Washington coast, usually upwind of Bellingham), including sea-level pressure
(hPa) and dew point (°C); its pressure and dew point records start in 2006, so they
are `nan` before then.

Label: 1 if tomorrow's precipitation exceeds 0.1 in (2.54 mm), else 0.

This package holds the train and dev splits. Test inputs are released separately; see the course schedule.
