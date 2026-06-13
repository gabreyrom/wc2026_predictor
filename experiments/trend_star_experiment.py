"""Test: value TREND and star CONCENTRATION as calibrator features."""
import sys; sys.path.insert(0, '.')
import numpy as np
import src.model.lgbm_calibrator as lc
from src.data.fetch_matches import fetch_and_process
from src.data.market_values import value_trend_diff, star_share_diff, _value_trend
from src.model.calibration import oof_calibration_report, paired_bootstrap_test
from src.model.lgbm_calibrator import LGBMCalibrator
import pandas as pd

matches = fetch_and_process(force=False)
fit_data = matches[matches['date'] >= '2010-01-01'].copy()

# Sanity
ts = pd.Timestamp('2026-06-11')
print("Value trends 2022→2025 (log):")
for t in ["Morocco", "Norway", "Ecuador", "Belgium", "Argentina", "England"]:
    print(f"  {t:<12} {_value_trend(t, ts):+.3f}")
print(f"star_share_diff(Argentina, England) = {star_share_diff('Argentina','England',ts):+.3f}")

fns = {'value_trend_diff': value_trend_diff, 'star_share_diff': star_share_diff}
results, _ = oof_calibration_report(fit_data, xi=0.003, extra_feature_fns=fns)
cal_train = results["cal_train_df"]
test_pred = results["test_pred_df"].reset_index(drop=True)

for c in fns:
    print(f"{c} coverage: train {cal_train[c].notna().mean():.1%}  test {test_pred[c].notna().mean():.1%}")

BASE = list(lc.FEATURE_COLS)
configs = {
    "base":        BASE,
    "+trend":      BASE + ['value_trend_diff'],
    "+star":       BASE + ['star_share_diff'],
    "+both":       BASE + ['value_trend_diff', 'star_share_diff'],
}
lls, cals = {}, {}
for name, cols in configs.items():
    lc.FEATURE_COLS = cols
    cal = LGBMCalibrator().fit_cv(cal_train, verbose=False)
    lls[name] = cal.predict_proba_df(test_pred)['cal_log_loss'].values
    cals[name] = cal
    print(f"{name:<8} CV {cal.cv_log_loss_:.4f}   test {lls[name].mean():.4f}")
lc.FEATURE_COLS = BASE

print("\nPaired bootstraps vs base (full test):")
for name in ["+trend", "+star", "+both"]:
    b = paired_bootstrap_test(ll_base=lls["base"], ll_alt=lls[name])
    sig = (b['ci_low'] > 0) or (b['ci_high'] < 0)
    v = ('HELPS' if b['mean_diff'] < 0 else 'HURTS') if sig else 'no difference'
    print(f"  {name:<8} Δ={b['mean_diff']:+.4f} CI[{b['ci_low']:+.4f},{b['ci_high']:+.4f}] → {v}")

# Subset: matches with big trend gaps (rising vs falling teams)
big = np.abs(test_pred['value_trend_diff'].fillna(0).values) > 0.4
b = paired_bootstrap_test(ll_base=lls['base'][big], ll_alt=lls['+trend'][big])
print(f"\nBig trend gaps (n={big.sum():,}): Δ={b['mean_diff']:+.4f} CI[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]")
# Subset: Argentina (star concentration hypothesis)
arg = ((test_pred.home_team == 'Argentina') | (test_pred.away_team == 'Argentina')).values
b = paired_bootstrap_test(ll_base=lls['base'][arg], ll_alt=lls['+star'][arg])
print(f"Argentina matches (n={arg.sum()}): Δ={b['mean_diff']:+.4f} CI[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]")

for name in ["+trend", "+star"]:
    imp = cals[name].feature_importance_
    f = 'value_trend_diff' if name == '+trend' else 'star_share_diff'
    vals = sorted(imp.values(), reverse=True)
    print(f"{f}: importance {imp.get(f,0):.0f} (rank {vals.index(imp.get(f,0))+1} of {len(imp)})")
