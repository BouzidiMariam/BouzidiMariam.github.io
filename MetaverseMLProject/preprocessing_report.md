# Preprocessing report

**Scenario:** `no_score` - risk_score removed (it is the risk assessment itself). Still easy: transaction_type + age_group + night-time almost determine the label.
**Split:** stratified (70%/30%) | **Scaler:** standard | **Seed:** 42

## Pipeline steps

1. **Load & validate** - 78,600 rows x 14 columns, schema OK
2. **Clean** - 78,600 -> 78,600 rows | duplicates removed: 0 | out-of-range values -> NaN: 0 | hour/timestamp mismatches fixed: 0 | sorted by timestamp
3. **Encode target** - low_risk=0 < moderate_risk=1 < high_risk=2 (fixed order, class names now match the encoding)
4. **Engineer features** - time: is_night, hour_sin, hour_cos; night=0-6h (weekday/month/weekend computed for the audit but excluded: mutual information ~ 0); interactions: amount_per_minute, session_per_login
5. **Audit** - leakage scan flagged ['risk_score']; redundant pairs: [('purchase_pattern', 'age_group')]; noise categoricals: ['location_region', 'ip_prefix']
6. **Select features** - dropped: risk_score [scenario 'no_score']; purchase_pattern [duplicate of 'age_group' (Cramér's V=1.0)]; location_region [no association with target (V=0.0075, p=0.3589)]; ip_prefix [no association with target (V=0.0067, p=0.5332)]; identifiers ['sending_address', 'receiving_address'] and raw timestamp removed
7. **Split** - stratified, test=30%: train 55,020 / test 23,580; class share train {0: 0.8078, 1: 0.1096, 2: 0.0826} vs test {0: 0.8078, 1: 0.1095, 2: 0.0827}
8. **Fit transformer (train only)** - numeric: median-impute -> clip to (0.001, 0.999) quantiles -> standard scale; binary: passthrough; nominal: one-hot (unknown-safe); ordinal: age_group ['new', 'established', 'veteran'] -> 0..1 | 15 final features
9. **Imbalance** - class_weight='balanced' weights {0: 0.4126, 1: 3.0425, 2: 4.0343} (train class shares {0: 0.8078, 1: 0.1096, 2: 0.0826})

## Single-feature leakage scan

| feature | single-feature accuracy | baseline | verdict |
|---|---|---|---|
| risk_score | 1.000 | 0.807 | LEAK |
| transaction_type | 0.889 | 0.807 | strong |
| login_frequency | 0.807 | 0.807 | weak |
| session_duration | 0.807 | 0.807 | weak |
| location_region | 0.807 | 0.807 | weak |
| hour_of_day | 0.807 | 0.807 | weak |
| ip_prefix | 0.807 | 0.807 | weak |
| purchase_pattern | 0.807 | 0.807 | weak |
| is_weekend | 0.807 | 0.807 | weak |
| age_group | 0.807 | 0.807 | weak |
| day_of_week | 0.807 | 0.807 | weak |
| month | 0.807 | 0.807 | weak |
| is_night | 0.807 | 0.807 | weak |
| amount | 0.806 | 0.807 | weak |
| session_per_login | 0.806 | 0.807 | weak |
| amount_per_minute | 0.806 | 0.807 | weak |

## Categorical association with the target

| feature | Cramér's V | p-value | noise? |
|---|---|---|---|
| transaction_type | 0.7921 | 0.0 | False |
| location_region | 0.0075 | 0.3589 | True |
| ip_prefix | 0.0067 | 0.5332 | True |
| purchase_pattern | 0.434 | 0.0 | False |
| age_group | 0.434 | 0.0 | False |

## Dropped columns

- `sending_address`: identifier
- `receiving_address`: identifier
- `timestamp`: replaced by engineered time features
- `risk_score`: scenario 'no_score'
- `purchase_pattern`: duplicate of 'age_group' (Cramér's V=1.0)
- `location_region`: no association with target (V=0.0075, p=0.3589)
- `ip_prefix`: no association with target (V=0.0067, p=0.5332)

## Final feature groups

- **numeric** (8): amount, login_frequency, session_duration, hour_of_day, amount_per_minute, session_per_login, hour_sin, hour_cos
- **binary** (1): is_night
- **nominal** (1): transaction_type
- **ordinal** (1): age_group

Final model matrix: **15** columns.

## Rule probe (depth-4 tree, risk_score excluded)

```
|--- transaction_type_scam <= 0.50
|   |--- transaction_type_phishing <= 0.50
|   |   |--- transaction_type_sale <= 0.50
|   |   |   |--- class: 0
|   |   |--- transaction_type_sale >  0.50
|   |   |   |--- hour_of_day <= 6.50
|   |   |   |   |--- class: 1
|   |   |   |--- hour_of_day >  6.50
|   |   |   |   |--- class: 0
|   |--- transaction_type_phishing >  0.50
|   |   |--- class: 2
|--- transaction_type_scam >  0.50
|   |--- class: 2
```