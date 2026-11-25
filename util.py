import tensorflow as tf
import numpy as np
import pandas as pd


def nasa_loss(y_true, y_pred):
    # Flatten both to 1D so (n, 1) and (n,) can't broadcast into an n x n grid
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    d = y_pred - y_true  # positive = predicted too late (overestimate)

    penalties = tf.where(
        d < 0,
        tf.exp(-d / 13.0) - 1.0,  # early prediction, milder penalty
        tf.exp(d / 10.0) - 1.0    # late prediction, harsher penalty
    )

    return tf.reduce_mean(penalties)


def create_windows(df, feature_cols, window_length=30):
    """Build every sliding window for every engine.

    Returns
    -------
    X : array (n_windows, window_length, n_features)
    y : array (n_windows,) RUL at each window's last cycle, or empty if df has no 'rul' column
    index : DataFrame (n_windows rows) with the 'engine' and 'cycle' each window ends on.
        Rows are sorted by engine then cycle, so the last row per engine is its latest window.
    """

    X, y, engines, cycles = [], [], [], []

    for engine_id, engine_df in df.groupby('engine'):

        engine_df = engine_df.sort_values('cycle')
        features = engine_df[feature_cols].to_numpy()

        # Engines shorter than one window: pad the front by repeating the first cycle
        if len(features) < window_length:
            pad = np.repeat(features[:1], window_length - len(features), axis=0)
            features = np.vstack([pad, features])
            end_rows = [len(features) - 1]
        else:
            end_rows = range(window_length - 1, len(features))

        windows = np.lib.stride_tricks.sliding_window_view(
            features, window_length, axis=0
        ).transpose(0, 2, 1)  # -> (n_windows, window_length, n_features)
        X.append(windows)

        engine_cycles = engine_df['cycle'].to_numpy()
        cycles.extend(engine_cycles[-len(windows):])
        engines.extend([engine_id] * len(windows))

        if 'rul' in engine_df.columns:
            y.extend(engine_df['rul'].to_numpy()[-len(windows):])

    index = pd.DataFrame({'engine': engines, 'cycle': cycles})
    return np.concatenate(X), np.array(y), index


def mc_dropout_predict(model, X, n_samples=50, batch_size=4096, interval=0.90):
    """Monte Carlo dropout: run the model n_samples times with dropout left on.

    Each window is repeated n_samples times and pushed through in large batches;
    dropout draws an independent mask per row, so every copy is a separate sample.

    Returns a DataFrame with one row per window: mean, std, and the lower/upper
    quantiles of the central `interval` (e.g. 5th and 95th percentile for 0.90).
    """
    repeated = np.repeat(X, n_samples, axis=0).astype(np.float32)
    samples = np.concatenate([
        np.asarray(model(repeated[i:i + batch_size], training=True)).reshape(-1)
        for i in range(0, len(repeated), batch_size)
    ]).reshape(len(X), n_samples)

    tail = (1 - interval) / 2
    return pd.DataFrame({
        'mc_mean': samples.mean(axis=1),
        'mc_std': samples.std(axis=1),
        'mc_lower': np.quantile(samples, tail, axis=1),
        'mc_upper': np.quantile(samples, 1 - tail, axis=1),
    })


def scale_interval(band, scale):
    """Widen an mc_dropout_predict band around its mean by a calibration factor.

    `scale` comes from the calibration step in preprocess.ipynb (calibration.json):
    each half-width (mean -> lower, mean -> upper) is multiplied by it.
    """
    band = band.copy()
    center = band['mc_mean']
    band['mc_lower'] = center - scale * (center - band['mc_lower'])
    band['mc_upper'] = center + scale * (band['mc_upper'] - center)
    return band


def trajectory_stability(cycles, predictions, lookback=30, cap=125, cap_margin=15):
    """How consistently the recent predictions decline at 1 RUL per cycle.

    In the degradation phase, true RUL drops by exactly 1 each cycle, so
    prediction + cycle should be constant. The spread of that sum (std, in
    cycles) over the last `lookback` windows measures how jumpy the model is.

    Returns None while the engine is still near the RUL cap (not degrading yet),
    because a flat prediction there is expected and the score is meaningless.
    """
    cycles = np.asarray(cycles)[-lookback:]
    predictions = np.asarray(predictions)[-lookback:]

    if len(predictions) < 5 or np.median(predictions) > cap - cap_margin:
        return None

    return float(np.std(predictions + cycles))


def last_windows(X, index):
    """Keep only the latest window for each engine (rows stay in engine order)."""
    last = index.groupby('engine').tail(1).index.to_numpy()
    return X[last], index.loc[last].reset_index(drop=True)


def local_feature_importance(model, data, feature_cols):

    X, _, index = create_windows(data, feature_cols)
    X, _ = last_windows(X, index)  # importance for the current (latest) window

    # Batch the baseline and all 17 "feature zeroed" versions into one predict call
    variants = [X[0]]
    for i in range(len(feature_cols)):
        x_mod = X[0].copy()
        x_mod[:, i] = 0
        variants.append(x_mod)

    preds = model.predict(np.stack(variants), verbose=0).flatten()
    original_prediction = preds[0]

    results = []

    for i, feature in enumerate(feature_cols):

        mod_pred = preds[i + 1]

        prediction_difference = mod_pred - original_prediction

        results.append({
            'feature' : feature,
            'predicted_rul' : mod_pred,
            'prediction_change' : prediction_difference,
            'importance' : abs(prediction_difference)
        })

    results = pd.DataFrame(results)

    results = results.sort_values('importance', ascending=False)
    return results

AVG_HRS_PER_CYCLE=5

def calculate_reactive_maintenance_cost(labor_cost, fix_time, inv_time, aircraft_ev):
    missed_cycles = (inv_time + fix_time) / AVG_HRS_PER_CYCLE


    return ((labor_cost + aircraft_ev) * (fix_time + inv_time))


def calculate_predictive_maintenance_cost(labor_cost, fix_time, aircraft_ev, engine_cost, engine_life, missed_cycles):
    """
    engine_life: per-engine estimated total life in cycles (age + predicted RUL).
    missed_cycles: cycles of life thrown away per engine by replacing early (scalar or per-engine array).
    """
    engine_life = np.asarray(engine_life, dtype=float)
    # Early replacement doesn't ground the aircraft; it throws away unused engine life,
    # so missed cycles are priced at each engine's cost per cycle, not the aircraft's earnings.
    engine_cost_per_cycle = engine_cost / engine_life
    downtime = (labor_cost + aircraft_ev) * fix_time * len(engine_life)
    return downtime + np.sum(engine_cost_per_cycle * missed_cycles)


def remaining_life_value(age, predicted_rul, engine_cost):
    """Dollar value of an engine's predicted remaining life: cost per cycle (from age + RUL) x RUL."""
    engine_life = age + predicted_rul
    return engine_cost / engine_life * predicted_rul if engine_life > 0 else 0.0


def aircraft_replacement_recommendation(engines, engine_cost, labor_cost, fix_time, aircraft_ev):
    """
    Decide which of an aircraft's engines to replace now.

    engines: list of dicts with 'status', 'age' and 'predicted_rul', one per engine.

    - If any engine is critical, it is replaced, and the aircraft is going down anyway.
      Each non-critical engine is also replaced if its remaining life is worth less than
      the future downtime event that replacing it now (in the same downtime) avoids.
    - If no engine is critical, replacing now needs its own downtime event. Both engines
      are replaced together if their combined remaining life is worth less than one
      downtime event (one event now instead of two later); otherwise both are kept.
    """
    downtime_cost = (labor_cost + aircraft_ev) * fix_time
    values = [remaining_life_value(e['age'], e['predicted_rul'], engine_cost) for e in engines]
    any_critical = any(e['status'] == 'Critical' for e in engines)

    recs = []
    if any_critical:
        for e, value in zip(engines, values):
            # engine = e['engine']
            if e['status'] == 'Critical':
                replace, reason = True, 'Engine is critical.'
            elif value < downtime_cost:
                replace, reason = True, ('Replacing it during the critical engine\'s downtime is cheaper '
                                         'than keeping its remaining life and grounding the aircraft again later.')
            else:
                replace, reason = False, ('Its remaining life is worth more than the future downtime '
                                          'saved by replacing it alongside the critical engine.')
            recs.append({'replace': replace, 'reason': reason, 'rul_value': value})
    else:
        replace_both = sum(values) < downtime_cost
        reason = ('Combined remaining life of both engines is worth less than one downtime event, '
                  'so replace both together now.' if replace_both else
                  'Combined remaining life of both engines is worth more than one downtime event.')
        recs = [{'replace': replace_both, 'reason': reason, 'rul_value': v} for v in values]

    return {
        'engines': recs,
        'downtime_cost': downtime_cost,
        'combined_rul_value': sum(values),
        'any_critical': any_critical,
    }
