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