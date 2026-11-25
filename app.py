import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from util import (
    nasa_loss, local_feature_importance, create_windows,
    mc_dropout_predict, scale_interval, trajectory_stability, calculate_reactive_maintenance_cost,
    calculate_predictive_maintenance_cost, aircraft_replacement_recommendation
)
import json
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

feature_cols = [2, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
feature_cols = [str(index) for index in feature_cols]

RUL_CAP = 125
Z_95 = 1.645  # one-sided 95% safety buffer on the prediction

# (minimum predicted RUL, status, color), checked top to bottom
STATUS_LEVELS = [
    (100, 'Healthy', '#2e7d32'),   # green
    (75, 'Watch', '#7cb342'),      # light green
    (50, 'Plan', '#f9a825'),       # yellow
    (25, 'Worn', '#ef6c00'),       # orange
    (0, 'Critical', '#c62828'),    # red
]
STATUS_COLORS = {status: color for _, status, color in STATUS_LEVELS}


def health_status(predicted_rul):
    for threshold, status, _ in STATUS_LEVELS:
        if predicted_rul >= threshold:
            return status
    return 'Critical'


# Stability = std (in cycles) of the recent predictions around a steady 1-per-cycle decline
STABILITY_LEVELS = [
    (5, 'Stable', '#2e7d32'),
    (10, 'Somewhat jumpy', '#f9a825'),
    (float('inf'), 'Unstable', '#c62828'),
]


def stability_label(score):
    for limit, label, color in STABILITY_LEVELS:
        if score <= limit:
            return label, color


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        'single_lstm_64.keras',
        custom_objects={'nasa_loss': nasa_loss}
    )


@st.cache_data
def load_data():

    test = pd.read_csv('test.csv')
    rul = pd.read_csv('rul.csv')
    # rul.csv has one row per test engine, in engine order (engine 1 first)
    true_rul = rul['0'].to_numpy()
    return test, true_rul


@st.cache_data
def load_calibration():
    """Band scale factor from the calibration step in preprocess.ipynb, or None if missing."""
    try:
        with open('calibration.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def build_fleet_table(_model, test, true_rul, interval_scale):
    """Predict every engine in one batch and compute its maintenance metrics."""

    # Predict on every sliding window for every engine...
    windows, _, window_index = create_windows(test, feature_cols)
    window_index['predicted_rul'] = _model.predict(windows, batch_size=1024, verbose=0).flatten()

    # ...but only report each engine's latest window (rows stay in engine order)
    latest = window_index.groupby('engine').tail(1).index.to_numpy()
    raw_predictions = window_index.loc[latest, 'predicted_rul'].to_numpy()

    # MC dropout uncertainty on each engine's latest window
    uncertainty = scale_interval(mc_dropout_predict(_model, windows[latest]), interval_scale)

    rmse = root_mean_squared_error(np.clip(true_rul, 0, RUL_CAP), raw_predictions)
    nasa = nasa_loss(np.clip(true_rul, 0, RUL_CAP), raw_predictions)
    mae = mean_absolute_error(np.clip(true_rul, 0, RUL_CAP), raw_predictions)
    margin = Z_95 * rmse

    age = test.groupby('engine').size()
    predicted_rul = np.maximum(0, np.floor(raw_predictions)).astype(int)
    replace_in = np.maximum(0, np.floor(predicted_rul - margin)).astype(int)

    engine_ids = age.index.to_numpy()

    fleet = pd.DataFrame({
        # Arbitrary pairing: engines 1-2 go on aircraft 1, engines 3-4 on aircraft 2, etc.
        'Aircraft': [f'AC-{(e - 1) // 2 + 1:03d}' for e in engine_ids],
        'Engine': engine_ids,
        'Age (cycles)': age.to_numpy(),
        'Predicted RUL': predicted_rul,
        'RUL Low (90%)': np.maximum(0, np.floor(uncertainty['mc_lower'])).astype(int),
        'RUL High (90%)': np.maximum(0, np.floor(uncertainty['mc_upper'])).astype(int),
        'Wear (%)': 100 * age.to_numpy() / (age.to_numpy() + predicted_rul),
        'Status': [health_status(r) for r in predicted_rul],
        'Replace In (cycles)': replace_in,
        'Replace By Cycle': age.to_numpy() + replace_in,
        # 'Actual RUL': true_rul,
    })

    fleet = fleet.sort_values(['Replace In (cycles)', 'Engine']).reset_index(drop=True)
    return fleet, rmse, margin, window_index, mae

# @st.cache_data
# def calculate_missed_cycles(_model, test, true_rul, interval_scale):
#     windows, _, window_index = create_windows(test, feature_cols)
#     window_index['predicted_rul'] = _model.preict(windows, batch_size=1024, verbose=0).flatten()
#     latest = window_index.groupby('engine').tail(1).index.to_numpy()
#     raw_preds = window_index.loc[latest, 'predicted_rul'].to_numpy()



@st.cache_data
def engine_uncertainty(_model, _test, engine_id, interval_scale):
    """Calibrated MC dropout band for every window of one engine (cached per engine)."""
    engine_data = _test[_test['engine'] == engine_id]
    windows, _, window_index = create_windows(engine_data, feature_cols)
    band = scale_interval(mc_dropout_predict(_model, windows), interval_scale)
    return pd.concat([window_index, band], axis=1)


model = load_model()
test, true_rul = load_data()
calibration = load_calibration()
interval_scale = calibration['interval_scale'] if calibration else 1.0
fleet, rmse, margin, index, mae = build_fleet_table(model, test, true_rul, interval_scale)


st.title('Engine Degradation Predictor')

# ---------------------------------------------------------------- Fleet overview
st.header('Fleet Overview')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Engines', len(fleet))
col2.metric('Replace Now', int((fleet['Replace In (cycles)'] == 0).sum()))
col3.metric('Critical', int((fleet['Status'] == 'Critical').sum()))
col4.metric('Safety Buffer', f'{margin:.0f} cycles')

st.caption(
    f'Replacement dates subtract a {margin:.0f}-cycle safety buffer '
    f'(1.645 × model RMSE of {rmse:.1f}) from the predicted RUL. '
    'Actual RUL is shown for hindsight comparison only.'
)

status_filter = st.multiselect(
    'Show statuses:',
    options=list(STATUS_COLORS),
    default=list(STATUS_COLORS),
)
shown = fleet[fleet['Status'].isin(status_filter)]
shown = shown.drop(columns=['RUL Low (90%)', 'RUL High (90%)'])

styled = shown.style.map(
    lambda s: f'color: {STATUS_COLORS[s]}; font-weight: 600',
    subset=['Status'],
)



st.dataframe(
    styled,
    hide_index=True,
    use_container_width=True,
    # Short labels + 'small' widths so all columns fit without horizontal scrolling;
    # the full meaning is in each column's hover tooltip.
    column_config={
        'Aircraft': st.column_config.TextColumn('Aircraft', width='small'),
        'Engine': st.column_config.NumberColumn('Engine', width='small'),
        'Age (cycles)': st.column_config.NumberColumn(
            'Age', width='small', help='Age (cycles)'
        ),
        'Predicted RUL': st.column_config.NumberColumn(
            'Pred. RUL', width='small', help='Predicted remaining useful life (cycles)'
        ),
        'Wear (%)': st.column_config.ProgressColumn(
            'Wear', format='%.0f%%', min_value=0, max_value=100, width='small',
            help='Age as a share of estimated total life'
        ),
        'Status': st.column_config.TextColumn('Status', width='small'),
        'Replace In (cycles)': st.column_config.NumberColumn(
            'Replace In', width='small', help='Cycles until replacement (predicted RUL minus safety buffer)'
        ),
        'Replace By Cycle': st.column_config.NumberColumn(
            'Replace By', width='small', help='Engine age (cycles) at which to replace'
        ),
    },
)

st.divider()

st.header('Fleet Maintenance Cost Calculator')

col1, col2, col3, col4 = st.columns(4)
labor_cost = col1.number_input('Labor Cost ($/hr)')
fix_time = col2.number_input('Time to Fix (hrs)')
inv_time = col3.number_input('Time for Causal Investigation (hrs)')
aircraft_ev = col4.number_input('Aircraft Economic Value ($/hr)')
engine_cost = st.number_input('Engine Cost ($)')






reactive_cost = calculate_reactive_maintenance_cost(
    labor_cost,
    fix_time,
    inv_time,
    aircraft_ev
) * len(fleet)

# Each engine's estimated total life, from its current age plus the model's predicted RUL
engine_life = fleet['Age (cycles)'] + fleet['Predicted RUL']
missed_cycles = mae  # per engine

predictive_cost = calculate_predictive_maintenance_cost(
    labor_cost,
    fix_time,
    aircraft_ev,
    engine_cost,
    engine_life,
    missed_cycles,
)


def format_dollars(amount):
    sign = '-' if amount < 0 else ''
    amount = abs(amount)
    if amount >= 1e6:
        return f'{sign}${amount / 1e6:.2f}M'
    return f'{sign}${amount / 1e3:.2f}k'





col1, col2, col3 = st.columns(3)
col1.metric('Reactive Maint. Cost', format_dollars(reactive_cost))
col2.metric('Predictive Maint. Cost', format_dollars(predictive_cost))

savings = reactive_cost - predictive_cost
if reactive_cost > 0:
    # Shown as the change in cost (negative = cheaper), so a saving gets a down arrow;
    # 'inverse' colors that down arrow green and a cost increase red.
    cost_change = -100 * savings / reactive_cost
    delta = f'{cost_change:.2f}%'
else:
    delta = None
col3.metric(
    'Predictive Savings',
    format_dollars(savings),
    delta=delta,
    delta_color='inverse',
)




st.divider()



# ------------------------------------------------- Aircraft replacement advice
st.header('Aircraft Replacement Recommendation')

aircraft_id = st.selectbox('Choose an aircraft:', sorted(fleet['Aircraft'].unique()))
aircraft_engines = fleet[fleet['Aircraft'] == aircraft_id].sort_values('Engine')

advice = aircraft_replacement_recommendation(
    [
        {'status': e['Status'], 'age': e['Age (cycles)'], 'predicted_rul': e['Predicted RUL']}
        for _, e in aircraft_engines.iterrows()
    ],
    engine_cost,
    labor_cost,
    fix_time,
    aircraft_ev,
)

if advice['any_critical']:
    st.caption(
        f"A critical engine grounds the aircraft now; replacing another engine in the same downtime "
        f"avoids a future downtime event worth {format_dollars(advice['downtime_cost'])}."
    )
else:
    st.caption(
        f"No critical engines. Replacing now needs its own downtime event "
        f"({format_dollars(advice['downtime_cost'])}), so both engines are evaluated together: "
        f"combined remaining life is worth {format_dollars(advice['combined_rul_value'])}."
    )

for col, (_, eng), rec in zip(st.columns(len(aircraft_engines)), aircraft_engines.iterrows(), advice['engines']):
    with col:
        st.markdown(
            f"**Engine {eng['Engine']}** · "
            f"<span style='color: {STATUS_COLORS[eng['Status']]}; font-weight: 600'>{eng['Status']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Predicted RUL: {eng['Predicted RUL']} cycles")
        st.write(f"Value of remaining life: {format_dollars(rec['rul_value'])}")
        if rec['replace']:
            st.error(f"Replace now. {rec['reason']}")
        else:
            st.success(f"Keep flying. {rec['reason']}")


if True in [engine['replace'] for engine in advice['engines']]:
    st.error(f'GROUND {aircraft_id} NOW', title='Action Recommendation')
else:
    st.success(f'No action necessary', title='Action Recommendation')

# ---------------------------------------------------------------- Engine detail
st.divider()
st.header('Engine Detail')

engine_input = st.number_input(
    f'Choose an engine (1-{len(fleet)}):',
    min_value=1,
    max_value=len(fleet),
    value='min'
)

engine = fleet[fleet['Engine'] == engine_input].iloc[0]
data = test[test['engine'] == engine_input]

# st.markdown(f'#### Current Engine Age: {engine["Age (cycles)"]} cycles')
# st.markdown(
#     f'#### Predicted Remaining Useful Life: {engine["Predicted RUL"]} cycles '
#     f'(90% range: {engine["RUL Low (90%)"]}–{engine["RUL High (90%)"]})'
# )
# st.markdown(f'#### Estimated Percent Wear: {engine["Wear (%)"]:.1f}%')
# st.progress(engine['Wear (%)'] / 100)

# st.markdown(
#     f'#### Engine Health: <span style="color:{STATUS_COLORS[engine["Status"]]}">'
#     f'{engine["Status"]}</span>',
#     unsafe_allow_html=True,
# )

# if engine['Replace In (cycles)'] > 0:
#     st.markdown(
#         f'#### Replace in {engine["Replace In (cycles)"]} cycles '
#         f'(cycle {engine["Replace By Cycle"]})'
#     )
# else:
#     st.markdown(
#         f'### <span style="color:{STATUS_COLORS["Critical"]}">'
#         'REPLACE ENGINE NOW, DO NOT USE</span>',
#         unsafe_allow_html=True,
#     )

col1, col2, col3 = st.columns(3)
col1.metric('Current Engine Age', f'{engine["Age (cycles)"]} cycles')
col2.metric('Estimated RUL', f'{engine["Predicted RUL"]} cycles')
col3.metric('90% RUL Range', f'{engine["RUL Low (90%)"]}–{engine["RUL High (90%)"]} cycles')



col1, col2, col3 = st.columns(3)
col1.metric('Engine Health', f'{engine["Status"]}')
col2.metric('Replace In', f'{engine["Replace In (cycles)"]} cycles')
col3.metric('Replacement Cycle', f'Cycle {engine["Replace By Cycle"]}' if engine["Replace In (cycles)"] > 0
            else "REPLACE NOW")

st.markdown(f'#### Estimated Percent Wear: {engine["Wear (%)"]:.1f}%')
st.progress(engine['Wear (%)'] / 100)

st.divider()
predictions = index[index['engine'] == engine_input].reset_index(drop=True)
band = engine_uncertainty(model, test, engine_input, interval_scale)

st.subheader('Predicted RUL Over Time')

stability = trajectory_stability(predictions['cycle'], predictions['predicted_rul'])
if stability is None:
    st.markdown('#### Prediction Stability: not yet degrading (predictions near the 125-cycle cap)')
else:
    label, color = stability_label(stability)
    st.markdown(
        f'#### Prediction Stability: <span style="color:{color}">{label}</span> '
        f'(±{stability:.1f} cycles over the last 30 predictions)',
        unsafe_allow_html=True,
    )

fig = go.Figure([
    go.Scatter(
        x=band['cycle'], y=band['mc_upper'],
        mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False,
    ),
    go.Scatter(
        x=band['cycle'], y=band['mc_lower'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)', name='90% range (MC dropout)',
    ),
    go.Scatter(
        x=predictions['cycle'], y=predictions['predicted_rul'],
        mode='lines', line=dict(color='#1f77b4'), name='Predicted RUL',
    ),
])

fig.update_layout(xaxis_title='Cycle', yaxis_title='Predicted RUL', hovermode='x unified')
fig.update_yaxes(range=[0, max(band['mc_upper'].max(), predictions['predicted_rul'].max()) + 5])

fig.add_hline(
    y=25,
    line_color='red',
    annotation_text="Critical Point (25 cycles left)",
    annotation_position='top left'
)

st.plotly_chart(fig, use_container_width=True)

if calibration:
    st.caption(
        f'Shaded band: MC dropout range widened {interval_scale:.2f}× so that '
        f'{calibration["interval"]:.0%} of validation engines\' true RUL falls inside it '
        f'(uncalibrated, only {calibration["raw_coverage"]:.0%} did).'
    )
else:
    st.warning(
        'calibration.json not found: the shaded band is uncalibrated and too narrow. '
        'Run the calibration cell in preprocess.ipynb.'
    )

# st.subheader(f'Actual Remaining Useful Life: {engine["Actual RUL"]} cycles')

feature_importance = local_feature_importance(model, data, feature_cols)

# st.subheader('Most Important Sensors')
# for row in range(len(feature_importance.head(10))):
#     feature = f'Sensor {feature_importance.iloc[row]["feature"]}'
#     importance = feature_importance.iloc[row]['prediction_change']
#     sign = '+' if importance > 0 else ''
#     color = 'red' if importance < 0 else 'green'
#     importance = f'{sign}{importance:.2f}'
#     st.write(f'{feature}: :{color}[{importance}] cycles')
