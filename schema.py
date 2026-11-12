"""Single source of truth for the feature contract.

Both ``train.py`` and ``app.py`` import this module so the columns the model is
trained on can never drift from the columns the API accepts.  The ranges here
mirror the data-contract table in README.md and the Pydantic schema in app.py.
"""

NUMERIC_FEATURES = [
    "speed_limit",
    "num_vehicles",
    "num_casualties",
    "driver_age",
    "hour",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "weather",
    "road_type",
    "light_condition",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "severity"

# Allowed values, enforced in app.py's Pydantic schema.
WEATHER_VALUES = ["clear", "rain", "fog", "snow"]
ROAD_TYPE_VALUES = ["urban", "rural", "highway", "intersection"]
LIGHT_CONDITION_VALUES = ["daylight", "dark", "dusk"]
