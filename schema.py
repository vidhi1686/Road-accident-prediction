"""Single source of truth for the accident-severity feature contract.

The cleaned dataset produced by ``data/preprocess.py`` contains exactly the
columns listed here.  ``train.py`` trains on these features, ``app.py`` accepts
them, and ``drift_check.py`` monitors them.

Some of these fields (for example ``num_casualties`` and ``type_of_collision``)
are accident-report information and may only be known after the collision has
occurred.  This project performs accident-severity classification from report
data; it is not a pre-accident crash predictor.
"""

NUMERIC_FEATURES = [
    "hour",
    "is_weekend",
    "num_vehicles",
    "num_casualties",
]

ORDINAL_FEATURES = [
    "age_band_of_driver",
    "driving_experience",
]

NOMINAL_FEATURES = [
    "sex_of_driver",
    "type_of_vehicle",
    "area_accident_occurred",
    "types_of_junction",
    "road_surface_type",
    "road_surface_conditions",
    "light_conditions",
    "weather_conditions",
    "type_of_collision",
]

CATEGORICAL_FEATURES = ORDINAL_FEATURES + NOMINAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "severe"

# Explicit, verified ordering for the two genuinely ordinal fields.  The
# ordering comes from the source definitions in the original notebook/CSV and
# is used by OrdinalEncoder.  Nominal fields deliberately have no ordering.
ORDINAL_CATEGORIES = [
    [
        "Under 18",
        "18-30",
        "31-50",
        "Over 51",
        "Unknown",
    ],
    [
        "No Licence",
        "Below 1yr",
        "1-2yr",
        "2-5yr",
        "5-10yr",
        "Above 10yr",
        "Unknown",
    ],
]

# Verified against the cleaned dataset's value counts.  Missing values are not
# listed here: they are imputed inside the training pipeline.  Genuine
# "Unknown" categories from the source data are preserved.
CATEGORY_VALUES = {
    "age_band_of_driver": [
        "Under 18",
        "18-30",
        "31-50",
        "Over 51",
        "Unknown",
    ],
    "driving_experience": [
        "No Licence",
        "Below 1yr",
        "1-2yr",
        "2-5yr",
        "5-10yr",
        "Above 10yr",
        "Unknown",
    ],
    "sex_of_driver": [
        "Female",
        "Male",
        "Unknown",
    ],
    "type_of_vehicle": [
        "Automobile",
        "Bajaj",
        "Bicycle",
        "Long lorry",
        "Lorry (11?40Q)",
        "Lorry (41?100Q)",
        "Motorcycle",
        "Other",
        "Pick up upto 10Q",
        "Public (12 seats)",
        "Public (13?45 seats)",
        "Public (> 45 seats)",
        "Ridden horse",
        "Special vehicle",
        "Stationwagen",
        "Taxi",
        "Turbo",
    ],
    "area_accident_occurred": [
        "Church areas",
        "Hospital areas",
        "Industrial areas",
        "Market areas",
        "Office areas",
        "Other",
        "Outside rural areas",
        "Recreational areas",
        "Residential areas",
        "Rural village areas",
        "Rural village areasOffice areas",
        "School areas",
        "Unknown",
    ],
    "types_of_junction": [
        "Crossing",
        "No junction",
        "O Shape",
        "Other",
        "T Shape",
        "Unknown",
        "X Shape",
        "Y Shape",
    ],
    "road_surface_type": [
        "Asphalt roads",
        "Asphalt roads with some distress",
        "Earth roads",
        "Gravel roads",
        "Other",
    ],
    "road_surface_conditions": [
        "Dry",
        "Flood over 3cm. deep",
        "Snow",
        "Wet or damp",
    ],
    "light_conditions": [
        "Darkness - lights lit",
        "Darkness - lights unlit",
        "Darkness - no lighting",
        "Daylight",
    ],
    "weather_conditions": [
        "Cloudy",
        "Fog or mist",
        "Normal",
        "Other",
        "Raining",
        "Raining and Windy",
        "Snow",
        "Unknown",
        "Windy",
    ],
    "type_of_collision": [
        "Collision with animals",
        "Collision with pedestrians",
        "Collision with roadside objects",
        "Collision with roadside-parked vehicles",
        "Fall from vehicles",
        "Other",
        "Rollover",
        "Unknown",
        "Vehicle with vehicle collision",
        "With Train",
    ],
}
