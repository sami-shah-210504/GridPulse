from datetime import timedelta

from feast import FeatureService, FeatureView

from entities import region
from features import FEATURE_FIELDS, source

energy_fv = FeatureView(
    name="energy_features",
    entities=[region],
    ttl=timedelta(0),  # 0 = features never expire
    schema=FEATURE_FIELDS,
    source=source,
    tags={"owner": "sami", "freshness": "hourly", "source": "EIA + Open-Meteo"},
)

# Stable, versioned bundle that training and serving both request
energy_service = FeatureService(name="energy_demand_v1", features=[energy_fv])
