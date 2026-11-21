# task07_covertype

UCI Covertype: cartographic variables for 30x30 m cells in Roosevelt National Forest; predict forest cover type. Wilderness_Area and Soil_Type columns are binary indicators.

Stratified random 70/20/10 split (seed 471).

| Split | Rows | Share |
| --- | --- | --- |
| train | 406708 | 70.0% |
| dev | 116202 | 20.0% |

| Class | train | dev |
| --- | --- | --- |
| 0 = Spruce/Fir | 36.5% | 36.5% |
| 1 = Lodgepole Pine | 48.8% | 48.8% |
| 2 = Ponderosa Pine | 6.2% | 6.2% |
| 3 = Cottonwood/Willow | 0.5% | 0.5% |
| 4 = Aspen | 1.6% | 1.6% |
| 5 = Douglas-fir | 3.0% | 3.0% |
| 6 = Krummholz | 3.5% | 3.5% |

**Files:** `{train,dev}.X` and `.CT`. `feature_names.txt` names each `.X` column; `classes.txt` maps label ids to class names.

This package holds the train and dev splits. Test inputs are released separately; see the course schedule.
