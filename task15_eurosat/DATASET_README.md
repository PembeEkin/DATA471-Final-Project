# task15_eurosat

EuroSAT (Sentinel-2 RGB, 64x64): predict the land-use class of each image patch.

Stratified random 70/20/10 split (seed 471).

| Split | Rows | Share |
| --- | --- | --- |
| train | 18900 | 70.0% |
| dev | 5400 | 20.0% |

| Class | train | dev |
| --- | --- | --- |
| 0 = AnnualCrop | 11.1% | 11.1% |
| 1 = Forest | 11.1% | 11.1% |
| 2 = HerbaceousVegetation | 11.1% | 11.1% |
| 3 = Highway | 9.3% | 9.3% |
| 4 = Industrial | 9.3% | 9.3% |
| 5 = Pasture | 7.4% | 7.4% |
| 6 = PermanentCrop | 9.3% | 9.3% |
| 7 = Residential | 11.1% | 11.1% |
| 8 = River | 9.3% | 9.3% |
| 9 = SeaLake | 11.1% | 11.1% |

**Files:** `{train,dev}.X` and `.CT`. `classes.txt` maps label ids to class names.

**Layout:** each `.X` line is one 64x64 RGB image: 12288 integers in 0-255, in row-major
pixel order with the three channels of each pixel together, i.e. index
`(row * 64 + col) * 3 + channel` (channel 0 = R, 1 = G, 2 = B). In numpy:
`X.reshape(-1, 64, 64, 3)`.

This package holds the train and dev splits. Test inputs are released separately; see the course schedule.
