# task08_har

UCI Human Activity Recognition: 561 precomputed time/frequency features from smartphone accelerometer and gyroscope windows; predict the activity. The original train/test files are pooled and re-split; subject ids are not provided. Features are already normalized to [-1, 1] by the dataset authors.

Split **by subject** (seed 471): each of the 30 volunteers is entirely in one split (21 train, 6 dev, 3 test), so the model is tested on people it has never seen. Windows overlap 50%, so a random split would leak near-copies.

| Split | Rows | Share |
| --- | --- | --- |
| train | 7220 | 70.1% |
| dev | 2051 | 19.9% |

| Class | train | dev |
| --- | --- | --- |
| 0 = WALKING | 16.9% | 16.5% |
| 1 = WALKING_UPSTAIRS | 15.0% | 15.2% |
| 2 = WALKING_DOWNSTAIRS | 13.5% | 13.9% |
| 3 = SITTING | 17.2% | 17.4% |
| 4 = STANDING | 18.4% | 18.6% |
| 5 = LAYING | 19.0% | 18.4% |

**Files:** `{train,dev}.X` and `.CT`. `feature_names.txt` names each `.X` column; `classes.txt` maps label ids to class names.

This package holds the train and dev splits. Test inputs are released separately; see the course schedule.
