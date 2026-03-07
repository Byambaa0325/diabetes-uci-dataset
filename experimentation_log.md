### Objective

In this project, we aim to train a "good" classifier for a medical classification dataset, "Diabetes 130-US Hospitals for Years 1998-2008". We will analyze, preprocess, and prepare the dataset for a classification task. Then, we will experiment multiple classification model against the 3 category of outcomes to predict the hospitalization of the patient from the encounter record.

### The Dataset

The dataset comprises of multivariate dataset of 47 features and 101766 records that were recorded during ten years period[1]. The features consist of patient, hospital outcomes, and information extracted from the database for encounters according to following criteria:

- A hospital admission
- A diabetic diagnosis in the system
- Length of stay was at least 1 day and at most 14 days
- Lab tests were performed
- Medication was administered.

### Notes

**[Exploratory Data Analysis]** The full dataset was loaded locally ana analyzed. 

- We understand that 50 columns are present for 101766 rows.
- Moreover, the three target classes are imbalanced (53.9, 34.9, 11.2) for No, above 30 and below 30 days respectively.
- There are several features that are missing from 39.5% to 96.8% of the data missing. We will need to drop this or salvage if necessary.
- Domain wise, patients can be revisit, multiple records per patient. It's considered a leakage risk.
- We also see imbalances in age and race.
- We see medication prescription rate is also imbalanced.
- No strong correlation to the target.

The problem is that imbalance for below 30 day revisit is sever at 11%. These are positive classes that the domain use case would want to know.

**Implications for the metrics** Recall on the positive classes will need to be considered first. The paper [1] says goal is to detect early admission within 30 days of discharge for various reasons listed for the care and survivability of the patient.The cost of missing an early admission case is worse than false positive case. Secondly, F1 helps us gain balanced view of the recall and precision. Finally, ROC-AUC would help us understand the classifier before tuning the threshold for F1 score. We would like to know the performance across different thresholds for the classification however, recall and F1 are prioritized.

Feature engineering: For the intital set of experiments, we use basic feature engineering without custom features to understand the baseline performance of the estimators.

**Experiment 1:** We train a MLP to classify the dataset. The validation loss is volatile reaching 0.239 at lowest and early threshold would trigger on 0.24. The ROC score

Pipeline: I am refactoring the codebase to be modular in line with FullStackDeepLearning course format and other best practice sources. The new design would allow us to experiment with the baselines faster for ablation and parameter sweeps.

### References

1. include citation dataset
