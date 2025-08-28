# Flight Delay Predictor
## A machine learning model that predicts flight delays and cancelations using historical flight data

# Overview
Flight delays are a common problem in the aviation industry. Not only does it inconvenience passengers, but also affects airline operations. This project uses machine learning to predict flight delays and cancellations using historical flight data. It leverages data preprocessing, feature engineering, and classification models to identify flights likely to be delayed.

## Tech Stack
- **Python**: Data analysis, preprocessing, and modeling
- **Jupyter Notebook / VS Code**: Interactive development and visualization
- **pandas**: Data manipulation
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning and preprocessing
- **xgboost**: Advanced modeling (optional)
- **numpy**: Numerical operations

## Files
- `FlightDelayNotebook.ipynb`: Main notebook for data analysis, feature engineering, model training, and evaluation. 
- `flights.csv/flights.csv`: Main dataset containing flight records.

## Workflow
1. **Data Loading**: Loads flight data from `flights.csv`.
2. **Sampling**: Samples 1,000,000 records for analysis.
3. **Preprocessing**: Handles missing values and selects relevant columns.
4. **Feature Engineering**:
	- Time-based features (hour of day, cyclical encoding).
	- Delay/cancellation labeling.
	- One-hot encoding for airlines and airports.
	- Aggregated delay statistics per airline and airport.
5. **Train/Test Split**: Splits data into training and testing sets (80/20).
6. **Scaling**: Scales numerical features, passes through categorical features.
7. **Modeling**: Trains a Random Forest classifier to predict delays.
8. **Evaluation**: Reports accuracy, confusion matrix, and classification metrics.



## Results
Validation Accuracy: 0.768555 


Confusion Matrix:                                      
<img width="533" height="432" alt="image" src="https://github.com/user-attachments/assets/05a6be91-b3db-4a3b-b2e8-daa196e943d2" />



The model achieves strong accuracy with class 0(on-time) flights, but its performance with class 1(delayed flights) is poor. It's recall for class 0 is high, at 0.95. For class 1, the recall is at 0.14. It is ineffective and cannot reliably detect a delay.

## Analysis and Report
Data imbalance is the most significant cause of the model's skewed performance. The on-time flights outnumber the delayed flights by a ratio of almost 3 to 1. I believe to improve this model's accuracy, one would need more informative data, rather than better algorithms or processing techniques. For example, weather is a major factor when predicting flight delays/cancellations in the real world, and this data is barely present here. However, this outcome illustrates the challenges of working with real-world, imbalanced data, and was a good educational experience. The skills I have learnt from this project are invaluable.

