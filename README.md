# Logistic-Regression-Project
This project applies **Logistic Regression** to predict whether an internet user will click on an online advertisement based on their browsing behavior and demographic information.
It uses a **synthetic (fake)** advertising dataset provided for educational purposes as part of the Python for Data Science and Machine Learning Bootcamp.

The main objective is to build a classification model that distinguishes between users who **clicked on an ad (1) and those who did not (0).**

## Dataset Description
The dataset contains various user behavior and demographic attributes, along with a binary target variable (Clicked on Ad).

## Features
- **Daily Time Spent on Site:** Time (in minutes) the consumer spends on the website per day
- **Age:** Customer age in years
- **Area Income:** Average income of the consumer’s geographical area
- **Daily Internet Usage:** Average daily minutes the user spends online
- **Ad Topic Line:** Headline of the advertisement
- **City:** City of the consumer
- **Male:** 1 if male, 0 if female
- **Country:** Country of the consumer
- **Timestamp:** Time when the user clicked on the ad or closed the page
- **Clicked on Ad:** Target variable — 1 if the user clicked, 0 otherwise
  
## Objectives
-Explore and clean the dataset
-Perform **Exploratory Data Analysis (EDA)** with visualizations
-Train a **Logistic Regression** model to predict ad-click probability
-Evaluate model performance using accuracy, confusion matrix, and classification report
-Gain insights into which user attributes influence ad engagements
  
## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- PyCharm

## How to Run
1. Clone the repository:  
```bash
git clone <your-repo-url>
```
2. Install dependencies:
 ```bash
pip install <name of libraries>
```
3. Run the main Python script:
 ```bash
python logistic_regression_project.py
```

## Notes
- The dataset is **synthetic** and meant for **educational use only**
- Focuses on **binary classification** and understanding model interpretation
- A great introduction to **Logistic Regression** and predictive modeling with Python
