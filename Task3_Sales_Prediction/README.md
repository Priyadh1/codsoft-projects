# 💜 Sales Studio: Sales Prediction using Python

CodSoft Data Science Internship, **Task 3**

## 👉 [OPEN THE LIVE APP](https://codsoft-priyadh-dadpf3r37tuj6hp7wqn2sk.streamlit.app/)

https://codsoft-priyadh-dadpf3r37tuj6hp7wqn2sk.streamlit.app/

Predicts product sales from the advertising budget spent on TV, Radio and Newspaper,
and recommends the best way to split a budget. If the app is asleep, click the wake-up button and wait a few seconds.

## What the app does
- 🎛️ **Simulator:** move the TV, Radio and Newspaper sliders to see predicted sales, a likely range, and where the next 10 thousand dollars should go.
- 💰 **Budget Planner:** enter a total budget and get the best split, compared with spending evenly or all on TV.
- 📊 **Insights:** what drives sales, the model comparison, and the limits of the data.

## Results (5-fold cross-validation)
| Model | R² | Average error |
|---|---|---|
| Linear: TV only (common starting point) | 0.59 | 2.58 |
| Linear: TV + Radio | 0.89 | 1.29 |
| Linear: all 3 channels | 0.89 | 1.31 |
| Random Forest | 0.97 | 0.61 |
| **Gradient Boosting (final)** | **0.97** | **0.60** |

## Key findings
- **TV** is the strongest driver of sales, then **Radio**.
- **Newspaper adds almost nothing**: adding it did not improve the model.
- TV and Radio work **together**, so tree models beat straight-line models.

## What I did
- Data: Advertising dataset, 200 markets, no missing values or duplicates.
- Started from the TV-only linear regression and extended it to all channels and tree models.
- Checked overfitting by comparing train, test and cross-validation scores.
- Built and deployed the app with Streamlit.
- Limits: only 200 markets, and the data shows association, not proof of cause.

## Files
- `Sales_Prediction.ipynb`: analysis notebook
- `app.py`, `requirements.txt`: the Streamlit app
- `CodSoft_Sales_Prediction_Dataset.xlsx`: dataset

#codsoft #datascience #machinelearning
