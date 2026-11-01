# 🎬 Movie Rating Prediction

CodSoft Data Science Internship, **Task 2**

Predicts a movie's IMDb rating (1 to 10) from its genre, director, cast, year and duration.

## Try it
Open `Movie_Rating_Prediction.ipynb`, run all cells, and use the **dropdown form** in the last cell:
pick a director and actors from the list, choose genres, and click **Predict rating**.
You get the rating, a likely range, and the reasons behind it.

## Results
| Model | Test MAE | Test R² |
|---|---|---|
| Always guess the average | 1.11 | 0.00 |
| Gradient Boosting (final) | **0.78** | **0.43** |
| Gradient Boosting, no votes (for new movies) | 0.86 | 0.35 |

- 43% of predictions are within 0.5 stars, and 71% are within 1 star.
- Ratings depend on story and acting, which are not in the data, so about 43% is a realistic result.

## What I did
- **Data:** IMDb India Movies, 15,509 movies. Kept the 7,919 that have a rating.
- **Cleaning:** turned "(2019)", "109 min" and "20,942" into numbers, and fixed a stray "$5.16M" in the votes column.
- **Star power features:** each director's and cast's track record (average rating of their other movies).
  Calculated only from training data, so the model never sees the answer.
- **Models compared:** Ridge, Random Forest, Gradient Boosting. Random Forest overfit (train R² 0.82 vs test 0.42),
  so Gradient Boosting won on balance.
- **Explained with SHAP:** cast and director track record matter most, then votes and release year.
- **Honest note on votes:** votes only exist after release, so the predictor uses the no-votes model.



## Files
- `codsofttask2.ipynb`: full notebook
- `IMDb Movies India.csv`: dataset (from Kaggle, adrianmcmahon/imdb-india-movies)

#codsoft #datascience #machinelearning
