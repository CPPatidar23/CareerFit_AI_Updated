# CareerFit Streamlit App

## Folder structure

Keep these files in the same folder:

- app.py
- careerfit_logistic_regression.pkl
- careerfit_tfidf_vectorizer.pkl
- Career_Dataset.csv

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Important

The app follows the feature construction used in the CareerFit notebook:
Skills + Interests + Favorite Subjects + Personality + Career Goal
-> TF-IDF vectorizer -> Logistic Regression -> career probabilities.

The Education, Academic Stream and CGPA fields are displayed as part of the
student profile, but the saved model itself was trained from the combined text
fields above, so those three numeric/categorical fields are not directly passed
to the trained classifier.
