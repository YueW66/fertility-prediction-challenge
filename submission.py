"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """


    ## This script contains a bare minimum working example
    # Create new variable with age
    df["age_respondent"] = df["cf17j004"]
    df["birth_year"] = df["birthyear_bg"]
    df["age"] = df["age_bg"]
    df["primary_occupation"] = df["belbezig_2015"]
    df["gross_household_income_2010"] = df["brutohh_f_2010"]
    df["gross_household_income_2017"] = df["brutohh_f_2017"]
    df["gross_household_income_2019"] = df["brutohh_f_2019"]
    df["gross_household_income_2020"] = df["brutohh_f_2020"]
    df["nett_household_income_2009"] = df["nettohh_f_2009"]
    df["nett_household_income_2020"] = df["nettohh_f_2020"]

    # Selecting variables for modelling
    keepcols = [
        "nomem_encr",  # ID variable required for predictions,
        "age_respondent",         # newly created variable
        "birth_year", 
        "age", 
        "primary_occupation", 
        "gross_household_income_2010", 
        "gross_household_income_2017", 
        "gross_household_income_2019", 
        "gross_household_income_2020", 
        "nett_household_income_2009", 
        "nett_household_income_2020"
    ] 

    # Keeping data with variables selected
    df = df[keepcols]

    # Imputing missing values with the median
    df["age_respondent"] = df["age_respondent"].fillna(df["age_respondent"].median())
    df["birth_year"] = df["birth_year"].fillna(df["birth_year"].median())
    df["age"] = df["age"].fillna(df["age"].median())
    df["primary_occupation"] = df["primary_occupation"].fillna(df["primary_occupation"].median())
    df["gross_household_income_2010"] = df["gross_household_income_2010"].fillna(df["gross_household_income_2010"].median())
    df["gross_household_income_2017"] = df["gross_household_income_2017"].fillna(df["gross_household_income_2017"].median())
    df["gross_household_income_2019"] = df["gross_household_income_2019"].fillna(df["gross_household_income_2019"].median())
    df["gross_household_income_2020"] = df["gross_household_income_2020"].fillna(df["gross_household_income_2020"].median())
    df["nett_household_income_2009"] = df["nett_household_income_2009"].fillna(df["nett_household_income_2009"].median())
    df["nett_household_income_2020"] = df["nett_household_income_2020"].fillna(df["nett_household_income_2020"].median())

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict