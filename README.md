# Flight Price Prediction
Travel agencies need to track and predict fluctuations in flight prices to provide competitive tour packages. In this project we try to build a model to estimate flight prices for different routes with layover as well.

## Table of Contents
- [Architecture](#architecture)
- [Languages & Tools](#languages--tools)
- [Directory Structure](#directory-structure)
- [Data](#data)
- [Output](#output)
- [Application Link](#application-link)
  
## Architecture
**1. Data Ingestion:**  The data is loaded from local environment used for further processing..

**2. Data Preprocessing:** Imputing Missing Values, Removing the Duplicate rows, Dropping the unnecessary columns. Created muliple Transfomers to be used in FeatureUnion to generate new set of features and dropping the original parent columns. Encoding of Categorical features is done by creating a column transformer for the one hot encoding. Both these FeatureUnion and Encoder are fitted into the pipeline. The transformed data is used for model training. These transformers are stored as pickle files so that during prediction on webapp the user input features are transformed using these steps.

**3. Model Training:** Various models are trained on the preprocessed data such as Linear, Ridge, Lasso, Decision Tree, XGBoost and few others.

**4. Model Evaluation:** Among the different models, KNN, Random Forest and XGBoost were the top performers.

**5. Deployment:** In the deployment phase, the selected model is deployed using Streamlit.

## Languages & Tools
<div align="">
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="80" height="60"/>
  </a>
  <a href="https://code.visualstudio.com" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/vscode/vscode-original.svg" alt="vscode" width="50" height="60"/>
  </a>
  <a href="https://streamlit.io" target="_blank" rel="noreferrer">
    <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="streamlit" width="100" height="60"/>
  </a>
</div>

## Directory Structure

```
C:.
├───bin
│       encoder.pkl
│       features.pkl
│       model.pkl
│       pipe.pkl
│
├───data
│       dataset.csv
│       testset.csv
│       trainset.csv
│       Data_Train.xlsx
│
├───images
│       flight_back.png
│       flight_rotate.gif
│
└───src
        app.py
        check.ipynb
        Modelling.py
        Preprocessing.py
```
## Data
[Dataset](https://github.com/mansoorali1/Flight_Price_Prediction/blob/main/data/Data_Train.xlsx)
## Output
### WebApp UI
![Flight_Price_Prediction_UI](https://github.com/mansoorali1/Flight_Price_Prediction/blob/main/images/flight_UI.png)

### Prediction
![Flight_Price_Prediction](https://github.com/mansoorali1/Flight_Price_Prediction/blob/main/images/flight_output.png)

## Application Link
Here is the App: https://flightpricepredictionapp.streamlit.app/
