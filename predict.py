import joblib
import pandas as pd
from utils import n_grams

df_test = pd.read_csv('test_df.csv')

model_pipeline = joblib.load("model_pipeline.pkl")

list_domains = list(df_test.domain)
is_dga = model_pipeline.predict(list_domains)
df_test['is_dga'] = is_dga
df_test.to_csv('prediction.csv', index=False)