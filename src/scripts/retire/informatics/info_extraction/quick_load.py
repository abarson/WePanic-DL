import pandas as pd
from we_panic_utils.nn.data_load import sorted_stratified_kfold

df = pd.read_csv('wepanic_collated_catalogue.csv')

a = sorted_stratified_kfold(df, k=5)

