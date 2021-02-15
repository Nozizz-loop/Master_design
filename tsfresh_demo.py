import pandas as pd
import sys
import tsfresh
df = pd.read_csv(r'D:\Goes\Goes_xray\2011\csv\20111231_Gp_xr_1m.csv',
                 names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'], header=1, index_col=None)
df.fillna(0, inplace=True)
print(df.head())

#
# def main():
from tsfresh.feature_extraction import extract_features
features = extract_features(df,  column_id='Short', column_sort='Sec')
print(features.loc[:10, ['']])

# if __name__ == '__main__':
# sys.exit(main())