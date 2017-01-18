#!//Users/snappergoldenhelix/anaconda/envs/biorepo/bin/python

"""Functions used in this pipeline."""

import pandas as pd
import numpy as np

# files
# file = "./data/raw/SnapperGIBioreposito_DATA_2017-01-10_1643.csv"

# generate dataframe
df = pd.read_csv("./data/raw/SnapperGIBioreposito_DATA_2017-01-10_1643.csv")
print("dataframe read.")
print(df.shape)

# generate compound key for visit entry
df['vis_id'] = df[['biorepidnumber', 'sample_date']].astype(str).apply(lambda x: ''.join(x), axis=1)
#print(df['vis_id'])
print("visit key generated.")


# generate family and member id for subject entity
fm = df.iloc[:,4].astype(str).str.split('.')
df['family_id'] = fm.str.get(0)
df['member_id'] = fm.str.get(1)
print("family and member id generated")
print(df.shape)
#df.to_csv("./data/processed/edit.csv")

# create subject dataframe
sub = df.iloc[:, np.r_[4, 8, 9, 11:34 , 35:51, 214, 215]]
sub = sub.drop_duplicates()
sub.to_csv("./data/processed/subject2.csv", index = False)
print("subject")
print(sub.shape)

# create sample dataframe
sam = df.iloc[:, np.r_[0 : 3, 4, 5 : 8, 10, 55 : 126, 213]]
sam = sam.drop_duplicates()
sam.to_csv("./data/processed/sample2.csv", index = False)
print("sample")
print(sam.shape)

# create visit dataframe
vis = df.iloc[:, np.r_[213, 4, 7, 1, 53, 54, 126 : 213]]
vis = vis.drop_duplicates()
vis.to_csv("./data/processed/visit2.csv", index = False)
print("visit")
print(vis.shape)
