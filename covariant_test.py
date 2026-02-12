import pandas as pd
from scipy.stats import ks_2samp

train = pd.read_csv("nsl-kdd/KDDTrain+.csv")
test  = pd.read_csv("nsl-kdd/KDDTest+.csv")

features = ["src_bytes", "dst_bytes", "count", "srv_count"]

for f in features:
    stat, p = ks_2samp(train[f], test[f])
    print(f, "p-value =", p)
