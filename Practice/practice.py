# Source - https://stackoverflow.com/q/52224813
# Posted by Eun Jee Lee, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-05, License - CC BY-SA 4.0

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
print(regr.predict([[0, 0, 0, 0]]))

