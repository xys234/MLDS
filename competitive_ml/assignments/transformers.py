pipeline = Pipeline([
    ('features', FeatureUnion([
        ('continuous', Pipeline([
            ('extract', ColumnExtractor(CONTINUOUS_FIELDS)),
            ('scale', Normalizer())
        ])),
        ('factors', Pipeline([
            ('extract', ColumnExtractor(FACTOR_FIELDS)),
            ('one_hot', OneHotEncoder(n_values=5)),
            ('to_dense', DenseTransformer())
        ])),
        ('weekday', Pipeline([
            ('extract', DayOfWeekTransformer()),
            ('one_hot', OneHotEncoder()),
            ('to_dense', DenseTransformer())
        ])),
        ('hour_of_day', HourOfDayTransformer()),
        ('month', Pipeline([
            ('extract', ColumnExtractor(['datetime'])),
            ('to_month', DateTransformer()),
            ('one_hot', OneHotEncoder()),
            ('to_dense', DenseTransformer())
        ])),
        ('growth', Pipeline([
            ('datetime', ColumnExtractor(['datetime'])),
            ('to_numeric', MatrixConversion(int)),
            ('regression', ModelTransformer(LinearRegression()))
        ]))
    ])),
    ('estimators', FeatureUnion([
        ('knn', ModelTransformer(KNeighborsRegressor(n_neighbors=5))),
        ('gbr', ModelTransformer(GradientBoostingRegressor())),
        ('dtr', ModelTransformer(DecisionTreeRegressor())),
        ('etr', ModelTransformer(ExtraTreesRegressor())),
        ('rfr', ModelTransformer(RandomForestRegressor())),
        ('par', ModelTransformer(PassiveAggressiveRegressor())),
        ('en', ModelTransformer(ElasticNet())),
        ('cluster', ModelTransformer(KMeans(n_clusters=2)))
    ])),
    ('estimator', KNeighborsRegressor())
])



from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureMultiplier(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, X, *_):
        return X * self.factor

    def fit(self, *_):
        return self

fm = FeatureMultiplier(2)

test = np.diag((1,2,3,4))
print test

fm.transform(test)



class ColumnExtractor(TransformerMixin):
	def __init__(self, cols):
		self.cols = cols
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		assert isinstance(X, pd.DataFrame)
		Xcols = X[self.cols]
		return Xcols

		
		
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
transformer = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        # Part 1
        ('boolean', Pipeline([
            ('selector', TypeSelector('bool')),
        ])),  # booleans close
        
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
        ])),  # numericals close
        
        # Part 2
        ('categoricals', Pipeline([
            ('selector', TypeSelector('category')),
            ('labeler', StringIndexer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]))  # categoricals close
    ])),  # features close
])  # pipeline close


class group_by_featurizer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col, feature):
        self.group_col = group_col
        self.value_col = value_col
        self.feature = feature
        self.gb = None
    def fit(self, X):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(group_col, list)
        self.gb = X.groupby(group_col, as_index=False).agg({value_col:feature})
        return self
    def transform(self, X):
        self.fit(X)
        return pd.merge(X, self.gb, on=group_col)
