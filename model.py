from sklearn import ensemble, linear_model
import copy


class Model():
    model_param_dict = {'RandomForestRegressor': {'n_estimators': [10, 100],
                                                  'criterion': ['mae', 'mse'],
                                                  'max_depth': [2, 8],
                                                  'min_samples_split': [2],
                                                  'max_features': ['sqrt', 'log2', None],
                                                  },
                        'ElasticNet': {'l1_ratio': (0.0, 1.0),
                                       'fit_intercept': [True, False],
                                       'selection': ['random', 'cyclic']
                                       },
                        'GradientBoostingRegressor': {'loss': ['ls', 'lad', 'huber', 'quantile'],
                                                      'learning_rate': (.05, .25),
                                                      'n_estimators': (10, 500),
                                                      'subsample': [1.0, .8, .5],
                                                      'criterion': ['friedman_mse', 'mse', 'mae'],
                                                      'max_depth': (2, 8),
                                                      'max_features': ['sqrt', 'log2', None],
                                                      'alpha': (0.0, 1.0),
                                                      'n_iter_no_change': [None, 10],
                                                      'min_samples_split': (2, 20)},
                        'SGDRegressor': {
                            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                            'penalty': ['l1', 'l2', 'elasticnet'],
                            'alpha': [.001, .0001, .00001],
                            'l1_ratio': (0.0, 1.0),
                            'fit_intercept': [True, False],
                            'epsilon': (0.0, 0.5),
                            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                            'eta0': [.001, .01, .1],
                            'early_stopping': [True, False]}
                        }

    def __init__(self, model_params):
        self.model_type = model_params['model_type']
        self.parameters = copy.deepcopy(model_params)
        del self.parameters['model_type']
        if self.model_type == 'RandomForestRegressor':
            self.model = ensemble.RandomForestRegressor(**self.parameters)
        if self.model_type == 'GradientBoostingRegressor':
            self.model = ensemble.GradientBoostingRegressor(**self.parameters)
        if self.model_type == 'ElasticNet':
            self.model = linear_model.ElasticNet(**self.parameters)
        if self.model_type == 'SGDRegressor':
            self.model = linear_model.SGDRegressor(**self.parameters)

    def fit(self, x, y):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            self.model.fit(x, y)

    def predict(self, x):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            return self.model.predict(x)
