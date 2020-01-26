from sklearn import ensemble, linear_model
import copy
from common import pick_parameter
import statsmodels as sm
import lightgbm as lgb

def get_possible_models(problem_type, num_of_classes = None):
    transformations_dict = dict()

    if problem_type == 'regression':


        transformations_dict['ElasticNet'] = [{'name': 'alpha',
                       'selection_type': 'float_range',
                       'options': [0.0, 2.0]},
                                 {'name': 'alpha',
                                  'selection_type': 'l1_ratio',
                                  'options': [0.0, 1.0]}]
        transformations_dict['lightgbm'] = [{'name': 'objective',
                       'selection_type': 'choice',
                       'options': ['l1', 'l2', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'tweedie']},
                                            {'name': 'bagging_fraction',
                                             'selection_type': 'float_range',
                                             'options': [.1, 1.0]},
                                            {'name': 'bagging_fraction',
                                             'selection_type': 'float_range',
                                             'options': [.1, 1.0]},
                                            {'name': 'feature_fraction',
                                             'selection_type': 'float_range',
                                             'options': [.1, 1.0]},
                                            {'name': 'feature_fraction_bynode',
                                             'selection_type': 'float_range',
                                             'options': [.1, 1.0]}
        ]

    return transformations_dict


class Model():


    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.parameters = model_params

        if self.model_type == 'ElasticNet':
            self.model = linear_model.ElasticNet(**self.parameters)


    def fit(self, x, y):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            self.model.fit(x, y)
        else:
            raise NotImplementedError

    def predict(self, x):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            return self.model.predict(x)
        else:
            raise NotImplementedError
