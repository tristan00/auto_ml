from sklearn import ensemble, linear_model
import copy
from common import pick_parameter
import statsmodels as sm
import lightgbm as lgb
import uuid
import random


def get_possible_models(problem_type, num_of_classes = None):
    model_param_dict = dict()

    if problem_type == 'regression':


        # model_param_dict['ElasticNet'] = [{'name': 'alpha',
        #                'selection_type': 'float_range',
        #                'options': [0.0, 2.0]},
        #                          {'name': 'l1_ratio',
        #                           'selection_type': 'float_range',
        #                           'options': [0.0, 1.0]}]
        # model_param_dict['RandomForestRegressor'] = [{'name': 'criterion',
        #                'selection_type': 'choice',
        #                'options': ['mse', 'mae']},
        #                          {'name': 'max_depth',
        #                           'selection_type': 'int_range',
        #                           'options': [2, 12]}]

        model_param_dict['LGBMRegressor'] = [{'name': 'objective',
                       'selection_type': 'choice',
                       'options': ['l1', 'l2', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'tweedie']},
                                            {'name': 'boosting_type',
                                             'selection_type': 'choice',
                                             'options': ['gbdt', 'dart', 'goss']},
                                             {'name': 'num_leaves',
                                              'selection_type': 'int_range',
                                              'options': [2, 128]},
                                             {'name': 'learning_rate',
                                              'selection_type': 'float_range',
                                              'options': [.01, .2]},
                                             {'name': 'n_estimators',
                                              'selection_type': 'int_range',
                                              'options': [10, 400]},

        ]
        # del model_param_dict['ElasticNet']
    elif problem_type == 'classification':
        model_param_dict['LGBMRegressor'] = [{'name': 'objective',
                                              'selection_type': 'choice',
                                              'options': ['multiclass']},
                                             {'name': 'boosting_type',
                                              'selection_type': 'choice',
                                              'options': ['gbdt', 'dart', 'goss']},
                                             {'name': 'num_leaves',
                                              'selection_type': 'int_range',
                                              'options': [2, 128]},
                                             {'name': 'learning_rate',
                                              'selection_type': 'float_range',
                                              'options': [.01, .2]},
                                             {'name': 'n_estimators',
                                              'selection_type': 'int_range',
                                              'options': [10, 100]},

                                             ]

    return model_param_dict


def get_random_model(problem_type):
    possible_model_param_dict = get_possible_models(problem_type)
    model_type = random.choice(list(possible_model_param_dict.keys()))
    model_params = dict()
    for i in possible_model_param_dict[model_type]:
        if i['name'] in model_params.keys():
            continue
        model_params[i['name']] = pick_parameter(i['options'], i['selection_type'])
    m = Model(model_type, model_params)
    return m


def get_n_random_models(problem_type, n):
    return [get_random_model(problem_type) for _ in range(n)]


class Model:

    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model_id = str(uuid.uuid4().hex)


        if self.model_type == 'ElasticNet':
            self.model = linear_model.ElasticNet(**self.model_params)
        if self.model_type == 'RandomForestRegressor':
            self.model = ensemble.RandomForestRegressor(**self.model_params)
        if self.model_type == 'LGBMRegressor':
            self.model = lgb.LGBMRegressor(**self.model_params)

    def fit(self, x, y):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            self.model.fit(x, y)
        elif self.model_type in ['LGBMRegressor']:
            self.model.fit(x, y)
        else:
            raise NotImplementedError

    def predict(self, x):
        if self.model_type in ['RandomForestRegressor', 'ElasticNet', 'SGDRegressor', 'GradientBoostingRegressor']:
            return self.model.predict(x)
        elif self.model_type in ['LGBMRegressor']:
            return self.model.predict(x)
        else:
            raise NotImplementedError

    def get_model_description(self):
        temp_dict =  {'model_id':self.model_id,
                'model_type':self.model_type}

        model_param_dict = {'model_params_{}'.format(k):v for k, v in self.model_params.items()}
        temp_dict.update(model_param_dict)
        return temp_dict




