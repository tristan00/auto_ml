import pandas as pd
import numpy as np
import random
from sklearn import preprocessing, decomposition, model_selection, feature_extraction, cluster
import re

pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel_neg = -999999999
numerical_sentinel_pos = -999999999

string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'


def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def get_possible_transformations():
    transformations_dict = dict()
    transformations_dict['sum'] = {'min_columns': 2,
                                   'max_columns': 2,
                                   'input_type': ['numeric'],
                                   'output_columns': 1,
                                   'parameters': list()}
    transformations_dict['product'] = {'min_columns': 2,
                                       'max_columns': 2,
                                       'input_type': ['numeric'],
                                       'output_columns': 1,
                                       'parameters': list()}
    transformations_dict['subtraction'] = {'min_columns': 2,
                                           'max_columns': 2,
                                           'input_type': ['numeric'],
                                           'output_columns': 1,
                                           'parameters': list()}
    transformations_dict['ratio'] = {'min_columns': 2,
                                     'max_columns': 2,
                                     'input_type': ['numeric'],
                                     'output_columns': 1,
                                     'parameters': [{'name': 'inf_fill_value',
                                                     'selection_type': 'choice',
                                                     'options': [0, numerical_sentinel_neg, numerical_sentinel_pos]}]}
    transformations_dict['min'] = {'min_columns': 2,
                                   'max_columns': 2,
                                   'input_type': ['numeric'],
                                   'output_columns': 1,
                                   'parameters': []}
    transformations_dict['max'] = {'min_columns': 2,
                                   'max_columns': 2,
                                   'input_type': ['numeric'],
                                   'output_columns': 1,
                                   'parameters': []}
    transformations_dict['var'] = {'min_columns': 2,
                                   'max_columns': 2,
                                   'input_type': ['numeric'],
                                   'output_columns': 1,
                                   'parameters': []}
    transformations_dict['identity'] = {'min_columns': 2,
                                        'max_columns': 2,
                                        'input_type': ['numeric'],
                                        'output_columns': 1,
                                        'parameters': []}
    transformations_dict['pca'] = {'min_columns': 2,
                                   'max_columns': 8,
                                   'input_type': ['numeric'],
                                   'output_columns': 'input',
                                   'parameters': []}
    transformations_dict['quantile_scaler'] = {'min_columns': 1,
                                               'max_columns': 1,
                                               'input_type': ['numeric'],
                                               'output_columns': 1,
                                               'parameters': []}
    transformations_dict['standard_scaler'] = {'min_columns': 1,
                                               'max_columns': 1,
                                               'input_type': ['numeric'],
                                               'output_columns': 1,
                                               'parameters': []}
    transformations_dict['standard_scaler'] = {'min_columns': 1,
                                               'max_columns': 1,
                                               'input_type': ['numeric'],
                                               'output_columns': 1,
                                               'parameters': []}
    transformations_dict['power_transform'] = {'min_columns': 1,
                                               'max_columns': 1,
                                               'input_type': ['numeric'],
                                               'output_columns': 1,
                                               'parameters': []}
    transformations_dict['dictionary_encode'] = {'min_columns': 1,
                                                 'max_columns': 1,
                                                 'input_type': ['string', 'numeric'],
                                                 'output_columns': 1,
                                                 'parameters': []}
    transformations_dict['one_hot_encoding'] = {'min_columns': 1,
                                                'max_columns': 1,
                                                'input_type': ['string', 'numeric'],
                                                'output_columns': 'param',
                                                'parameters': [{'name': 'output_cols',
                                                                'selection_type': 'int_range',
                                                                'options': [2, 50],
                                                                'link': 'output_columns'}]}
    cluster_params = [{'name': 'clustering_algorithm',
                       'selection_type': 'choice',
                       'options': ['KMeans',
                                   'MiniBatchKMeans',
                                   'Birch'],
                       },
                      {'name': 'eps',
                       'param_requirement': {'clustering_algorithm': ['DBSCAN']},
                       'selection_type': 'float_range',
                       'options': [0.0, 1.0],
                       },
                      {'name': 'affinity',
                       'param_requirement': {'clustering_algorithm': ['AgglomerativeClustering',
                                                                      'SpectralClustering']},
                       'selection_type': 'requirement_dict_choice',
                       'options': {'AgglomerativeClustering': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
                                   'SpectralClustering': ['nearest_neighbors', 'rbf']},
                       },
                      {'name': 'n_clusters',
                       'selection_type': 'int_range',
                       'options': [2, 16],
                       'link': 'output_columns'}]
    transformations_dict['clustering'] = {'min_columns': 2,
                                          'max_columns': 8,
                                          'input_type': 'string',
                                          'output_columns': 'param',
                                          'parameters': cluster_params}
    bow_params = [{'name':'max_features',
                   'selection_type': 'int_range',
                   'options': [2, 100],
                   'link': 'output_columns'},
                  {'name': 'norm',
                   'selection_type': 'choice',
                   'options': ['l1',
                               'l2',
                               None],
                   },
                  {'name': 'analyzer',
                   'selection_type': 'choice',
                   'options': ['word',
                               'char',
                               'char_wb'],
                   },
                  {'name': 'ngram_range_min_n',
                   'selection_type': 'choice',
                   'options': [1, 2],
                   },
                  {'name': 'ngram_range_max_n',
                   'selection_type': 'choice',
                   'options': [2, 3, 4],
                   },
                  {'name':'max_df',
                   'selection_type': 'float_range',
                   'options': [0.0, 1.0]},
                  {'name':'binary',
                   'selection_type':'choice',
                   'options':[True, False]},
                  {'name':'use_idf',
                   'selection_type':'choice',
                   'options':[True, False]}]
    transformations_dict['TextBOWVectorizer'] = {'min_columns': 1,
                                                'max_columns': 1,
                                                'input_type': ['string'],
                                                'output_columns': 'param',
                                                'parameters': bow_params}
    return transformations_dict


def get_random_transformation(dataset_description):
    transformation_dict = get_possible_transformations()
    tranformation = random.choice(list(transformation_dict.keys()))

    # Get input columns of correct type
    #




class TextBOWVectorizer:
    def __init__(self, params):
        self.params = params
        self.params['ngram_range'] = self.params['ngram_range_min_n'], self.params['ngram_range_max_n']
        del self.params['ngram_range_min_n'], self.params['ngram_range_max_n']
        self.model = feature_extraction.text.TfidfVectorizer(**self.params)

    def fit(self, x):
        self.model.fit(x)

    def transform(self, x):
        pred_value = self.model.transform(x).toarray()
        if pred_value.shape[1] < self.params['max_features']:
            pred_value = np.hstack([pred_value, np.zeros((pred_value.shape[0], self.params['max_features'] - pred_value.shape[1]))])
        return pred_value


# TODO: for non transductive algorithms such as , add options such as classifier predict.
class Cluster:
    def __init__(self, params):

        self.algorithm = params['clustering_algorithm']
        self.params = params
        del self.params['clustering_algorithm']

        if self.algorithm == 'KMeans':
            self.model = cluster.KMeans(**self.params)
        if self.algorithm == 'MiniBatchKMeans':
            self.model = cluster.MiniBatchKMeans(**self.params)
        if self.algorithm == 'Birch':
            self.model = cluster.Birch(**self.params)

    def fit(self, x):
        self.model.fit(x)

    def transform(self, x):
        return self.model.predict(x)


class DictionaryEncoder():

    def fit(self, x):
        vc = dict()
        for i in x:
            vc.setdefault(i, 0)
            vc[i] += 1
        vc_list = sorted(list(vc.items()), key=lambda i: i[1], reverse=True)

        self.other_filler = len(vc_list)
        self.result_dict = dict()
        for n, i in enumerate(vc_list):
            self.result_dict[i[0]] = n

    def transform(self, x):
        output_list = list()
        for i in x:
            output_list.append(self.result_dict.get(i, self.other_filler))
        return output_list

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class OHE():
    def __init__(self, num_of_output_columns=10):
        self.num_of_output_columns = num_of_output_columns

    def fit(self, x):
        vc = x.fillna(string_sentinel).value_counts(normalize=True)
        vc_list = vc.index.tolist()
        allowed_values = vc_list[:self.num_of_output_columns - 1]

        self.result_dict = dict()
        for i in range(self.num_of_output_columns - 1):
            next_array = [0 for _ in range(self.num_of_output_columns)]
            next_array[i] = 1
            try:
                self.result_dict[allowed_values[i]] = np.array(next_array)
            except IndexError:
                self.result_dict['empty_value_{}'.format(i)] = np.array(next_array)

        self.other_filler = [0 for _ in range(self.num_of_output_columns)]
        self.other_filler[-1] = 1
        self.other_filler = np.array(self.other_filler)

    def transform(self, x):
        output_list = list()
        for i in x:
            output_list.append(self.result_dict.get(i, self.other_filler))
        data = np.vstack(output_list)
        return data

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class Transformation:

    def __init__(self, transformation_type, transformation_parameters, input_columns, output_columns):
        self.transformation_type = transformation_type
        self.transformation_parameters = transformation_parameters
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, df):

        if self.transformation_type in ['quantile_scaler', 'standard_scaler', 'power_transform']:
            if self.transformation_type == 'quantile_scaler':
                scaler = preprocessing.QuantileTransformer()
            elif self.transformation_type == 'standard_scaler':
                scaler = preprocessing.StandardScaler()
            elif self.transformation_type == 'power_transform':
                scaler = preprocessing.PowerTransformer()
            else:
                raise NotImplementedError
            scaler.fit(df[self.input_columns[0]].values.reshape(-1, 1))

        if self.transformation_type in ['quantile_scaler', 'standard_scaler', 'power_transform']:
            if self.transformation_type == 'quantile_scaler':
                scaler = preprocessing.QuantileTransformer()
            elif self.transformation_type == 'standard_scaler':
                scaler = preprocessing.StandardScaler()
            elif self.transformation_type == 'power_transform':
                scaler = preprocessing.PowerTransformer()
            else:
                raise NotImplementedError
            scaler.fit(df[self.input_columns[0]].values.reshape(-1, 1))

    def transform(self):
        pass
