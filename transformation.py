import pandas as pd
import numpy as np
import random
from sklearn import preprocessing, decomposition, model_selection, feature_extraction, cluster
import re
import json
import uuid
from common import pick_parameter, string_sentinel


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
    transformations_dict['difference'] = {'min_columns': 2,
                                          'max_columns': 2,
                                          'input_type': ['numeric'],
                                          'output_columns': 1,
                                          'parameters': list()}
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

    transformations_dict['identity'] = {'min_columns': 2,
                                        'max_columns': 2,
                                        'input_type': ['numeric'],
                                        'output_columns': 1,
                                        'parameters': []}
    transformations_dict['pca'] = {'min_columns': 2,
                                   'max_columns': 4,
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
                                          'max_columns': 4,
                                          'input_type': 'string',
                                          'output_columns': 'param',
                                          'parameters': cluster_params}
    bow_params = [{'name': 'max_features',
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
                  {'name': 'max_df',
                   'selection_type': 'float_range',
                   'options': [0.0, 1.0]},
                  {'name': 'binary',
                   'selection_type': 'choice',
                   'options': [True, False]},
                  {'name': 'use_idf',
                   'selection_type': 'choice',
                   'options': [True, False]}]
    transformations_dict['TextBOWVectorizer'] = {'min_columns': 1,
                                                 'max_columns': 1,
                                                 'input_type': ['string'],
                                                 'output_columns': 'param',
                                                 'parameters': bow_params}
    del transformations_dict['TextBOWVectorizer']
    del transformations_dict['clustering']
    del transformations_dict['one_hot_encoding']
    return transformations_dict


def get_random_transformation(dataset_description, name):
    transformation_dict = get_possible_transformations()
    print()
    transformation_type = random.choice(list(transformation_dict.keys()))

    if transformation_dict[transformation_type]['min_columns'] == transformation_dict[transformation_type][
        'max_columns']:
        number_of_columns = transformation_dict[transformation_type]['min_columns']
    else:
        number_of_columns = random.randint(transformation_dict[transformation_type]['min_columns'],
                                           transformation_dict[transformation_type]['max_columns'])

    valid_columns = [k for k, v in dataset_description['columns'].items() if
                     v['target'] == 0 and v['type'] in transformation_dict[transformation_type]['input_type']]

    if len(valid_columns) < number_of_columns:
        return None

    input_columns = random.sample(valid_columns, number_of_columns)
    num_of_output_columns = None
    transformation_parameters = dict()

    if transformation_dict[transformation_type]['output_columns'] == 'param':
        for i in transformation_dict[transformation_type]['parameters']:
            if i.get('link') == 'output_columns':
                transformation_parameters[i['name']] = pick_parameter(i['options'], i['selection_type'])
                num_of_output_columns = transformation_parameters[i['name']]
                break
    elif transformation_dict[transformation_type]['output_columns'] == 'input':
        num_of_output_columns = number_of_columns
    elif isinstance(transformation_dict[transformation_type]['output_columns'], int):
        num_of_output_columns = transformation_dict[transformation_type]['output_columns']
    else:
        raise NotImplementedError

    output_columns = ['{0}_{1}'.format(name, i) for i in range(num_of_output_columns)]

    for i in transformation_dict[transformation_type]['parameters']:
        if i in transformation_parameters.keys():
            continue
        transformation_parameters[i['name']] = pick_parameter(i['options'], i['selection_type'])

    return Transformation(name, transformation_type, transformation_parameters, input_columns, output_columns)


def get_n_random_transformations(dataset_description, n):
    transformations = list()
    while len(transformations) < n:
        name = str(uuid.uuid4().hex)
        temp_transformation = get_random_transformation(dataset_description, name)
        if temp_transformation:
            transformations.append(temp_transformation)
    return transformations


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
            pred_value = np.hstack(
                [pred_value, np.zeros((pred_value.shape[0], self.params['max_features'] - pred_value.shape[1]))])
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


class DictionaryEncoder:

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

    def __init__(self, name,
                 transformation_type,
                 transformation_parameters,
                 input_columns,
                 output_columns):
        self.name = name
        self.transformation_type = transformation_type
        self.transformation_parameters = transformation_parameters
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.model = None
        print('Transformation constructor', transformation_type, transformation_parameters, input_columns,
              output_columns)

    def fit(self, df):
        print('Transformation fit', df.columns.tolist())
        if self.transformation_type in ['quantile_scaler', 'standard_scaler', 'power_transform']:
            if self.transformation_type == 'quantile_scaler':
                self.model = preprocessing.QuantileTransformer()
            elif self.transformation_type == 'standard_scaler':
                self.model = preprocessing.StandardScaler()
            elif self.transformation_type == 'power_transform':
                self.model = preprocessing.PowerTransformer()
            else:
                raise NotImplementedError
            self.model.fit(df[self.input_columns[0]].values.reshape(-1, 1))

        if self.transformation_type in ['quantile_scaler', 'standard_scaler', 'power_transform']:
            if self.transformation_type == 'quantile_scaler':
                self.model = preprocessing.QuantileTransformer()
            elif self.transformation_type == 'standard_scaler':
                self.model = preprocessing.StandardScaler()
            elif self.transformation_type == 'power_transform':
                self.model = preprocessing.PowerTransformer()
            else:
                raise NotImplementedError
            self.model.fit(df[self.input_columns[0]].values.reshape(-1, 1))

        if self.transformation_type in ['pca']:
            self.model = decomposition.PCA()
            self.model.fit(df[self.input_columns])

        if self.transformation_type in ['dictionary_encode']:
            self.model = DictionaryEncoder()
            self.model.fit(df[self.input_columns[0]])

    def transform(self, df):
        df = df.copy()
        print('Transformation transform', df.columns.tolist())
        if self.transformation_type in ['quantile_scaler', 'standard_scaler', 'power_transform']:
            df[self.output_columns[0]] = self.model.transform(df[self.input_columns[0]].values.reshape(-1, 1))

        elif self.transformation_type in ['sum', 'product', 'difference', 'min', 'max', 'identity']:
            if self.transformation_type == 'sum':
                df[self.output_columns[0]] = df[self.input_columns].sum(axis=1).values
            elif self.transformation_type == 'product':
                df[self.output_columns[0]] = df[self.input_columns].product(axis=1).values
            elif self.transformation_type == 'difference':
                df[self.output_columns[0]] = df[self.input_columns[0]] - df[self.input_columns[1]]
            elif self.transformation_type == 'min':
                df[self.output_columns[0]] = df[self.input_columns].min(axis=1).values
            elif self.transformation_type == 'max':
                df[self.output_columns[0]] = df[self.input_columns].max(axis=1).values
            elif self.transformation_type == 'identity':
                df[self.output_columns[0]] = df[self.input_columns[0]].copy()
            else:
                raise NotImplementedError

        elif self.transformation_type in ['pca']:
            temp_data = self.model.transform(df[self.input_columns])
            temp_df = pd.DataFrame(data=temp_data,
                                   index=df.index,
                                   columns=self.output_columns)
            df = pd.concat([df, temp_df], axis=1)
        elif self.transformation_type in ['dictionary_encode']:
            df[self.output_columns[0]] = self.model.transform(df[self.input_columns[0]])
        else:
            raise NotImplementedError
        return df


if __name__ == '__main__':
    # with open('sample_dataset_description.json', 'r') as f:
    #     dataset_description = json.load( f)
    print(get_possible_transformations())
