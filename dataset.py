import pandas as pd
import numpy as np
import random
from sklearn import preprocessing, decomposition, model_selection, feature_extraction
import re

pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel = -999999999
string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'

def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def get_dataset_description(df, target, column_subset=None):
    dataset_profile = dict()
    dataset_profile['size'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
    }
    dataset_profile['columns'] = dict()
    for i in df.columns:
        if column_subset and i != 'target' and i not in column_subset:
            continue

        dataset_profile['columns'][i] = dict()
        if i == target:
            dataset_profile['columns'][i]['target'] = 1
        else:
            dataset_profile['columns'][i]['target'] = 0
        if df[i].dtype in pd_numerics:
            dataset_profile['columns'][i]['type'] = 'numeric'
            dataset_profile['columns'][i]['nan_count'] = int(df[i].isna().sum())
            dataset_profile['columns'][i]['mean'] = float(df[i].mean())
            dataset_profile['columns'][i]['median'] = float(df[i].median())
            dataset_profile['columns'][i]['skew'] = float(df[i].skew())
            dataset_profile['columns'][i]['nunique'] = int(df[i].nunique())
        else:
            dataset_profile['columns'][i]['type'] = 'string'
            dataset_profile['columns'][i]['nan_count'] = int(df[i].isna().sum())
            dataset_profile['columns'][i]['nunique'] = int(df[i].nunique())

        # print({j: type(j) for j in dataset_profile['columns'][i]})
    return dataset_profile


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
    def __init__(self, max_columns = 100):
        self.max_columns = max_columns

    def fit(self, x):
        vc = dict()
        for i in x:
            vc.setdefault(i, 0)
            vc[i] += 1
        vc_list = sorted(list(vc.items()), key=lambda i: i[1], reverse=True)

        self.result_dict = dict()
        if self.max_columns >= len(vc_list):
            for n, i in enumerate(vc_list[:self.max_columns]):
                next_array = [0 for _ in range(len(vc_list) + 1)]
                next_array[n] = 1
                self.result_dict[i[0]] = np.array(next_array)
        else:
            for n, i in enumerate(vc_list[:self.max_columns]):
                next_array = [0 for _ in range(self.max_columns+1)]
                next_array[n] = 1
                self.result_dict[i[0]] = np.array(next_array)
        self.other_filler = [0 for _ in range(min(self.max_columns+1, len(vc_list)+1))]
        self.other_filler[-1] = 1

        vc_list = vc_list[:self.max_columns]
        vc_list.append((other_category_filler, 0))

    def transform(self, x, index = None):
        output_list = list()
        for i in x:
            output_list.append(self.result_dict.get(i, self.other_filler))
        data = np.vstack(output_list)
        return data

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class Transformation:
    transformation_dict = {
                                'sum':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':1},
                                'product':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':1},
                                'subtraction':{'min_columns':2, 'max_columns':2, 'input_type':'numeric', 'output_columns':1},
                                'ratio':{'min_columns':2, 'max_columns':2, 'input_type':'numeric', 'output_columns':1},
                                'min':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':1},
                                'max':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':1},
                                'var':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':1},
                                'pca':{'min_columns':2, 'max_columns':5, 'input_type':'numeric', 'output_columns':'same'},
                                'identity':{'min_columns':1, 'max_columns':1, 'input_type':'numeric', 'output_columns':1},
                                'quantile_scaler':{'min_columns':1, 'max_columns':1, 'input_type':'numeric', 'output_columns':1},
                                'standard_scaler':{'min_columns':1, 'max_columns':1, 'input_type':'numeric', 'output_columns':1},
                                'power_transform':{'min_columns':1, 'max_columns':1, 'input_type':'numeric', 'output_columns':1},
                                'dictionary_encode':{'min_columns':1, 'max_columns':1, 'input_type':'string', 'output_columns':1},
                                'one_hot_encoding':{'min_columns':1, 'max_columns':1,
                                                    'input_type':'string', 'output_columns':'param', 'min_output_columns':1, 'max_output_columns':100},
                                }

    # transformation_parameters_dict = {'one_hot_encoding': }

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


    def get_random_transformation(self, dataset_description, max_attempts = 10):

        for _ in range(max_attempts):
            transformation_type = random.choice(list(self.transformation_dict.keys()))
            data_type_required = self.transformation_dict[transformation_type]['input_type']
            valid_columns = [k for k, v in dataset_description['columns'].items() if v['type'] == data_type_required]

            num_of_columns = min(random.randint(self.transformation_dict[transformation_type]['min_columns'],
                                                self.transformation_dict[transformation_type]['max_columns']), len(valid_columns))
            if len(valid_columns) < num_of_columns:
                continue

            input_columns = random.sample(valid_columns)

            transformation_parameters = dict()
            if isinstance(self.transformation_dict[transformation_type]['output_columns'], int):
                num_output_columns = self.transformation_dict[transformation_type]['output_columns']
            elif self.transformation_dict[transformation_type]['output_columns'] == 'same':
                num_output_columns = num_of_columns
            elif self.transformation_dict[transformation_type]['output_columns'] == 'param':
                if transformation_type == 'one_hot_encoding':
                    num_output_columns = random.randint(1, min(self.transformation_dict[transformation_type]['max_output_columns'],
                                                               dataset_description['columns'][input_columns[0]]['nunique']))
                    transformation_parameters['max_columns'] = num_output_columns
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            output_columns = [clean_text('{0}_{1}_{2}'.format('_'.join(input_columns), transformation_type, n)) for n in range(num_output_columns)]
            return {'transformation_type':transformation_type,
                    'transformation_parameters':transformation_parameters,
                    'input_columns':input_columns,
                    'output_columns':output_columns}


class DataSet():

    def load_data(self, data_path,
                  data_format='csv',
                  validation_size=.20,
                  test_size=.20,
                  csv_sep=',',
                  target = None,
                  nrows = None):

        self.datasets = dict()
        if data_format == 'csv':
            self.target = target
            if nrows:
                self.raw_data = pd.read_csv(data_path, csv_sep, nrows = nrows)
            else:
                self.raw_data = pd.read_csv(data_path, csv_sep)

            if target:
                self.raw_data = self.raw_data[[target] + [i for i in self.raw_data.columns if i != target]]
            # self.raw_data = self.raw_data.set_index(list(range(self.raw_data.shape[0])))
            train_indices, test_indices = model_selection.train_test_split(list(self.raw_data.index), random_state=1, test_size=test_size)
            train_indices, validation_indices = model_selection.train_test_split(train_indices, random_state=1, test_size=test_size)

            # self.test_indices = random.sample(list(self.raw_data.index), int(test_size * self.raw_data.shape[0]))
            # self.train_indices = [i for i in self.raw_data.index if i not in self.test_indices]

            self.train_data = self.raw_data.loc[train_indices].copy()
            self.validation_data = self.raw_data.loc[validation_indices].copy()
            self.test_data = self.raw_data.loc[test_indices].copy()

            self.dataset_description = get_dataset_description(self.train_data, self.target)


        else:
            raise NotImplementedError
        return self.dataset_description


    def apply_nan_fill(self):
        for column_name, column_dict in self.dataset_description['columns'].items():
            if column_dict['target'] or not column_dict['nan_count']:
                continue

            if column_dict['type'] == 'numeric':
                self.train_data['{}_nan_median'.format(column_name)] = self.train_data[column_name].median()
                self.train_data['{}_nan_max'.format(column_name)] = self.train_data[column_name].max()
                self.train_data['{}_nan_min'.format(column_name)] = self.train_data[column_name].min()

                self.validation_data['{}_nan_median'] = self.train_data[column_name].median()
                self.validation_data['{}_nan_max'] = self.train_data[column_name].max()
                self.validation_data['{}_nan_min'] = self.train_data[column_name].min()

                self.test_data['{}_nan_median'] = self.train_data[column_name].median()
                self.test_data['{}_nan_max'] = self.train_data[column_name].max()
                self.test_data['{}_nan_min'] = self.train_data[column_name].min()

                self.train_data = self.train_data.drop(column_name, axis = 1)
                self.validation_data = self.validation_data.drop(column_name, axis = 1)
                self.test_data = self.test_data.drop(column_name, axis = 1)

            elif column_dict['type'] == 'string':
                train_most_common_value = self.train_data[column_name].values_counts().index[0]
                self.train_data['{}_nan_mode'.format(column_name)] = train_most_common_value
                self.validation_data['{}_nan_mode'.format(column_name)] = train_most_common_value
                self.test_data['{}_nan_mode'.format(column_name)] = train_most_common_value

                self.train_data['{}_nan_sentinel'.format(column_name)] = string_sentinel
                self.validation_data['{}_nan_sentinel'.format(column_name)] = string_sentinel
                self.test_data['{}_nan_sentinel'.format(column_name)] = string_sentinel

            else:
                raise NotImplementedError


    def apply_transformations(self, data_key,
                        transformation_dicts):
        df_train = self.datasets[data_key]['train']
        df_val = self.datasets[data_key]['val']
        use_validation = self.datasets[data_key]['use_validation']

        for k, record in transformation_dicts.items():
            column = record['column']
            strategy = record['strategy']
            parameters = record['parameters']

            if strategy in ['quantile_scaler', 'standard_scaler', 'power_transform']:
                if strategy == 'quantile_scaler':
                    scaler = preprocessing.QuantileTransformer()
                elif strategy == 'standard_scaler':
                    scaler = preprocessing.StandardScaler()
                elif strategy == 'power_transform':
                    scaler = preprocessing.PowerTransformer()
                else:
                    continue
                df_train[column] = scaler.fit_transform(df_train[column].values.reshape(-1, 1))
                if use_validation:
                    df_val[column] = scaler.transform(df_val[column].values.reshape(-1, 1))

            elif strategy in ['dictionary_encode']:
                if strategy == 'dictionary_encode':
                    scaler = DictionaryEncoder()
                else:
                    continue

                df_train[column] = scaler.fit_transform(df_train[column].tolist())
                if use_validation:
                    df_val[column] = scaler.transform(df_val[column].tolist())

            elif strategy in ['one_hot_encoding']:
                if strategy == 'one_hot_encoding':
                    scaler = OHE(column_prefix = column)
                    temp_df = scaler.fit_transform(df_train[column].tolist(), index = df_train.index)
                    df_train = df_train.drop(column, axis = 1)
                    df_train = df_train.join(temp_df)

                    if use_validation:
                        temp_df = scaler.transform(df_val[column].tolist(), index = df_val.index)
                        df_val = df_val.drop(column, axis = 1)
                        df_val = df_val.join(temp_df)
                else:
                    continue
            elif strategy in ['char_n_gram_3_5']:
                scaler = feature_extraction.text.CountVectorizer(analyzer = 'char', ngram_range=(3, 5), max_features=40)
                scaler.fit(df_train[column].fillna(''))
                temp_df = pd.DataFrame(columns = ['{0}_{1}_char_count'.format(column, w) for w in scaler.vocabulary_],
                                       data = scaler.transform(df_train[column].fillna('')).toarray(),
                                       index = df_train.index)
                df_train = df_train.drop(column, axis = 1)
                df_train = df_train.join(temp_df)

                if use_validation:
                    temp_df = pd.DataFrame(columns = ['{0}_{1}_char_count'.format(column, w) for w in scaler.vocabulary_],
                                           data = scaler.transform(df_val[column].fillna('')).toarray(),
                                           index = df_val.index)
                    df_val = df_val.drop(column, axis = 1)
                    df_val = df_val.join(temp_df)




        self.datasets[data_key]['train'] = df_train
        self.datasets[data_key]['val'] = df_val
        self.datasets[data_key]['use_validation'] = use_validation

    def apply_feature_engineering(self, data_key,
                        transformation_dicts):
        df_train = self.datasets[data_key]['train']
        df_val = self.datasets[data_key]['val']
        use_validation = self.datasets[data_key]['use_validation']

        # print(df_train.shape, df_val.shape, transformation_dicts)
        for record in transformation_dicts.values():
            # print(record)
            columns = record['columns']
            strategy = record['strategy']
            parameters = record['parameters']

            if strategy in ['sum', 'product', 'subtraction', 'ratio', 'min', 'max', 'var', 'identity']:
                col_names = [clean_text('{0}_{1}_{2}'.format('_'.join(columns), strategy, n)) for n, _ in enumerate(columns)]
                if strategy == 'sum':
                    df_train[col_names[0]] = df_train[columns].sum(axis = 1)
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns].sum(axis = 1)
                elif strategy == 'product':
                    df_train[col_names[0]] = df_train[columns[0]] * df_train[columns[1]]
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns[0]] * df_val[columns[1]]
                elif strategy == 'subtraction':
                    df_train[col_names[0]] = df_train[columns[0]] - df_train[columns[1]]
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns[0]] - df_val[columns[1]]
                elif strategy == 'ratio' :
                    df_train[col_names[0]] = df_train[columns[0]] / df_train[columns[1]]
                    df_train[col_names[0]] = df_train[col_names[0]].replace(np.inf, 0)
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns[0]] / df_val[columns[1]]
                        df_val[col_names[0]] = df_val[col_names[0]].replace(np.inf, 0)
                elif strategy == 'min' :
                    df_train[col_names[0]] = df_train[columns].min(axis = 1)
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns].min(axis = 1)
                elif strategy == 'max' :
                    df_train[col_names[0]] = df_train[columns].max(axis = 1)
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns].max(axis = 1)
                elif strategy == 'var' :
                    df_train[col_names[0]] = df_train[columns].var(axis = 1)
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns].var(axis = 1)
                elif strategy == 'pca' :
                    pca = decomposition.PCA()
                    pca.fit_transform(df_train[columns])
                    df_train[col_names] = pca.transform(df_train[columns])
                    if use_validation:
                        df_val[col_names] = pca.transform(df_val[columns])
                elif strategy == 'identity' :
                    df_train[col_names[0]] = df_train[columns].copy()
                    if use_validation:
                        df_val[col_names[0]] = df_val[columns].copy()
                else:
                    continue

        self.datasets[data_key]['train'] = df_train
        self.datasets[data_key]['val'] = df_val
        self.datasets[data_key]['use_validation'] = use_validation

    def get_data(self, data_key):
        return self.datasets[data_key]

    def get_data_description(self, data_key, column_subset=None):
        return get_dataset_description(self.datasets[data_key]['train'], self.target, column_subset=column_subset)


if __name__ == '__main__':
    a = [random.randint(1, 5) + 100 for _ in range(100)]
    ohe = DictionaryEncoder()
    ohe.fit(a)
    b = ohe.transform(a)

