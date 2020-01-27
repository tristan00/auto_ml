import pandas as pd
import numpy as np
import random
from sklearn import preprocessing, decomposition, model_selection, feature_extraction
import re
from scipy import stats
import json
from transformation import get_n_random_transformations
import time
import uuid

pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel = -999999999
string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'


def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def get_dataset_description(df, target, original_columns):
    dataset_profile = dict()

    feature_col_counter = 0


    if df[target].dtype in pd_numerics:
        target_type = 'numeric'
    else:
        target_type = 'string'

    dataset_profile['columns'] = dict()
    for i in df.columns:
        dataset_profile['columns'][i] = dict()
        if i == target:
            dataset_profile['columns'][i]['target'] = 1
        else:
            dataset_profile['columns'][i]['target'] = 0

        if i in original_columns:
            dataset_profile['columns'][i]['original_column'] = 1
        else:
            dataset_profile['columns'][i]['original_column'] = 0
            feature_col_counter += 1

        if df[i].dtype in pd_numerics:
            dataset_profile['columns'][i]['type'] = 'numeric'
            dataset_profile['columns'][i]['nan_count'] = int(df[i].isna().sum())
            dataset_profile['columns'][i]['mean'] = float(df[i].mean())
            dataset_profile['columns'][i]['min'] = float(df[i].min())
            dataset_profile['columns'][i]['max'] = float(df[i].max())
            dataset_profile['columns'][i]['median'] = float(df[i].median())
            dataset_profile['columns'][i]['skew'] = float(df[i].skew())
            dataset_profile['columns'][i]['nunique'] = int(df[i].nunique())
            dataset_profile['columns'][i]['perc_of_values_mode'] = df[i].value_counts(normalize=True, dropna=False).iloc[0]
            if target_type == 'numeric':
                _, _, r_value, _, _ = stats.linregress(df[i], df[target])
                dataset_profile['columns'][i]['target_r_value'] = r_value
            else:
                dataset_profile['columns'][i]['target_r_value'] = None

        else:
            dataset_profile['columns'][i]['type'] = 'string'
            dataset_profile['columns'][i]['nan_count'] = int(df[i].isna().sum())
            dataset_profile['columns'][i]['nunique'] = int(df[i].nunique())
            dataset_profile['columns'][i]['perc_of_values_mode'] = df[i].value_counts(normalize=True, dropna=False).iloc[0]

    dataset_profile['general'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'feature_columns':feature_col_counter
    }


    return dataset_profile


class DataSet:

    def __init__(self):
        self.transformation_record = list()
        self.output_columns = list()
        self.dataset_id = str(uuid.uuid4().hex)

    def load_data(self, data_path,
                  data_format='csv',
                  validation_size=.2,
                  test_size=.2,
                  csv_sep=',',
                  target=None,
                  nrows=None):
        self.data_path = data_path

        self.datasets = dict()
        if data_format == 'csv':
            self.target = target
            if nrows:
                self.raw_data = pd.read_csv(self.data_path, csv_sep, nrows=nrows)
            else:
                self.raw_data = pd.read_csv(self.data_path, csv_sep)

            if target:
                self.raw_data = self.raw_data[[target] + [i for i in self.raw_data.columns if i != target]]
            # self.raw_data = self.raw_data.set_index(list(range(self.raw_data.shape[0])))
            train_indices, test_indices = model_selection.train_test_split(list(self.raw_data.index), random_state=1,
                                                                           test_size=test_size)
            train_indices, validation_indices = model_selection.train_test_split(train_indices, random_state=1,
                                                                                 test_size=validation_size)

            # self.test_indices = random.sample(list(self.raw_data.index), int(test_size * self.raw_data.shape[0]))
            # self.train_indices = [i for i in self.raw_data.index if i not in self.test_indices]

            self.train_data = self.raw_data.loc[train_indices].copy()
            self.validation_data = self.raw_data.loc[validation_indices].copy()
            self.test_data = self.raw_data.loc[test_indices].copy()

            tmp_columns = self.train_data.columns.tolist()
            temp_dataset_description = get_dataset_description(self.train_data, self.target,
                                                               tmp_columns)

            self.apply_nan_fill(temp_dataset_description)

            self.initial_columns = self.train_data.columns.tolist()
            self.initial_dataset_description = get_dataset_description(self.train_data, self.target,
                                                                       self.initial_columns)

        else:
            raise NotImplementedError

    def apply_nan_fill(self, dataset_description):
        for column_name, column_dict in dataset_description['columns'].items():
            if column_dict['target'] or not column_dict['nan_count']:
                continue

            if column_dict['type'] == 'numeric':
                col_max = self.train_data[column_name].max() + 1
                col_min = self.train_data[column_name].min() - 1

                self.train_data['{}_nan_max'.format(column_name)] = self.train_data[column_name].fillna(col_max)
                self.train_data['{}_nan_min'.format(column_name)] = self.train_data[column_name].fillna(col_min)

                self.validation_data['{}_nan_max'.format(column_name)] = self.validation_data[column_name].fillna(
                    col_max)
                self.validation_data['{}_nan_min'.format(column_name)] = self.validation_data[column_name].fillna(
                    col_min)

                self.test_data['{}_nan_max'.format(column_name)] = self.test_data[column_name].fillna(col_max)
                self.test_data['{}_nan_min'.format(column_name)] = self.test_data[column_name].fillna(col_min)

            elif column_dict['type'] == 'string':
                self.train_data['{}_nan_sentinel'.format(column_name)] = self.train_data[column_name].fillna(
                    string_sentinel)
                self.validation_data['{}_nan_sentinel'.format(column_name)] = self.validation_data[column_name].fillna(
                    string_sentinel)
                self.test_data['{}_nan_sentinel'.format(column_name)] = self.test_data[column_name].fillna(
                    string_sentinel)

            else:
                raise NotImplementedError

            self.train_data = self.train_data.drop(column_name, axis=1)
            self.validation_data = self.validation_data.drop(column_name, axis=1)
            self.test_data = self.test_data.drop(column_name, axis=1)

    def apply_transformation(self, transformation_obj, dataset_description):
        transformation_obj.fit(self.train_data)
        self.train_data = transformation_obj.transform(self.train_data)
        self.validation_data = transformation_obj.transform(self.validation_data)
        self.test_data = transformation_obj.transform(self.test_data)
        self.output_columns.extend(transformation_obj.output_columns)
        self.output_columns = sorted(list(set(self.output_columns)))

        input_column_descriptions = list()
        for i in transformation_obj.input_columns:
            input_column_descriptions.append(dataset_description['columns'][i])

        self.transformation_record.append({'input_column_descriptions': input_column_descriptions,
                                           'dataset_id': self.dataset_id,
                                           'datapath': self.data_path,
                                           'transformation_type': transformation_obj.transformation_type,
                                           'transformation_parameters': transformation_obj.transformation_parameters})

    def apply_n_random_transformations(self, n):
        for iteration in range(n):
            dataset_description = get_dataset_description(self.train_data, self.target, self.initial_columns)
            transformation_objs = get_n_random_transformations(dataset_description, 1)
            transformation_obj = transformation_objs[0]
            self.apply_transformation(transformation_obj, dataset_description)

    def get_n_random_transformations(self, n):
        dataset_description = get_dataset_description(self.train_data, self.target, self.initial_columns)
        transformation_objs = get_n_random_transformations(dataset_description, n)
        return transformation_objs

    def get_train_data(self):
        return self.train_data[self.output_columns], self.train_data[self.target]

    def get_validation_data(self):
        return self.train_data[self.output_columns], self.train_data[self.target]

    def get_test_data(self):
        return self.train_data[self.output_columns], self.train_data[self.target]


if __name__ == '__main__':
    path = '/home/td/Documents/datasets/housing_prices/train.csv'

    d = DataSet()
    d.load_data(path, target='SalePrice')
    d.apply_n_random_transformations(10)
    train_x, train_y = d.get_train_data()
    recs = d.transformation_record
    a = 1
    # df = pd.read_csv(path)
    # columns = df.columns.tolist()
    # with open('sample_dataset_description.json', 'w') as f:
    #     json.dump(get_dataset_description(df, 'SalePrice', columns), f)
