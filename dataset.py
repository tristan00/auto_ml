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
temp_col = 'temp_col'

def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def get_col_description_dict(s):
    d = dict()
    if s.dtype in pd_numerics:
        d['nan_count'] = int(s.isna().sum())
        d['mean'] = float(s.mean())
        d['min'] = float(s.min())
        d['max'] = float(s.max())
        d['median'] = float(s.median())
        d['skew'] = float(s.skew())
        d['perc_unique'] = (int(s.nunique()) - 1)/max(s.shape)
        d['perc_of_values_mode'] = s.value_counts(normalize=True, dropna=False).iloc[0]
        d['mode'] = s.value_counts(normalize=True, dropna=False).index[0]
        d['kurtosis'] = stats.kurtosis(s.dropna().tolist())
    else:
        d['nan_count'] = int(s.isna().sum())
        d['nunique'] = int(s.nunique())
        d['perc_of_values_mode'] = s.value_counts(normalize=True, dropna=False).iloc[0]
        d['mode'] = s.value_counts(normalize=True, dropna=False).index[0]
    return d


def get_dataset_description(df, target, original_columns):
    dataset_profile = dict()

    if df[target].dtype in pd_numerics:
        target_type = 'numeric'
    else:
        target_type = 'string'

    dataset_profile['columns'] = dict()

    if temp_col in df.columns:
        df = df.drop(temp_col, axis=1)

    df_columns = df.columns.tolist()
    for i in df_columns:
        dataset_profile['columns'][i] = dict()
        if i == target:
            dataset_profile['columns'][i]['target'] = 1
        else:
            dataset_profile['columns'][i]['target'] = 0

        if i in original_columns:
            dataset_profile['columns'][i]['original_column'] = 1
        else:
            dataset_profile['columns'][i]['original_column'] = 0

        if df[i].dtype in pd_numerics:
            dataset_profile['columns'][i]['type'] = 'numeric'
            dataset_profile['columns'][i].update(get_col_description_dict(df[i]))
            dataset_profile['columns'][i]['perc_of_values_mode'] = df[i].value_counts(normalize=True, dropna=False).iloc[0]

            if target_type == 'numeric':
                slope, intercept, r_value, p_value, std_err = stats.linregress(df[i], df[target])
                dataset_profile['columns'][i]['target_slope'] = slope
                dataset_profile['columns'][i]['target_intercept'] = intercept
                dataset_profile['columns'][i]['target_r_value'] = r_value
                dataset_profile['columns'][i]['target_p_value'] = p_value
                dataset_profile['columns'][i]['target_std_err'] = std_err

            else:
                unique_values = set(df[target])
                value_mapping = {k: n for n, k in enumerate(unique_values)}
                df[temp_col] = df[target].replace(value_mapping).astype(int)
                subsets = [df[df[temp_col] == temp_target_value][i].tolist()
                           for temp_target_value in list(value_mapping.values())]
                f_stat, p_value = stats.f_oneway(*subsets)
                dataset_profile['columns'][i]['target_p_value'] = p_value
                dataset_profile['columns'][i]['target_f_stat'] = f_stat

        else:
            dataset_profile['columns'][i]['type'] = 'string'
            dataset_profile['columns'][i].update(get_col_description_dict(df[i]))

            if target_type == 'numeric':
                unique_values = set(df[i])
                value_mapping = {k: n for n, k in enumerate(unique_values)}
                df[temp_col] = df[i].replace(value_mapping).astype(int)
                subsets = [df[df[temp_col] == temp_target_value][target].tolist() for temp_target_value in list(value_mapping.values())]
                f_stat, p_value = stats.f_oneway(*subsets)
                dataset_profile['columns'][i]['target_p_value'] = p_value
                dataset_profile['columns'][i]['target_f_stat'] = f_stat

            else:
                unique_values_col = set(df[i])
                unique_values_target = set(df[target])

                value_mapping_col = {k: n + 1 for n, k in enumerate(unique_values_col)}
                value_mapping_target = {k: n + 1 for n, k in enumerate(unique_values_target)}

                temp_col_s = df[i].replace(value_mapping_col).astype(int).tolist()
                temp_target_s = df[target].replace(value_mapping_target).astype(int).tolist()

                chi_stat, p_value = stats.chisquare(temp_col_s, temp_target_s)
                dataset_profile['columns'][i]['target_p_value'] = p_value
                dataset_profile['columns'][i]['target_chi_stat'] = chi_stat

    original_columns_df = pd.DataFrame.from_dict([i for i in list(dataset_profile['columns'].values()) if i['original_column'] == 0])

    dataset_profile['general'] = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'feature_columns_size':original_columns_df.shape[0]}

    if original_columns_df.shape[0] > 0:
        for i in original_columns_df.columns.tolist():
            if original_columns_df[i].dtype in pd_numerics and original_columns_df[i].nunique() > 1:
                col_d = get_col_description_dict(original_columns_df[i])
                for k, v in col_d.items():
                    dataset_profile['general']['{0}_{1}'.format(i, k)] = v
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
            print(self.initial_dataset_description)

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
                                           'general_description': dataset_description['general'],
                                           'dataset_id': self.dataset_id,
                                           'datapath': self.data_path,
                                           'transformation_type': transformation_obj.transformation_type,
                                           'transformation_parameters': transformation_obj.transformation_parameters})

    def apply_n_random_transformations(self, n):
        for iteration in range(n):
            dataset_description = get_dataset_description(self.train_data, self.target, self.initial_columns)
            print(dataset_description)
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
    path = r'C:\Users\TristanDelforge\Documents\data_files\mushroom-classification/mushrooms.csv'

    d = DataSet()
    d.load_data(path, target='class')
    d.apply_n_random_transformations(100)
    train_x, train_y = d.get_train_data()
    recs = d.transformation_record
    a = 1
    # df = pd.read_csv(path)
    # columns = df.columns.tolist()
    # with open('sample_dataset_description.json', 'w') as f:
    #     json.dump(get_dataset_description(df, 'SalePrice', columns), f)
