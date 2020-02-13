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
import copy
import traceback


pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel = -999999999
numerical_sentinel_offset = 999
string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'
temp_col = 'temp_col'

def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def get_col_description_dict(s):
    d = dict()
    if s.dtype in pd_numerics:
        # d['nan_count'] = int(s.isna().sum())
        d['mean'] = float(s.mean())
        # d['min'] = float(s.min())
        # d['max'] = float(s.max())
        # d['median'] = float(s.median())
        d['var'] = float(s.skew())
        d['skew'] = float(s.skew())
        d['perc_unique'] = (int(s.nunique()) - 1)/max(s.shape)
        # d['perc_of_values_mode'] = float(s.value_counts(normalize=True, dropna=False).iloc[0])
        # d['mode'] = float(s.value_counts(normalize=True, dropna=False).index[0])
        d['kurtosis'] = float(stats.kurtosis(s.dropna().tolist()))
    else:
        # d['nan_count'] = int(s.isna().sum())
        d['nunique'] = int(s.nunique())
        d['perc_of_values_mode'] = float(s.value_counts(normalize=True, dropna=False).iloc[0])
        # d['mode'] = s.value_counts(normalize=True, dropna=False).index[0]
    return d


def get_fixed_size_col_descriptions_dict(df, s_list, target):
    col_descriptions = list()
    for s in s_list:
        pass

def get_column_relation_description(df, col1, col2, col1_type, col2_type):
    relation_description = dict()
    if col1_type == 'numeric' and col2_type == 'numeric':
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[col1], df[col2])
        relation_description['slope'] = slope
        relation_description['r_value'] = r_value
        relation_description['p_value'] = p_value

    elif (col1_type != 'numeric' and col2_type == 'numeric'):
        unique_values =set(df[col1])
        subsets = [df[df[col1] == temp_target_value][col2].tolist()
                   for temp_target_value in list(unique_values)]
        f_stat, p_value = stats.f_oneway(*subsets)
        relation_description['p_value'] = p_value
        relation_description['f_stat'] = f_stat

    elif (col1_type == 'numeric' and col2_type != 'numeric'):
        unique_values =set(df[col1])
        subsets = [df[df[col2] == temp_target_value][col1].tolist()
                   for temp_target_value in list(unique_values)]
        f_stat, p_value = stats.f_oneway(*subsets)
        relation_description['p_value'] = p_value
        relation_description['f_stat'] = f_stat

    elif (col1_type != 'numeric' and col2_type != 'numeric'):
        unique_values_col1 = set(df[col1])
        unique_values_col2 = set(df[col2])

        value_mapping_col1 = {k: n + 1 for n, k in enumerate(unique_values_col1)}
        value_mapping_col2 = {k: n + 1 for n, k in enumerate(unique_values_col2)}

        temp_col1 = df[col1].replace(value_mapping_col1).astype(int).tolist()
        temp_col2 = df[col2].replace(value_mapping_col2).astype(int).tolist()

        chi_stat, p_value = stats.chisquare(temp_col1, temp_col2)
        relation_description['p_value'] = p_value
        relation_description['chi_stat'] = chi_stat

    else:
        raise NotImplementedError

    return relation_description




class DataSet:

    def __init__(self):
        self.applied_transformation_objs = list()
        self.transformation_record = list()
        self.output_columns = list()
        self.dataset_id = str(uuid.uuid4().hex)
        self.dataset_description = dict()
        self.dataset_description['columns'] = dict()

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


            self.apply_nan_fill()

            self.initial_columns = self.train_data.columns.tolist()
            self.get_dataset_description()

        else:
            raise NotImplementedError

    def get_dataset_description(self):
        if self.train_data[self.target].dtype in pd_numerics:
            target_type = 'numeric'
        else:
            target_type = 'string'


        # self.dataset_description['columns'] = dict()

        if temp_col in self.train_data.columns:
            self.train_data = self.train_data.drop(temp_col, axis=1)

        df_columns = self.train_data.columns.tolist()
        for i in df_columns:
            if i in self.dataset_description.get('columns', dict()).keys():
                continue

            self.dataset_description['columns'][i] = dict()
            if i == self.target:
                self.dataset_description['columns'][i]['target'] = 1
            else:
                self.dataset_description['columns'][i]['target'] = 0

            # if i in self.initial_columns:
            #     self.dataset_description['columns'][i]['original_column'] = 1
            # else:
            #     self.dataset_description['columns'][i]['original_column'] = 0

            if self.train_data[i].dtype in pd_numerics:
                self.dataset_description['columns'][i]['type'] = 'numeric'
                self.dataset_description['columns'][i].update(get_col_description_dict(self.train_data[i]))
                self.dataset_description['columns'][i]['perc_of_values_mode'] = \
                self.train_data[i].value_counts(normalize=True, dropna=False).iloc[0]

                if target_type == 'numeric':
                    slope, intercept, r_value, p_value, std_err = stats.linregress(self.train_data[i], self.train_data[self.target])
                    self.dataset_description['columns'][i]['target_slope'] = slope
                    # self.dataset_description['columns'][i]['target_intercept'] = intercept
                    self.dataset_description['columns'][i]['target_r_value'] = r_value
                    self.dataset_description['columns'][i]['target_p_value'] = p_value
                    # self.dataset_description['columns'][i]['target_std_err'] = std_err

                else:
                    unique_values = set(self.train_data[self.target])
                    value_mapping = {k: n for n, k in enumerate(unique_values)}
                    self.train_data[temp_col] = self.train_data[self.target].replace(value_mapping).astype(int)
                    subsets = [self.train_data[self.train_data[temp_col] == temp_target_value][i].tolist()
                               for temp_target_value in list(value_mapping.values())]
                    f_stat, p_value = stats.f_oneway(*subsets)
                    self.dataset_description['columns'][i]['target_p_value'] = p_value
                    self.dataset_description['columns'][i]['target_f_stat'] = f_stat

            else:
                self.dataset_description['columns'][i]['type'] = 'string'
                self.dataset_description['columns'][i].update(get_col_description_dict(self.train_data[i]))

                if target_type == 'numeric':
                    unique_values = set(self.train_data[i])
                    value_mapping = {k: n for n, k in enumerate(unique_values)}
                    self.train_data[temp_col] = self.train_data[i].replace(value_mapping).astype(int)
                    subsets = [self.train_data[self.train_data[temp_col] == temp_target_value][self.target].tolist() for temp_target_value in
                               list(value_mapping.values())]
                    f_stat, p_value = stats.f_oneway(*subsets)
                    self.dataset_description['columns'][i]['target_p_value'] = p_value
                    self.dataset_description['columns'][i]['target_f_stat'] = f_stat

                else:
                    unique_values_col = set(self.train_data[i])
                    unique_values_target = set(self.train_data[self.target])

                    value_mapping_col = {k: n + 1 for n, k in enumerate(unique_values_col)}
                    value_mapping_target = {k: n + 1 for n, k in enumerate(unique_values_target)}

                    temp_col_s = self.train_data[i].replace(value_mapping_col).astype(int).tolist()
                    temp_target_s = self.train_data[self.target].replace(value_mapping_target).astype(int).tolist()

                    chi_stat, p_value = stats.chisquare(temp_col_s, temp_target_s)
                    self.dataset_description['columns'][i]['target_p_value'] = p_value
                    self.dataset_description['columns'][i]['target_chi_stat'] = chi_stat

        original_columns_df = pd.DataFrame.from_dict(
            [i for i in list(self.dataset_description['columns'].values()) if i not in self.initial_columns])
        self.dataset_description['general'] = {
            'rows': self.train_data.shape[0],
            'columns': self.train_data.shape[1],
            'feature_columns_size': original_columns_df.shape[0]}

        if original_columns_df.shape[0] > 0:
            for i in original_columns_df.columns.tolist():
                if original_columns_df[i].dtype in pd_numerics and original_columns_df[i].nunique() > 1:
                    col_d = get_col_description_dict(original_columns_df[i])
                    for k, v in col_d.items():
                        self.dataset_description['general']['{0}_{1}'.format(i, k)] = v
        self.dataset_description =  copy.deepcopy(self.dataset_description)

    def apply_nan_fill(self):
        columns = self.train_data.columns.tolist()
        for column_name in columns:
            if column_name == self.target or not (self.train_data[column_name].isna().sum() or self.validation_data[column_name].isna().sum() or self.test_data[column_name].isna().sum()):
                continue

            if self.train_data[column_name].dtype in pd_numerics:
                # col_max = self.train_data[column_name].max() + numerical_sentinel_offset
                col_min = self.train_data[column_name].min() - numerical_sentinel_offset
                if pd.isnull(col_min):
                    col_min = 0

                # self.train_data['{}_nan_max'.format(column_name)] = self.train_data[column_name].fillna(col_max)
                self.train_data['{}_nan_min'.format(column_name)] = self.train_data[column_name].fillna(col_min)

                # self.validation_data['{}_nan_max'.format(column_name)] = self.validation_data[column_name].fillna(
                #     col_max)
                self.validation_data['{}_nan_min'.format(column_name)] = self.validation_data[column_name].fillna(
                    col_min)

                # self.test_data['{}_nan_max'.format(column_name)] = self.test_data[column_name].fillna(col_max)
                self.test_data['{}_nan_min'.format(column_name)] = self.test_data[column_name].fillna(col_min)

            else:
                self.train_data['{}_nan_sentinel'.format(column_name)] = self.train_data[column_name].fillna(
                    string_sentinel)
                self.validation_data['{}_nan_sentinel'.format(column_name)] = self.validation_data[column_name].fillna(
                    string_sentinel)
                self.test_data['{}_nan_sentinel'.format(column_name)] = self.test_data[column_name].fillna(
                    string_sentinel)

            self.train_data = self.train_data.drop(column_name, axis=1)
            self.validation_data = self.validation_data.drop(column_name, axis=1)
            self.test_data = self.test_data.drop(column_name, axis=1)

    def apply_transformation(self, transformation_obj, fit_transformation = True):

        if fit_transformation:
            transformation_obj.fit(self.train_data)

        self.train_data = transformation_obj.transform(self.train_data)
        self.validation_data = transformation_obj.transform(self.validation_data)
        self.test_data = transformation_obj.transform(self.test_data)
        self.output_columns.extend(transformation_obj.output_columns)
        self.output_columns = sorted(list(set(self.output_columns)))

        transformation_parameters = {'transformation_parameter_{}'.format(k):v for k, v in transformation_obj.transformation_parameters.items()}

        input_column_descriptions = dict()
        for n, i in enumerate(transformation_obj.input_columns):
            input_column_description_copy = {'input_column_{0}_{1}'.format(n, k):v for k, v in self.dataset_description['columns'][i].items()}
            input_column_descriptions.update(input_column_description_copy)

        for n1, i in enumerate(transformation_obj.input_columns):
            for n2, j in enumerate(transformation_obj.input_columns):
                if i == j or n1 > n2:
                    continue
                relation_dict = get_column_relation_description(self.train_data, i, j, self.dataset_description['columns'][i]['type'], self.dataset_description['columns'][j]['type'])
                relation_dict = {'input_column_relation_{0}_{1}_{2}'.format(n1, n2, k): v for k, v in relation_dict.items()}
                input_column_descriptions.update(relation_dict)

        general_dataset_descriptions = {'general_dataset_description_{}'.format(k):v for k, v in self.dataset_description['general'].items()}


        new_record = {'dataset_id': self.dataset_id,
                      'transformation_id': transformation_obj.transformation_id,
                      'datapath': self.data_path,
                      'transformation_type': transformation_obj.transformation_type,
                      }
        new_record.update(transformation_parameters)
        new_record.update(input_column_descriptions)
        new_record.update(general_dataset_descriptions)

        self.transformation_record.append(copy.deepcopy(new_record))
        self.applied_transformation_objs.append(transformation_obj)

    def apply_n_random_transformations(self, n):
        for iteration in range(n):
            while True:
                try:
                    self.get_dataset_description()
                    transformation_objs = get_n_random_transformations(self.dataset_description, 1)
                    transformation_obj = transformation_objs[0]
                    self.apply_transformation(transformation_obj)
                    break
                except:
                    traceback.print_exc()

    def get_n_random_transformations(self, n):
        dataset_description = self.get_dataset_description()
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
