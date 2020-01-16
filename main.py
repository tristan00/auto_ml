import pandas as pd
from dataset import DataSet
from model import Model
import random
from sklearn import ensemble, metrics
import uuid
import numpy as np
import traceback
import tqdm
import math
import json


# def get_regression_metrics(truth, preds):
#     res_metric = metrics.mean_absolute_error(truth, preds)



def run_random_pipeline(data_path, target,
                        min_num_of_feature_engineering_steps = 4,
                        max_num_of_feature_engineering_steps = 12):
    print('start')
    data_manager = DataSet()
    initial_description = data_manager.load_data(data_path = data_path, target = target, test_size=.3)

    column_type_nan_fill_dict = dict()
    for i in initial_description['columns'].keys():
        if initial_description['columns'][i]['nan_count'] and not initial_description['columns'][i]['target']:
            c_type = initial_description['columns'][i]['type']
            column_type_nan_fill_dict[i] = random.choice(DataSet.column_type_nan_fill_dict[c_type])

    print('filled nans')
    column_type_transformation_dicts = dict()
    for i in initial_description['columns'].keys():
        if not initial_description['columns'][i]['target']:
            c_type = initial_description['columns'][i]['type']
            if c_type == 'string':
                strategy = random.choice(DataSet.column_type_transformation_dict[c_type])
                parameters = None
                column_type_transformation_dicts[i] = {'column':i,
                                                         'strategy':strategy,
                                                         'parameters':parameters}

    print('applied transformations')

    data_manager.apply_nan_fill('train_test', column_type_nan_fill_dict)
    data_manager.apply_transformations('train_test', column_type_transformation_dicts)

    starting_columns = list()
    num_of_feature_engineering_steps = random.randint(min_num_of_feature_engineering_steps, max_num_of_feature_engineering_steps)
    feature_engineering_steps = dict()

    latest_data_description = data_manager.get_data_description('train_test')
    columns = [i for i in latest_data_description['columns'].keys() if not latest_data_description['columns'][i]['target']]

    for step in tqdm.tqdm(list(range(num_of_feature_engineering_steps))):

        if not starting_columns:
            starting_columns = columns
        strategy = random.choice(list(DataSet.feature_engineering_dict.keys()))

        if DataSet.feature_engineering_dict[strategy]['min_columns'] == DataSet.feature_engineering_dict[strategy]['max_columns']:
            num_of_chosen_columns = DataSet.feature_engineering_dict[strategy]['min_columns']
        else:
            num_of_chosen_columns = random.randint(DataSet.feature_engineering_dict[strategy]['min_columns'],
                                                   DataSet.feature_engineering_dict[strategy]['max_columns'])

        chosen_columns = random.sample(columns, num_of_chosen_columns)
        next_steps = {'columns':chosen_columns,
                     'strategy':strategy,
                     'parameters':None}
        feature_engineering_steps[step] = next_steps

    data_manager.apply_feature_engineering('train_test', feature_engineering_steps)
    print('applied feature engineering')

    final_data_description =  data_manager.get_data_description('train_test')
    chosen_columns = [i for i in final_data_description['columns'].keys() if not final_data_description['columns'][i]['target']]
    chosen_columns = [i for i in chosen_columns if i not in starting_columns]
    final_data_description =  data_manager.get_data_description('train_test', column_subset=chosen_columns)

    data = data_manager.get_data('train_test')
    train_df = data['train']
    val_df = data['val']

    train_x = train_df[chosen_columns]
    train_y = train_df[target]
    val_x = val_df[chosen_columns]
    val_y = val_df[target]

    train_x = train_x.replace(np.inf, np.nan)
    train_x = train_x.replace(-np.inf, np.nan)
    train_x = train_x.fillna(0)
    val_x = val_x.replace(np.inf, np.nan)
    val_x = val_x.replace(-np.inf, np.nan)
    val_x = val_x.fillna(0)

    model_parameters = dict()
    model_choice = random.choice(list(Model.model_param_dict.keys()))
    for p, c in Model.model_param_dict[model_choice].items():
        if isinstance(c, list):
            model_parameters[p] = random.choice(c)
        if isinstance(c, tuple):
            if isinstance(c[0], float):
                model_parameters[p] = random.random() * (c[1] - c[0])
            elif isinstance(c[0], int):
                model_parameters[p] = random.randint(c[0], c[1])
            else:
                raise NotImplementedError

    model_parameters['model_type'] = model_choice
    model = Model(model_parameters)
    print('fitting model')
    model.fit(train_x, train_y)
    print('model fit')

    preds = model.predict(val_x)
    res_metric = metrics.mean_squared_log_error(val_y, preds)
    print(res_metric)

    # json.dumps(column_type_nan_fill_dict)
    # json.dumps(column_type_transformation_dicts)
    # json.dumps(feature_engineering_steps)
    # # print(initial_description)
    # json.dumps(initial_description)
    # json.dumps(final_data_description)
    # json.dumps(model_parameters)

    return {'column_type_nan_fill_dict': json.dumps(column_type_nan_fill_dict),
           'column_type_transformation_dicts': json.dumps(column_type_transformation_dicts),
           'feature_engineering_steps': json.dumps(feature_engineering_steps),
            'initial_description': json.dumps(initial_description),
            'final_data_description': json.dumps(final_data_description),
            'model_parameters': json.dumps(model_parameters),
           'res_metric': res_metric}


if __name__ == '__main__':
    housing_prices_dataset = {'data_path': '/media/td/Samsung_T5/ml_problems/housing_prices/train.csv',
                   'target': 'SalePrice',
                   'metric': 'rmlse'}
    allstate_dataset = {'data_path': '/media/td/Samsung_T5/ml_problems/allstate_severity/train_sample.csv',
                   'target': 'loss',
                   'metric': 'mae'}
    gpu_dataset = {'data_path': '/media/td/Samsung_T5/ml_problems/gpu/gpu_sample.csv',
                   'target': 'target',
                   'metric': 'mae'}

    dataset = housing_prices_dataset
    run_id = str(uuid.uuid4())
    result_list = list()
    for i in range(10000):
        try:
            print('iteration {}'.format(i))
            res = run_random_pipeline(dataset['data_path'],
                         dataset['target'])
            res.update(dataset)
            result_list.append(res)
            pd.DataFrame.from_dict(result_list).to_csv('/media/td/Samsung_T5/ml_problems/parameter_search/{}.csv'.format(run_id), index = False, sep = '|')
        except:
            traceback.print_exc()
