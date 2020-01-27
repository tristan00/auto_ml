import pandas as pd
from dataset import DataSet
from model import get_n_random_models
from common import path
import random
from sklearn import ensemble, metrics
import uuid
import numpy as np
import traceback
import tqdm
import math
import json
import time

def get_metrics(truth, preds, problem_type):
    if problem_type == 'regression':

        try:
            mae = metrics.mean_absolute_error(truth, preds)
        except:
            mae = None

        try:
            mse = metrics.mean_squared_error(truth, preds)
        except:
            mse = None

        try:
            msle = metrics.mean_squared_log_error(truth, preds)
        except:
            msle = None

        try:
            r2 = metrics.r2_score(truth, preds)
        except:
            r2 = None

        return {'mean_absolute_error':mae,
                'mean_squared_error':mse,
                'r2_score':r2,
                'mean_squared_log_error':msle}
    else:
        raise NotImplementedError


def run_data_pipeline_value_predictions(data_path, target, problem_type, ):
    pass


def run_random_pipelines(data_path, target, problem_type, num_of_data_pipelines = 2, num_of_model_parameters = 2, num_of_transformations = 10):
    start_time = time.time()
    run_id = str(uuid.uuid4().hex)
    dataset_records = list()
    model_records = list()
    result_records = list()
    datasets = list()

    for i in range(num_of_data_pipelines):
        d = DataSet()
        d.load_data(data_path, target=target)
        datasets.append(d)

    print('data loaded')
    for d in datasets:
        d.apply_n_random_transformations(num_of_transformations)
        dataset_records.append(d.transformation_record)

    print('transformations loaded')
    models = get_n_random_models(problem_type, num_of_model_parameters)
    for m in models:
        model_records.append(m.get_model_description())

    print('start')
    for d in datasets:
        x_train, y_train = d.get_train_data()
        x_val, y_val = d.get_validation_data()
        x_test, y_test = d.get_test_data()
        for m in models:
            rec = {'model_id':m.model_id,
                   'dataset_id':d.dataset_id,
                   'problem_type':problem_type,
                   'strategy': 'run_random_pipelines'}
            try:
                m.fit(x_train, y_train)
                val_preds = m.predict(x_val)
                rec['validation_metrics'] = get_metrics(y_val, val_preds, problem_type)
                test_preds = m.predict(x_test)
                rec['test_metrics'] = get_metrics(y_test, test_preds, problem_type)
                rec['success'] = 1
            except:
                rec['success'] = 0
                traceback.print_exc()
            result_records.append(rec)

    with open('{path}/{run_id}.json'.format(path=path, run_id=run_id), 'w') as f:
        json.dump({'models': model_records,
                   'datasets': dataset_records,
                   'results': result_records,
                   'runtime': time.time() - start_time}, f)


if __name__ == '__main__':
    for i in range(1000):
        print('iteration {}'.format(i))
        run_random_pipelines('/home/td/Documents/datasets/housing_prices/train.csv',
                             'SalePrice',
                             'regression',
                             num_of_data_pipelines=20,
                             num_of_model_parameters=20,
                             num_of_transformations=10)
