import pandas as pd
from dataset import DataSet
from transformation import Transformation
from model import Model
from model import get_n_random_models
from common import path, get_metrics
import random
from sklearn import ensemble, metrics
import uuid
import numpy as np
import traceback
import tqdm
import math
import json
import time
import os
import gc
import copy



def build_past_result_dataset(data_path, target, metric, ):
    pass


def run_data_pipeline_value_predictions(data_path, target, problem_type,
                                        run_id = None,
                                        num_of_random_iterations = 100,
                                        num_of_data_pipelines = 10,
                                        num_of_model_parameters = 10,
                                        prob_of_ending_transformations = .1,
                                        epsilon_decay = .1,
                                        starting_epsilon = .95,
                                        iteration_size = 100,
                                        iteration_early_stopping_patience = 1,
                                        transformation_to_query = 1000
                                        ):
    '''
    Algorithm:

    Run a fully random set of transformation pipelines and model parameters to create initial dataset.
    For each iteration:
        Read records, fit a transformation that creates fixed size tabular output given for the model parameters, current feature description and transformation description.
        Model parameters + current feature description are the state,the transformation description is the action.
        Train a model to estimate a loss metric given a state and action.
        For each iteration step:
            Rank model parameters based on avg past results + time decay. Use gittins index or similar.
            Pick new model param: Epsilon prob of picking a random model parameter. Else take one of the best model parameters so far.
            Load fresh dataset.
            For each transformation:
                Query n random possible transformations.
                Pick transformation: Epsilon prob of picking a random transformation from this set. Else use model to rank transformations and pick best one.

        Decay epsilon.
        If iteration does not improve on previous iterations, exit.

    '''


    # match dataset of
    if not run_id:
        run_id = str(uuid.uuid4().hex)

    # for _ in range(num_of_random_iterations):
    #     run_random_pipelines(data_path, target, problem_type, run_id=run_id,
    #                          num_of_data_pipelines=num_of_data_pipelines,
    #                          num_of_model_parameters=num_of_model_parameters,
    #                          prob_of_ending_transformations=prob_of_ending_transformations)
    with open('/home/td/Documents/datasets/auto_ml/1.json', 'r') as f:
        past_results = json.load(f)

def run_genetic_data_pipeline(data_path, target):
    '''

    Run a fully random set of transformation pipelines and model parameters to create initial dataset.
    For each iteration:
        Take best n data pipelines. Generate groups of these pipelines.


    '''



def fit_meta_model(train_files_path,
                   temp_dataset_path,
                   metric,
                   ):
    with open(train_files_path, 'r') as f:
        past_results = json.load(f)

    model_df = pd.DataFrame.from_dict(past_results['models'])
    results_df = pd.DataFrame.from_dict(past_results['results'])
    datasets_df = pd.DataFrame.from_dict(past_results['datasets'])

    merged_df = results_df[['validation_metrics_{}'.format(metric), 'model_id', 'dataset_id']].merge(datasets_df)
    merged_df = merged_df.merge(model_df)

    merged_df = merged_df.drop(['model_id', 'dataset_id', 'transformation_id'], axis = 1)
    merged_df.to_csv(temp_dataset_path, index = False, sep = '|')

    d = DataSet()
    d.load_data(temp_dataset_path, target='validation_metrics_{}'.format(metric))
    for k in d.dataset_description['columns'].keys():
        if d.dataset_description['columns']['target']:
            continue
        if d.dataset_description['columns']['type'] != 'numeric':
            t = Transformation(k, 'dictionary_encode',
                               dict(),
                 [k],
                 ['{}_dictionary_encoded'.format(k)])
            d.apply_transformation(t)
        if d.dataset_description['columns']['type'] == 'numeric':
            t = Transformation(k, 'identity',
                               dict(),
                 [k],
                 ['{}_dictionary_encoded'.format(k)])
            d.apply_transformation(t)

    m = Model('LGBMRegressor', {'objective':'l1',
                    'boosting_type':'gbdt',
                   'num_leaves':64,
                   'learning_rate':.1,
                   'n_estimators':100})

    x_train, y_train = d.get_train_data()
    m.fit(x_train, y_train)
    return m, d.da

def run_random_pipelines(data_path, target, problem_type, run_id = None,
                         iteration_num = None,
                         num_of_data_pipelines = 10,
                         num_of_model_parameters = 10,
                         prob_of_ending_transformations = .1,
                        min_num_of_transformations = 10,
                         max_num_of_transformations = 100,
                         output_path = None):
    test_run_id = str(uuid.uuid4().hex)

    gc.collect()
    start_time = time.time()
    if not run_id:
        run_id = str(uuid.uuid4().hex)
    dataset_records = list()
    model_records = list()
    result_records = list()
    datasets = list()

    print('Loading data: {}'.format(time.time() - start_time))

    for i in range(num_of_data_pipelines):
        if not datasets:
            d = DataSet()
            d.load_data(data_path, target=target)
            datasets.append(d)
        else:
            next_dataset = copy.deepcopy(datasets[0])
            next_dataset.dataset_id = str(uuid.uuid4().hex)
            datasets.append(next_dataset)
        print('Data loaded: {0} of {1}, {2}'.format(len(datasets), num_of_data_pipelines, time.time() - start_time))
    print()

    print('Applying transformations')
    for n, d in enumerate(datasets):
        for i in range(max_num_of_transformations):
            d.apply_n_random_transformations(1)
            if n > min_num_of_transformations and random.random() < prob_of_ending_transformations:
                break
        dataset_records.extend(d.transformation_record)
        print('{0} transformations applied to data pipeline {1} of {2}'.format(i, n, num_of_data_pipelines))
    print()

    models = get_n_random_models(problem_type, num_of_model_parameters)
    for m in models:
        model_records.append(m.get_model_description())

    print('starting model fitting, {0}, {1}'.format(len(datasets), len(models)))
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
                validation_metrics = get_metrics(y_val, val_preds, problem_type)
                validation_metrics = {'validation_metrics_{}'.format(k):v for k,v in validation_metrics.items()}
                rec.update(validation_metrics)
                test_preds = m.predict(x_test)
                test_metrics = get_metrics(y_test, test_preds, problem_type)
                test_metrics = {'test_metrics_{}'.format(k):v for k, v in test_metrics.items()}
                rec.update(test_metrics)
                rec['success'] = 1
            except:
                rec['success'] = 0
                traceback.print_exc()
            print(m.model_id, d.dataset_id, rec)
            result_records.append(rec)
        print('finished dataset: {}'.format(d.dataset_id))

    if os.path.exists('{path}/{run_id}.json'.format(path=output_path, run_id=run_id)):
        with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'r') as f:
            past_results = json.load(f)
            model_records.extend(past_results['models'])
            dataset_records.extend(past_results['datasets'])
            result_records.extend(past_results['results'])

    with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'w') as f:
        json.dump({'models': model_records,
                   'datasets': dataset_records,
                   'results': result_records}, f)
    print('finished')
    print()
    print()


if __name__ == '__main__':
    for i in range(1000):
        run_random_pipelines(r'C:\Users\TristanDelforge\Documents\data_files\stackoverflow_survey_clean.csv',
                             'ConvertedSalary',
                             'regression',
                             num_of_data_pipelines=4,
                             num_of_model_parameters=4,
                             prob_of_ending_transformations=.1,
                             min_num_of_transformations = 4,
                             max_num_of_transformations = 20,
                             run_id = 10,
                             output_path = r'C:\Users\TristanDelforge\Documents\data_files\auto_ml')
