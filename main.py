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
from sklearn.preprocessing import StandardScaler
import gc


def build_past_result_dataset(data_path, target, metric, ):
    pass


def run_data_pipeline_q_value_predictions(data_path, target, problem_type,
                                        output_path,
                                        run_id = None,
                                        num_of_random_iterations = 1,
                                        num_of_data_pipelines = 10,
                                        prob_of_ending_transformations = .1,
                                        transformations_to_consider_per_step = 10,
                                        epsilon_decay = .1,
                                        starting_epsilon = .95,
                                        temp_dataset_path = '/tmp/meta_model_dataset.csv',
                                        evaluation_metric = '',
                                        min_num_of_transformations = 8,
                                        max_num_of_transformations = 64
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

    base_d = DataSet()
    base_d.load_data(data_path, target=target)

    epsilon = starting_epsilon
    if run_id is None:
        run_id = str(uuid.uuid4().hex)
    result_records = list()
    model_records = list()
    dataset_records = list()

    for _ in range(num_of_random_iterations):
        run_random_pipelines(data_path, target, problem_type,
                             base_dataset = base_d,
                             output_path=output_path,
                             run_id=run_id,
                             num_of_data_pipelines=num_of_data_pipelines,
                             num_of_model_parameters=num_of_data_pipelines,
                             min_num_of_transformations=min_num_of_transformations,
                             max_num_of_transformations=max_num_of_transformations,
                             start_time=start_time)

    with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'r') as f:
        past_results = json.load(f)

    result_records.extend(past_results['results'])
    model_records.extend(past_results['models'])
    dataset_records.extend(past_results['datasets'])

    while True:
        meta_model, meta_model_transformation_objs, meta_model_columns, meta_model_starting_columns = fit_meta_model('{path}/{run_id}.json'.format(path=output_path, run_id=run_id),
                   temp_dataset_path,
                   evaluation_metric)
        datasets = list()
        for i in range(num_of_data_pipelines):
            datasets.append(base_d.get_copy())
            print('Data loaded: {0} of {1}, {2}'.format(len(datasets), num_of_data_pipelines, time.time() - start_time))

        models = get_n_random_models(problem_type, num_of_data_pipelines)
        for m in models:
            model_records.append(m.get_model_description())

        for d in datasets:
            num_of_transformations = random.randint(min_num_of_transformations, max_num_of_transformations)
            chosen_model = models.pop()
            chosen_model_description = chosen_model.get_model_description()

            for _ in range(num_of_transformations):
                if random.random() < epsilon:
                    d.apply_n_random_transformations(1)
                else:
                    while True:
                        try:
                            possible_transformations = d.get_n_random_transformations(transformations_to_consider_per_step)
                            transformation_features = [d.get_transformation_record(t) for t in possible_transformations]
                            [i.update(chosen_model_description) for i in transformation_features]
                            temp_df = pd.DataFrame.from_dict(transformation_features)
                            for i in meta_model_starting_columns:
                                if i not in temp_df.columns:
                                    temp_df[i] = 0

                            temp_df['dummy_target'] = 0
                            temp_dataset = DataSet()
                            temp_dataset.load_data(df = temp_df, target='dummy_target')
                            for t in meta_model_transformation_objs:
                                temp_dataset.apply_transformation(t, fit_transformation=False)

                            x, y = temp_dataset.get_all_data()
                            x = x.sort_index()
                            preds = meta_model.predict(x[meta_model_columns])
                            chosen_transformation = possible_transformations[np.argmin(preds)]
                            d.apply_transformation(chosen_transformation)
                            break
                        except:
                            traceback.print_exc()

            dataset_records.extend(d.transformation_record)
            x_train, y_train = d.get_train_data()
            x_val, y_val = d.get_validation_data()
            x_test, y_test = d.get_test_data()
            rec = {'model_id': chosen_model.model_id,
                   'dataset_id': d.dataset_id,
                   'problem_type': problem_type,
                   'strategy': 'run_random_pipelines',
                   'timestamp': time.time() - start_time,
                   'epsilon':epsilon}
            try:
                chosen_model.fit(x_train, y_train)
                val_preds = chosen_model.predict(x_val)
                validation_metrics = get_metrics(y_val, val_preds, problem_type)
                validation_metrics = {'validation_metrics_{}'.format(k): v for k, v in validation_metrics.items()}
                rec.update(validation_metrics)
                test_preds = chosen_model.predict(x_test)
                test_metrics = get_metrics(y_test, test_preds, problem_type)
                test_metrics = {'test_metrics_{}'.format(k): v for k, v in test_metrics.items()}
                rec.update(test_metrics)
                rec['success'] = 1
            except:
                rec['success'] = 0
                traceback.print_exc()
            result_records.append(rec)

        with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'w') as f:
            json.dump({'models': model_records,
                       'datasets': dataset_records,
                       'results': result_records}, f)
        epsilon = epsilon*(1 - epsilon_decay)
        gc.collect()


def run_data_pipeline_q_value_predictions_simple(data_path, target, problem_type,
                                        output_path,
                                        run_id = None,
                                        num_of_random_iterations = 1,
                                        num_of_data_pipelines = 10,
                                        transformations_to_consider_per_step = 10,
                                        temp_dataset_path = '/tmp/meta_model_dataset.csv',
                                        evaluation_metric = '',
                                        min_num_of_transformations = 8,
                                        max_num_of_transformations = 64,
                                                 num_of_final_pipelines = 100
                                        ):
    base_d = DataSet()
    base_d.load_data(data_path, target=target)

    if run_id is None:
        run_id = str(uuid.uuid4().hex)
    result_records = list()
    model_records = list()
    dataset_records = list()

    for _ in range(num_of_random_iterations):
        run_random_pipelines(data_path, target, problem_type,
                             base_dataset = base_d,
                             output_path=output_path,
                             run_id=run_id,
                             num_of_data_pipelines=num_of_data_pipelines,
                             num_of_model_parameters=num_of_data_pipelines,
                             min_num_of_transformations=min_num_of_transformations,
                             max_num_of_transformations=max_num_of_transformations,
                             start_time=start_time)

    with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'r') as f:
        past_results = json.load(f)

    result_records.extend(past_results['results'])
    model_records.extend(past_results['models'])
    dataset_records.extend(past_results['datasets'])

    while True:
        meta_model, meta_model_transformation_objs, meta_model_columns, meta_model_starting_columns = fit_meta_model('{path}/{run_id}.json'.format(path=output_path, run_id=run_id),
                   temp_dataset_path,
                   evaluation_metric)
        datasets = list()
        for i in range(num_of_final_pipelines):
            datasets.append(base_d.get_copy())
            print('Data loaded: {0} of {1}, {2}'.format(len(datasets), num_of_data_pipelines, time.time() - start_time))

        models = get_n_random_models(problem_type, num_of_data_pipelines)
        for m in models:
            model_records.append(m.get_model_description())

        for d in datasets:
            chosen_model = models.pop()
            chosen_model_description = chosen_model.get_model_description()

            predicted_loss = None
            for _ in range(max_num_of_transformations):
                try:
                    possible_transformations = d.get_n_random_transformations(transformations_to_consider_per_step)
                    transformation_features = [d.get_transformation_record(t) for t in possible_transformations]
                    [i.update(chosen_model_description) for i in transformation_features]
                    temp_df = pd.DataFrame.from_dict(transformation_features)
                    for i in meta_model_starting_columns:
                        if i not in temp_df.columns:
                            temp_df[i] = 0

                    temp_df['dummy_target'] = 0
                    temp_dataset = DataSet()
                    temp_dataset.load_data(df = temp_df, target='dummy_target')
                    for t in meta_model_transformation_objs:
                        temp_dataset.apply_transformation(t, fit_transformation=False)

                    x, y = temp_dataset.get_all_data()
                    x = x.sort_index()
                    preds = meta_model.predict(x[meta_model_columns])

                    if not predicted_loss:
                        predicted_loss = preds.min()

                    if predicted_loss < preds.min():
                        break

                    predicted_loss = min(predicted_loss, preds.min())

                    chosen_transformation = possible_transformations[np.argmin(preds)]
                    d.apply_transformation(chosen_transformation)
                except:
                    traceback.print_exc()

            dataset_records.extend(d.transformation_record)
            x_train, y_train = d.get_train_data()
            x_val, y_val = d.get_validation_data()
            x_test, y_test = d.get_test_data()
            rec = {'model_id': chosen_model.model_id,
                   'dataset_id': d.dataset_id,
                   'problem_type': problem_type,
                   'timestamp': time.time() - start_time}
            try:
                chosen_model.fit(x_train, y_train)
                val_preds = chosen_model.predict(x_val)
                validation_metrics = get_metrics(y_val, val_preds, problem_type)
                validation_metrics = {'validation_metrics_{}'.format(k): v for k, v in validation_metrics.items()}
                rec.update(validation_metrics)
                test_preds = chosen_model.predict(x_test)
                test_metrics = get_metrics(y_test, test_preds, problem_type)
                test_metrics = {'test_metrics_{}'.format(k): v for k, v in test_metrics.items()}
                rec.update(test_metrics)
                rec['success'] = 1
            except:
                rec['success'] = 0
                traceback.print_exc()
            result_records.append(rec)

        with open('{path}/{run_id}.json'.format(path=output_path, run_id=run_id), 'w') as f:
            json.dump({'models': model_records,
                       'datasets': dataset_records,
                       'results': result_records}, f)
        gc.collect()


def fit_meta_model(train_files_path,
                   temp_dataset_path,
                   metric,
                   ):
    with open(train_files_path, 'r') as f:
        past_results = json.load(f)

    model_df = pd.DataFrame.from_dict(past_results['models'])
    results_df = pd.DataFrame.from_dict(past_results['results'])
    datasets_df = pd.DataFrame.from_dict(past_results['datasets'])

    print(results_df.columns.tolist())

    merged_df = results_df[['validation_metrics_{}'.format(metric), 'model_id', 'dataset_id']].merge(datasets_df)
    merged_df = merged_df.merge(model_df)

    merged_df = merged_df.drop(['model_id', 'dataset_id', 'transformation_id'], axis = 1)

    merged_df = merged_df.replace(np.inf, np.nan)
    merged_df = merged_df.dropna(subset = ['validation_metrics_{}'.format(metric)])

    s = StandardScaler()
    merged_df['validation_metrics_{}'.format(metric)] = s.fit_transform(merged_df['validation_metrics_{}'.format(metric)].values.reshape((-1, 1)))
    # merged_df.to_csv(temp_dataset_path, index = False, sep = '|')

    d = DataSet()
    d.load_data(df = merged_df, target='validation_metrics_{}'.format(metric))

    for k in d.dataset_description['columns'].keys():
        if d.dataset_description['columns'][k]['target']:
            continue
        if d.dataset_description['columns'][k]['type'] != 'numeric':
            t = Transformation(k, 'dictionary_encode',
                               dict(),
                 [k],
                 ['{}_dictionary_encoded'.format(k)])
            d.apply_transformation(t)
        if d.dataset_description['columns'][k]['type'] == 'numeric':
            t = Transformation(k, 'identity',
                               dict(),
                 [k],
                 ['{}_dictionary_encoded'.format(k)])
            d.apply_transformation(t)

    m = Model('LGBMRegressor', {'objective':'l2',
                    'boosting_type':'gbdt',
                   'num_leaves':64,
                   'learning_rate':.1,
                   'n_estimators':100})

    d.get_dataset_description()
    x_train, y_train = d.get_train_data()
    m.fit(x_train, y_train)

    print('fit_meta_model', d.output_columns)
    return m, d.applied_transformation_objs, x_train.columns.tolist(), merged_df.columns.tolist()


def run_random_pipelines(data_path, target, problem_type,
                         base_dataset = None,
                         run_id = None,
                         iteration_num = None,
                         num_of_data_pipelines = 10,
                         num_of_model_parameters = 10,
                         prob_of_ending_transformations = .1,
                        min_num_of_transformations = 10,
                         max_num_of_transformations = 100,
                         output_path = None,
                         start_time = 0.0):
    local_start_time = time.time()
    gc.collect()
    if not run_id:
        run_id = str(uuid.uuid4().hex)
    dataset_records = list()
    model_records = list()
    result_records = list()
    datasets = list()

    print('Loading data: {}'.format(time.time() - local_start_time))

    for i in range(num_of_data_pipelines):
        if base_dataset is not None:
            datasets.append(base_dataset.get_copy())
        else:
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
        num_of_transformations = random.randint(min_num_of_transformations, max_num_of_transformations)
        for i in range(num_of_transformations):
            d.apply_n_random_transformations(1)
            # if n > min_num_of_transformations and random.random() < prob_of_ending_transformations:
            #     break
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
                   'strategy': 'run_random_pipelines',
                   'timestamp':time.time()-start_time}
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
    start_time = time.time()
    run_data_pipeline_q_value_predictions_simple(r'/home/td/Documents/datasets/auto_ml/stackoverflow_survey_clean.csv',
                                        'ConvertedSalary',
                                        'regression',
                                        output_path = r'/home/td/Documents/datasets/auto_ml/training_results',
                                        run_id = 'q_value_predictions_simple',
                                        num_of_random_iterations = 100,
                                        num_of_data_pipelines = 10,
                                        transformations_to_consider_per_step = 100,
                                        temp_dataset_path = '/tmp/meta_model_dataset.csv',
                                        evaluation_metric = 'mean_squared_error',
                                        min_num_of_transformations = 32,
                                        max_num_of_transformations = 32
                                        )

    # for i in range(1000):
    #     run_random_pipelines(r'/home/td/Documents/datasets/auto_ml/stackoverflow_survey_clean.csv',
    #                          'ConvertedSalary',
    #                          'regression',
    #                          num_of_data_pipelines=10,
    #                          num_of_model_parameters=10,
    #                          min_num_of_transformations = 8,
    #                          max_num_of_transformations = 64,
    #                          run_id = 'random_pipelines',
    #                          output_path = r'/home/td/Documents/datasets/auto_ml/training_results',
    #                          start_time = start_time)
