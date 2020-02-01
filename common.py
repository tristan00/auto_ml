import random
import re
from sklearn import metrics
import traceback

pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel_neg = -999999999
numerical_sentinel_pos = -999999999

string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'
path = '/home/td/Documents/datasets/auto_ml'


def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def pick_parameter(options, selection_type):
    if selection_type == 'int_range':
        return random.randint(options[0], options[1])
    if selection_type == 'float_range':
        return random.random() * (options[1] - options[0])
    if selection_type == 'choice':
        return random.choice(options)

def get_metrics(truth, preds, problem_type):
    if problem_type == 'classification':

        try:
            log_loss = metrics.log_loss(truth, preds)
        except:
            log_loss = None
        try:
            accuracy = metrics.accuracy_score(truth, preds)
        except:
            accuracy = None

        return {'log_loss':log_loss,
                'accuracy':accuracy}

    elif problem_type == 'regression':

        try:
            mae = metrics.mean_absolute_error(truth, preds)

        except:
            mae = None
            # traceback.print_exc()

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