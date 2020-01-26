import random
import re

pd_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_sentinel_neg = -999999999
numerical_sentinel_pos = -999999999

string_sentinel = '-999999999'
other_category_filler = 'other_category_filler'


def clean_text(s):
    return re.sub(r'\W+', '_', str(s))


def pick_parameter(options, selection_type):
    if selection_type == 'int_range':
        return random.randint(options[0], options[1])
    if selection_type == 'float_range':
        return random.random() * (options[1] - options[0])
    if selection_type == 'choice':
        return random.choice(options)
