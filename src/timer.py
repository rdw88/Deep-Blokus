from functools import wraps
from timeit import default_timer

import statistics


function_timers = dict()


def timer(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        start = default_timer()
        result = function(*args, **kwargs)
        end = default_timer()

        total_time = end - start

        global function_timers
        if function.__name__ not in function_timers:
            function_timers[function.__name__] = list()

        function_timers[function.__name__].append(total_time)

        return result

        #print(f'Execution time ({function.__name__}): {total_time} seconds')

    return decorator


def print_results():
    global function_timers

    print('\nExecution Metrics:')

    for function in function_timers.keys():
        print(f'{function}():')
        print(f'Called {len(function_timers[function])} times')
        print(f'min: {min(function_timers[function])} seconds')
        print(f'median: {statistics.median(function_timers[function])} seconds')
        print(f'mean: {statistics.mean(function_timers[function])} seconds')
        print(f'max: {max(function_timers[function])} seconds')
        print(f'total: {sum(function_timers[function])} seconds')