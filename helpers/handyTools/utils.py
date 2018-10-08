import sys
import requests
import os
import functools
import time


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    clear()
    sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def download_file(url, filename):
    req = requests.get(url)
    with open(filename, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            f.write(chunk)


def timeit(f):
    """Prints the time it takes to compute a function"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        print('Executed {0!r} in {1:4f} s'.format(f.__name__, time.time() - t0))
        return result
    return wrapper
