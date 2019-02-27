from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import numpy as np
import math
from functions.find_bubbles import *
import json # simplejson on clustercomp

np.seterr(all='ignore')

def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]


    output = p.map(sim_bubble_info, list_of_seeds)

    with open('all_many_bubbles_output.json', 'w') as fp:
        json.dump(output, fp) #simple_json on supercomp

    print('All outputs are: ', output)


if __name__ == '__main__':
    start_time = time.time()

    NRUNS = 16
    CORES = 4  # set the amount of cores equal to the amount of runs

    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
