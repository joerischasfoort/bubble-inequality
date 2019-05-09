from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import json
import numpy as np
import math
from functions.find_bubbles import *

np.seterr(all='ignore')

start_time = time.time()

# INPUT PARAMETERS
LATIN_NUMBER = 0
NRUNS = 4
BURN_IN = 250
CORES = NRUNS # set the amount of cores equal to the amount of runs

problem = {
  'num_vars': 7,
  'names': ['std_noise',
            'w_random', 'strat_share_chartists',
            'base_risk_aversion',
            "fundamentalist_horizon_multiplier",
            "mutation_intensity",
            "average_learning_ability"],
  'bounds': [[0.05, 0.20],
             [0.02, 0.20], [0.20, 0.90],
             [0.1, 14.0],
             [0.1, 1.0], [0.05, 0.7],
             [0.1, 1.0]]
}

with open('hypercube.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]

init_parameters = latin_hyper_cube[LATIN_NUMBER]

params = {"ticks": 800 + BURN_IN, "fundamental_value": 166, 'n_traders': 500, 'std_fundamental': 0.0530163128919286,
          'spread_max': 0.004087, "init_stocks": 50, 'trader_sample_size': 19,
          'horizon': 100, "trades_per_tick": 3}


def simulate_a_seed(seed_params):
    """Simulates the model for a single seed and outputs the associated cost"""
    seed = seed_params[0]
    params = seed_params[1]

    obs = []
    # run model with parameters
    traders, orderbook = init_objects_distr(params, seed)
    traders, orderbook = pb_distr_model(traders, orderbook, params, seed)
    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs, burn_in_period=BURN_IN)

    rets_skew = []
    rets_autocor = []
    rets_autocor_abs = []
    rets_kurt = []

    pd_rets_skew = []
    pd_rets_autocor = []
    pd_rets_autocor_abs = []
    pd_rets_kurt = []

    for idx, col in enumerate(mc_returns):
        rets_skew.append(pd.Series(mc_returns[col][1:]).skew())
        rets_autocor.append(autocorrelation_returns(mc_returns[col][1:], 25))
        rets_autocor_abs.append(autocorrelation_abs_returns(mc_returns[col][1:], 25))
        rets_kurt.append(kurtosis(pd.Series(mc_returns[col][1:])))

        sim_pd_rets = pd.Series(
            list(np.array([obs[idx].tick_close_price[1:]]) / np.array([obs[idx].fundamental[:]])[0][0])[
                0]).pct_change()[1 + BURN_IN:]
        pd_rets_skew.append(sim_pd_rets.skew())
        pd_rets_autocor.append(autocorrelation_returns(sim_pd_rets, 25))
        pd_rets_autocor_abs.append(autocorrelation_abs_returns(sim_pd_rets, 25))
        pd_rets_kurt.append(kurtosis(pd.Series(sim_pd_rets)))

    stylized_facts_sim = np.array([np.mean(rets_skew),
                                   np.mean(rets_autocor),
                                   np.mean(rets_autocor_abs),
                                   np.mean(rets_kurt),
                                   np.mean(pd_rets_skew),
                                   np.mean(pd_rets_autocor),
                                   np.mean(pd_rets_autocor_abs),
                                   np.mean(pd_rets_kurt)
                                   ])

    W = np.load('distr_weighting_matrix.npy')  # if this doesn't work, use: np.identity(len(stylized_facts_sim))

    empirical_moments = np.array([0.03613833, 0.00952201, 0.05360488, 2.67789658,
                                  0.03619233, 0.0075641 , 0.06170896, 2.64338843])

    # calculate the cost
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    return cost


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    def model_performance(input_parameters):
        """
        Simple function calibrate uncertain model parameters
        :param input_parameters: list of input parameters
        :return: average cost
        """
        # convert relevant parameters to integers
        new_input_params = []
        for idx, par in enumerate(input_parameters):
            new_input_params.append(par)

        # update params
        uncertain_parameters = dict(zip(problem['names'], new_input_params))
        params = {"ticks": 800 + BURN_IN, "fundamental_value": 166, 'n_traders': 500,
                  'std_fundamental': 0.0530163128919286, 'spread_max': 0.004087,
                  "init_stocks": 50, 'trader_sample_size': 19,
                  'horizon': 100, "trades_per_tick": 3}
        params.update(uncertain_parameters)

        list_of_seeds_params = [[seed, params] for seed in list_of_seeds]

        costs = p.map(simulate_a_seed, list_of_seeds_params) # first argument is function to execute, second argument is tuple of all inputs TODO uncomment this

        return np.mean(costs)

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=1, full_output=True)

    with open('estimated_params.json', 'w') as f:
        json.dump(list(output['xopt']), f)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
