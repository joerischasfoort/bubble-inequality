from objects.trader import *
from objects.orderbook import *
import pandas as pd
import numpy as np
from functions.helpers import calculate_covariance_matrix


def init_objects_distr(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    max_horizon = parameters['horizon'] * 2  # this is the max horizon of an agent if 100% fundamentalist
    historical_stock_returns = np.random.normal(0, parameters["std_fundamental"], max_horizon)

    for idx in range(n_traders):
        weight_fundamentalist = abs(np.random.laplace(parameters['w_fundamentalists'], parameters['w_fundamentalists']**2))
        weight_chartist = abs(np.random.laplace(parameters['w_momentum'], parameters['w_momentum']**2))
        weight_random = abs(np.random.laplace(parameters['w_random'], parameters['w_random']**2))
        forecast_adjust = 1. / (weight_fundamentalist + weight_chartist + weight_random)

        init_stocks = int(np.random.uniform(0, parameters["init_stocks"]))
        init_money = np.random.uniform(0, (parameters["init_stocks"] * parameters['fundamental_value']))

        # initialize co_variance_matrix
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["std_fundamental"])

        lft_vars = TraderVariablesDistribution(weight_fundamentalist, weight_chartist, weight_random, forecast_adjust,
                                               init_money, init_stocks, init_covariance_matrix,
                                               parameters['fundamental_value'])

        # determine heterogeneous horizon and risk aversion based on
        individual_horizon = np.random.randint(10, parameters['horizon'])

        individual_risk_aversion = abs(np.random.normal(parameters["base_risk_aversion"], parameters["base_risk_aversion"] / 5.0))#parameters["base_risk_aversion"] * relative_fundamentalism
        individual_learning_ability = np.random.uniform(high=parameters["mutation_probability"]) #TODO update to be more elegant

        lft_params = TraderParametersDistribution(individual_horizon, individual_risk_aversion,
                                                  individual_learning_ability, parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["std_fundamental"],
                               max_horizon,
                               parameters['ticks'])

    # initialize order-book returns for initial variance calculations
    orderbook.returns = list(historical_stock_returns)

    return traders, orderbook