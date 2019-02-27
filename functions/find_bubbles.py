import numpy as np
import pandas as pd
import math
import json
from distribution_model import *
from init_objects import *
from functions.helpers import organise_data
from functions.inequality import gini


def ADF(y, IC=0, adflag=0):
    """
    Calculates the augmented Dickey-Fuller (ADF) test statistic with lag order set fixed or selected by AIC or BIC

    Port from github psymonitor
    Credits to: Phillips, P C B, Shi, S, & Yu, J
    Testing for multiple bubbles: Historical episodes of exuberance and collapse in the SP 500
    International Economic Review 2015

    :param y: list data
    :param IC:
    :param adflag:
    :return: float ADF test statistic
    """
    T0 = len(y)
    T1 = len(y) - 1
    const = np.ones(T1)

    y1 = np.array(y[0:T1])
    y0 = np.array(y[1:T0])
    dy = y0 - y1
    x1 = np.c_[y1, const]

    t = T1 - adflag
    dof = t - adflag - 2

    if IC > 0:
        ICC = np.zeros([adflag + 1, 1])
        ADF = np.zeros([adflag + 1, 1])
        for k in range(adflag + 1):
            dy01 = dy[k:T1, ]
            x2 = np.zeros([T1 - k, k])

            for j in range(k):
                x2[:, j] = dy[k - j - 1:T1 - j - 1]

            x2 = np.concatenate((x1[k:T1, ], x2,), axis=1)

            # OLS regression
            beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
            eps = dy01 - np.dot(x2, beta)
            # Information criteria
            npdf = sum(-1 / 2.0 * np.log(2 * np.pi) - 1 / 2.0 * (eps ** 2))
            if IC == 1:
                ICC[k, ] = -2 * npdf / float(t) + 2 * len(beta) / float(t)
            elif IC == 2:
                ICC[k, ] = -2 * npdf / float(t) + len(beta) * np.log(t) / float(t)
            se = np.dot(eps.T, eps / dof)
            sig = np.sqrt(np.diag((np.ones([len(beta), len(beta)]) * se) * np.linalg.solve(np.dot(x2.T, x2),
                                                                                           np.identity(
                                                                                               len(np.dot(x2.T, x2))))))
            ADF[k,] = beta[0,] / sig[0]
        lag0 = np.argmin(ICC)
        ADFlag = ADF[lag0,][0]  # TODO check if this is correct
    elif IC == 0:
        # Model Specification
        dy01 = dy[adflag:T1, ]
        x2 = np.zeros([t, adflag])

        for j in range(adflag):
            x2[:, j] = dy[adflag - j - 1:T1 - j - 1]

        x2 = np.concatenate((x1[adflag:T1, ], x2,), axis=1)

        # OLS regression
        beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
        eps = dy01 - np.dot(x2, beta)
        se = np.dot(eps.T, eps / dof)
        sig = np.sqrt(np.diag((np.ones([len(beta), len(beta)]) * se) * np.linalg.solve(np.dot(x2.T, x2), np.identity(
            len(np.dot(x2.T, x2))))))

        ADFlag = beta[0,] / sig[0]  # check if this is correct

    if IC == 0:
        result = ['fixed lag of order 1', ADFlag]

    if IC == 1:
        result = ['ADF Statistic using AIC', ADFlag]

    if IC == 2:
        result = ['ADF Statistic using BIC', ADFlag]

    return result[1]


def PSY(y, swindow0, IC, adflag):
    """
    Estimate PSY's BSADF sequence of test statistics
    implements the real time bubble detection procedure of Phillips, Shi and Yu (2015a,b)

    param: y: np.array of the data
    param: swindow0: integer minimum window size
    param: adflag: An integer, lag order when IC=0; maximum number of lags when IC>0 (default = 0).

    For every period in time calculate the max ADF statistic using a rolling window.

    return: list BSADF test statistic.
    """
    t = len(y)

    if not swindow0:
        swindow0 = int(math.floor(t * (0.01 + 1.8 / np.sqrt(t))))

    bsadfs = np.zeros([t, 1])  # create empty column array at lenght of the data (zeros)

    for r2 in range(swindow0, t + 1):
        # loop over the range 47 - 647
        # create column vector of increasing lenght and fill with - 999
        rwadft = np.ones([r2 - swindow0 + 1, 1]) * -999
        for r1 in range(r2 - swindow0 + 1):
            # loop over the range 0 - 500
            # perform ADF test on data from r1 --> r2
            # insert in row
            rwadft[r1] = float(ADF(y.iloc[r1:r2], IC, adflag))

        # take max value an insert in bsadfs array
        bsadfs[r2 - 1] = max(rwadft.T[0])

    # create shortened version of array
    bsadf = bsadfs[swindow0-1 : t]

    return bsadf


def ADFres(y, IC=0, adflag=0):
    """"""
    T0 = len(y)
    T1 = len(y) - 1

    y1 = np.array(y[0:T1])
    y0 = np.array(y[1:T0])
    dy = y0 - y1
    t = T1 - adflag

    if IC > 0:
        ICC = np.zeros([adflag + 1, 1])
        betaM = []  # np.zeros([adflag + 1, 1])
        epsM = []  # np.zeros([adflag + 1, 1])
        for k in range(adflag + 1):
            dy01 = dy[k:T1, ]
            x_temp = np.zeros([T1 - k, k])

            for j in range(k):
                x_temp[:, j] = dy[k - j - 1:T1 - j - 1]

            x2 = np.ones([T1 - k, k + 1])
            x2[:, 1:] = x_temp

            # OLS regression time
            betaM.append(
                np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01)))
            epsM.append(dy01 - np.dot(x2, betaM[k]))

            # Information criteria
            npdf = sum(-1 / 2.0 * np.log(2 * np.pi) - 1 / 2.0 * (epsM[k] ** 2))
            if IC == 1:
                ICC[k] = -2 * npdf / float(t) + 2 * len(betaM[k]) / float(t)
            elif IC == 2:
                ICC[k] = -2 * npdf / float(t) + len(betaM[k]) * np.log(t) / float(t)

        lag0 = np.argmin(ICC)
        beta = betaM[lag0]
        eps = epsM[lag0]
        lag = lag0

    elif IC == 0:
        dy01 = dy[adflag:T1, ]
        x_temp = np.zeros([t, adflag])

        for j in range(adflag):
            x_temp[:, j] = dy[adflag - j - 1:T1 - j - 1]

        x2 = np.ones([T1 - adflag, adflag + 1]) #TODO debug this line
        x2[:, 1:] = x_temp

        # OLS regression
        beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
        eps = dy01 - np.dot(x2, beta)
        lag = adflag

    else:
        beta, eps, lag = None

    return beta, eps, lag


def cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot=199):
    """
    Computes a matrix of 90, 95 and 99 critical values which can be used to compare to the bsadf statistics.
    param: y: data array or pandas df
    param: swindow0: integer minimum window size
    param: adflag: An integer, lag order when IC=0; maximum number of lags when IC>0 (default = 0).
    param: control_sample_size: Integer the simulated sample size
    param: nboot: positive integer. Number of bootstrap replications (default = 199).
    param: nCores = integernumber of cores (supports multithreading
    return: A matrix. BSADF bootstrap critical value sequence at the 90, 95 and 99 percent level.
    """
    qe = np.array([0.90, 0.95, 0.99])

    beta, eps, lag = ADFres(y, IC, adflag)

    T0 = len(eps)
    t = len(y)
    dy = np.array(y.iloc[0:(t - 1)]) - np.array(y.iloc[1:t])
    g = len(beta)

    # create matrix filled with random ints < T0 with rows TB and cols: nboot
    rN = np.random.randint(0, T0, (Tb, nboot))
    # create weigth matrix filled a random normal float
    wn = np.random.normal(0) * np.ones([Tb, nboot])

    # dyb = 69 row, 99 col matrix of zeros
    dyb = np.zeros([Tb - 1, nboot])
    # fill first 6 rows with first six
    dyb[:lag + 1, ] = np.split(np.tile(dy[[l for l in range(lag + 1)]], nboot), lag + 1, axis=0)

    for j in range(nboot):
        # loop over all columns
        if lag == 0:
            for i in range(lag, Tb - 1):
                # loop over rows, start filling the rest of the dyb rows with random numbers
                dyb[i, j] = wn[i - lag, j] * eps[rN[i - lag, j]]
        elif lag > 0:
            x = np.zeros([Tb - 1, lag])
            for i in range(lag, Tb - 1):
                # create a new empy array of simlar proportions to dyb
                x = np.zeros([Tb - 1, lag])
                for k in range(lag):
                    # every row after the first 6, fill the first six column values with
                    # values of the dyb six rows that came before it
                    x[i, k] = dyb[i - k, j]

                # matrix multiplication
                # fill the rows below the first 6 with
                # the i row of x *
                dyb[i, j] = np.dot(x[i,], beta[1:g]) + wn[i - lag, j] * eps[rN[i - lag, j]]

    dyb0 = np.ones([Tb, nboot]) * y[1]
    dyb0[1:, :] = dyb
    yb = np.cumsum(dyb0, axis=0)

    dim = Tb - swindow0 + 1
    i = 0

    # for every .. column perform PSY, since there are 99 columns...
    # this gives a new matrix with 24 columns and 4 rows... so every row= ser
    MPSY = []
    for col in range(nboot):
        MPSY.append(PSY(pd.Series(yb[:, col]), swindow0, IC, adflag))

    MPSY = np.array(MPSY)
    # then, find the max value for each point in time?
    SPSY = MPSY.max(axis=1)
    # then, find the quantile for each
    Q_SPSY = pd.Series(SPSY.T[0]).quantile(qe)

    return Q_SPSY


def is_end_date(value, next_value):
    """determine if this is the end date of a time series"""
    if value != next_value - 1:
        return True
    else:
        return False


def is_start_date(value, previous_value):
    """determine if this is the start date of a time series."""
    if value != previous_value + 1:
        return True
    else:
        return False


def find_sequences_datetime(p, md):
    """
    Transform bubble occurence time sequences to a series of sequences
    :param p: list of periods with bubbles in date string format
    :param md: list of all dates of interest
    :return: Dataframe with start and end dates of a bubble.
    """
    all_dates = pd.to_datetime(md)

    locs = []
    for idx, date in enumerate(p):
        locs.append((all_dates == date).argmax())

    end_dates = [is_end_date(value, next_value) for value, next_value in zip(locs[:-1], locs[1:])] + [True]
    start_dates = [True] + [is_start_date(value, previous_value) for value, previous_value in zip(locs[1:], locs[:-1])]

    end_locs = np.array(locs)[np.array(end_dates)]
    start_locs = np.array(locs)[np.array(start_dates)]

    return pd.DataFrame({'end_date': all_dates[(end_locs)], 'start_date': all_dates[(start_locs)]})[
        ['start_date', 'end_date']]


def find_sequences_ints(p, md):
    """
    Transform bubble occurence time sequences of ints to a series of sequences
    :param p: list of periods with bubbles in date string format
    :param md: list of all dates of interest
    :return: Dataframe with start and end dates of a bubble.
    """
    locs = []
    for date in p:
        locs.append((md == date).argmax())

    end_dates = [is_end_date(value, next_value) for value, next_value in zip(locs[:-1], locs[1:])] + [True]
    start_dates = [True] + [is_start_date(value, previous_value) for value, previous_value in zip(locs[1:], locs[:-1])]

    end_locs = np.array(locs)[np.array(end_dates)]
    start_locs = np.array(locs)[np.array(start_dates)]

    return pd.DataFrame({'end_date': md[(end_locs)], 'start_date': md[(start_locs)]})[
        ['start_date', 'end_date']]


def bubble_period(all_dates, bubbly_date_serie):
    """
    Return all dates of a single bubble period given a start, end date and the full time series
    :param all_dates: list of all dates
    :param bubbly_date_serie: list of dates in which there was a bubble.
    :return:
    """
    first_date = (all_dates == bubbly_date_serie['start_date']).argmax()
    second_date = (all_dates == bubbly_date_serie['end_date']).argmax()

    if first_date == second_date:
        return all_dates[first_date-2:first_date]
    else:
        return all_dates[first_date:second_date + 1]


def p_bubbles(bubbly_dates):
    lenghts_of_bubbles = []
    for row in range(len(bubbly_dates)):
        lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)
    lenghts_of_bubbles = np.array(lenghts_of_bubbles)
    av_lenghts_of_bubbles = np.mean(lenghts_of_bubbles)
    long_bubble_condition = lenghts_of_bubbles > av_lenghts_of_bubbles
    r = np.array(range(len(long_bubble_condition)))
    locs_long_bubbles = r[long_bubble_condition]
    return locs_long_bubbles


def sim_bubble_info(seed):
    """
    Simulate model once and return accompanying info on
    Inequality:
    - bubble_type
    - bubble-episode price
    - wealth_start
    - wealth_end
    + wealth_gini_over_time
    + palma_over_time
    + twentytwenty_over_time

    Information on agent characteristics
    - risk aversion
    - horizon
    - learning ability
    - chartist expectation
    - fundamentalist expectation
    """
    BURN_IN = 200
    with open('parameters.json', 'r') as f:
        params = json.loads(f.read())
    # simulate model once

    obs = []
    # run model with parameters
    print('Simulate once')
    traders, orderbook = init_objects_distr(params, seed)
    traders, orderbook = pb_distr_model(traders, orderbook, params, seed)

    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs, burn_in_period=BURN_IN)

    y = pd.Series(mc_prices[0][:-1] / mc_fundamentals[0])

    obs = len(y)
    r0 = 0.01 + 1.8 / np.sqrt(obs)
    swindow0 = int(math.floor(r0 * obs))
    dim = obs - swindow0 + 1
    IC = 2
    adflag = 6
    yr = 2
    Tb = 12 * yr + swindow0 - 1
    nboot = 99

    # calc bubbles
    bsadfs = PSY(y, swindow0, IC, adflag)

    quantilesBsadf = cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot=nboot)

    monitorDates = y.iloc[swindow0 - 1:obs].index
    quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
    ind95 = (bsadfs.T[0] > quantile95[1,])
    periods = monitorDates[ind95]

    bubble_types = []
    bubble_prices = []
    wealth_starts = []
    wealth_ends = []
    ginis_ot = []
    palmas_ot = []
    twtws_ot = []
    risk_aversions = []
    horizons = []
    learning_abilities = []
    chartist_expectations = []
    fundamentalist_expectations = []

    if True in ind95:
        bubbly_dates = find_sequences_ints(periods, monitorDates)
        proper_bubbles = bubbly_dates.iloc[p_bubbles(bubbly_dates)]

        # classify the bubbles
        start_dates = []
        end_dates = []

        # add bubble episodes
        for l in range(len(proper_bubbles)):
            start_dates.append(proper_bubbles.iloc[l]['start_date'])
            end_dates.append(proper_bubbles.iloc[l]['end_date'])

            if abs(y[end_dates[l]] - y[start_dates[l]]) > y[:end_dates[l]].std():
                # classify as boom or bust
                if y[start_dates[l]] > y[end_dates[l]]:
                    bubble_type = 'bust'
                else:
                    bubble_type = 'boom'
            else:
                if y[start_dates[l]:end_dates[l]].mean() > y[start_dates[l]]:
                    # classify as boom-bust or bust-boom
                    bubble_type = 'boom-bust'
                else:
                    bubble_type = 'bust-boom'
            bubble_types.append(bubble_type)

            # determine the start and end wealth of the bubble
            money_start = np.array([x.var.money[BURN_IN + start_dates[l]] for x in traders])
            stocks_start = np.array([x.var.stocks[BURN_IN + start_dates[l]] for x in traders])
            wealth_start = money_start + (stocks_start * mc_prices[0].iloc[start_dates[l]])

            money_end = np.array([x.var.money[BURN_IN + end_dates[l]] for x in traders])
            stocks_end = np.array([x.var.stocks[BURN_IN + end_dates[l]] for x in traders])
            wealth_end = money_end + (stocks_end * mc_prices[0].iloc[end_dates[l]])

            # determine characteristics of the agents
            risk_aversions.append([x.par.risk_aversion for x in traders])
            horizons.append([x.par.horizon for x in traders])
            learning_abilities.append([x.par.learning_ability for x in traders])
            chartist_expectations.append([x.var.weight_chartist[BURN_IN + start_dates[l]: BURN_IN + end_dates[l]] for x in traders])
            fundamentalist_expectations.append([x.var.weight_fundamentalist[BURN_IN + start_dates[l]: BURN_IN + end_dates[l]] for x in
                                         traders])

            wealth_gini_over_time = []
            palma_over_time = []
            twentytwenty_over_time = []
            for t in range(BURN_IN + start_dates[l], BURN_IN + end_dates[l]):
                money = np.array([x.var.money[t] for x in traders])
                stocks = np.array([x.var.stocks[t] for x in traders])
                wealth = money + (stocks * orderbook.tick_close_price[t])

                share_top_10 = sum(np.sort(wealth)[int(len(wealth) * 0.9):]) / sum(wealth)
                share_bottom_40 = sum(np.sort(wealth)[:int(len(wealth) * 0.4)]) / sum(wealth)
                palma_over_time.append(share_top_10 / share_bottom_40)

                share_top_20 = np.mean(np.sort(wealth)[int(len(wealth) * 0.8):])
                share_bottom_20 = np.mean(np.sort(wealth)[:int(len(wealth) * 0.2)])
                twentytwenty_over_time.append(share_top_20 / share_bottom_20)

                wealth_gini_over_time.append(gini(wealth))

            bubble_prices.append(list(mc_prices[0].iloc[start_dates[l]: end_dates[l]]))
            wealth_starts.append(list(wealth_start))
            wealth_ends.append(list(wealth_end))
            ginis_ot.append(wealth_gini_over_time)
            palmas_ot.append(palma_over_time)
            twtws_ot.append(twentytwenty_over_time)

    return bubble_types, bubble_prices, wealth_starts, wealth_ends, ginis_ot, palmas_ot, twtws_ot, risk_aversions, horizons, learning_abilities, chartist_expectations, fundamentalist_expectations


def sim_synthetic_bubble(seed):
    """
    Simulate model once with a shock and return accompanying info on
    - bubble_type
    - bubble-episode price
    - wealth_start
    - wealth_end
    + wealth_gini_over_time
    + palma_over_time
    + twentytwenty_over_time
    """
    BURN_IN = 200
    SHOCK = 12000.0
    SHOCK_PERIOD = 400

    params = {"spread_max": 0.004087, "fundamental_value": 166, "fundamentalist_horizon_multiplier": 0.73132061,
              "n_traders": 500, "w_fundamentalists": 37.20189844, "base_risk_aversion": 11.65898537,
              "mutation_probability": 0.30623129, "init_stocks": 50, "trader_sample_size": 19, "ticks": 700,
              "std_fundamental": 0.0530163128919286, "std_noise": 0.29985649, "trades_per_tick": 5,
              "average_learning_ability": 0.57451773, "w_momentum": 0.01, "horizon": 200, "w_random": 1.0}

    # simulate model once
    obs = []
    obs_no_shock = []
    # run model with parameters
    print('Simulate once')
    traders, orderbook = init_objects_distr(params, seed)
    traders, orderbook = pb_distr_model_shock(traders, orderbook, params, SHOCK, SHOCK_PERIOD, seed)
    obs.append(orderbook)

    traders_no_shock, orderbook_no_shock = init_objects_distr(params, seed)
    traders_no_shock, orderbook_no_shock = pb_distr_model_shock(traders_no_shock, orderbook_no_shock, params, 0.0, SHOCK_PERIOD, seed)
    obs_no_shock.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs, burn_in_period=BURN_IN)

    mc_prices_ns, mc_returns_ns, mc_autocorr_returns_ns, mc_autocorr_abs_returns_ns, mc_volatility_ns, mc_volume_ns, mc_fundamentals_ns = organise_data(
        obs_no_shock, burn_in_period=BURN_IN)

    y = pd.Series(mc_prices[0][:-1] / mc_fundamentals[0])
    y_ns = pd.Series(mc_prices_ns[0][:-1] / mc_fundamentals_ns[0])

    obs = len(y)
    r0 = 0.01 + 1.8 / np.sqrt(obs)
    swindow0 = int(math.floor(r0 * obs))
    dim = obs - swindow0 + 1
    IC = 2
    adflag = 6
    yr = 2
    Tb = 12 * yr + swindow0 - 1
    nboot = 99

    # calc bubbles
    bsadfs = PSY(y, swindow0, IC, adflag)

    quantilesBsadf = cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot=99)

    monitorDates = y.iloc[swindow0 - 1:obs].index
    quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
    ind95 = (bsadfs.T[0] > quantile95[1,])
    periods = monitorDates[ind95]

    bubble_types = []
    bubble_prices = []
    wealth_starts = []
    wealth_ends = []
    ginis_ot = []
    palmas_ot = []
    twtws_ot = []

    bubble_prices_ns = []
    wealth_ends_ns = []
    ginis_ot_ns = []
    palmas_ot_ns = []
    twtws_ot_ns = []

    if True in ind95:
        bubbly_dates = find_sequences_ints(periods, monitorDates)
        proper_bubbles = bubbly_dates.iloc[p_bubbles(bubbly_dates)]

        # classify the bubbles
        start_dates = []
        end_dates = []

        # then add the first bubble episodes
        start_dates.append(proper_bubbles.iloc[0]['start_date'])
        end_dates.append(proper_bubbles.iloc[0]['end_date'])

        if abs(y[end_dates[0]] - y[start_dates[0]]) > y[:end_dates[0]].std():
            # classify as boom or bust
            if y[start_dates[0]] > y[end_dates[0]]:
                bubble_type = 'bust'
            else:
                bubble_type = 'boom'
        else:
            if y[start_dates[0]:end_dates[0]].mean() > y[start_dates[0]]:
                # classify as boom-bust or bust-boom
                bubble_type = 'boom-bust'
            else:
                bubble_type = 'bust-boom'
        bubble_types.append(bubble_type)

        # determine the start and end wealth of the bubble
        money_start = np.array([x.var.money[BURN_IN + start_dates[0]] for x in traders])
        stocks_start = np.array([x.var.stocks[BURN_IN + start_dates[0]] for x in traders])
        wealth_start = money_start + (stocks_start * mc_prices[0].iloc[start_dates[0]])

        money_end = np.array([x.var.money[BURN_IN + end_dates[0]] for x in traders])
        stocks_end = np.array([x.var.stocks[BURN_IN + end_dates[0]] for x in traders])
        wealth_end = money_end + (stocks_end * mc_prices[0].iloc[end_dates[0]])

        # track money + wealth no shocks  TODO check if this works
        money_end_ns = np.array([x.var.money[BURN_IN + end_dates[0]] for x in traders_no_shock])
        stocks_end_ns = np.array([x.var.stocks[BURN_IN + end_dates[0]] for x in traders_no_shock])
        wealth_end_ns = money_end_ns + (stocks_end_ns * mc_prices_ns[0].iloc[end_dates[0]])

        wealth_gini_over_time = []
        palma_over_time = []
        twentytwenty_over_time = []

        # also record the gini etc. over time without the shock
        wealth_gini_over_time_ns = []
        palma_over_time_ns = []
        twentytwenty_over_time_ns = []
        for t in range(BURN_IN + start_dates[0], BURN_IN + end_dates[0]):
            money = np.array([x.var.money[t] for x in traders])
            stocks = np.array([x.var.stocks[t] for x in traders])
            wealth = money + (stocks * orderbook.tick_close_price[t])

            share_top_10 = sum(np.sort(wealth)[int(len(wealth) * 0.9):]) / sum(wealth)
            share_bottom_40 = sum(np.sort(wealth)[:int(len(wealth) * 0.4)]) / sum(wealth)
            palma_over_time.append(share_top_10 / share_bottom_40)

            share_top_20 = sum(np.sort(wealth)[int(len(wealth) * 0.8):]) / sum(wealth)
            share_bottom_20 = sum(np.sort(wealth)[:int(len(wealth) * 0.2)]) / sum(wealth)
            twentytwenty_over_time.append(share_top_20 / share_bottom_20)

            wealth_gini_over_time.append(gini(wealth))

            # No shocks
            money_ns = np.array([x.var.money[t] for x in traders_no_shock])
            stocks_ns = np.array([x.var.stocks[t] for x in traders_no_shock])
            wealth_ns = money_ns + (stocks_ns * orderbook_no_shock.tick_close_price[t])

            share_top_10_ns = sum(np.sort(wealth_ns)[int(len(wealth_ns) * 0.9):]) / sum(wealth_ns)
            share_bottom_40_ns = sum(np.sort(wealth_ns)[:int(len(wealth_ns) * 0.4)]) / sum(wealth_ns)
            palma_over_time_ns.append(share_top_10_ns / share_bottom_40_ns)

            share_top_20_ns = np.mean(np.sort(wealth_ns)[int(len(wealth_ns) * 0.8):])
            share_bottom_20_ns = np.mean(np.sort(wealth_ns)[:int(len(wealth_ns) * 0.2)])
            twentytwenty_over_time_ns.append(share_top_20_ns / share_bottom_20_ns)

            wealth_gini_over_time_ns.append(gini(wealth_ns))

        bubble_prices.append(list(mc_prices[0].iloc[start_dates[0]: end_dates[0]]))
        wealth_starts.append(list(wealth_start))
        wealth_ends.append(list(wealth_end))
        ginis_ot.append(wealth_gini_over_time)
        palmas_ot.append(palma_over_time)
        twtws_ot.append(twentytwenty_over_time)

        bubble_prices_ns.append(list(mc_prices_ns[0].iloc[start_dates[0]: end_dates[0]]))
        wealth_ends_ns.append(list(wealth_end_ns))
        ginis_ot_ns.append(wealth_gini_over_time_ns)
        palmas_ot_ns.append(palma_over_time_ns)
        twtws_ot_ns.append(twentytwenty_over_time_ns)

    return bubble_types, bubble_prices, wealth_starts, wealth_ends, ginis_ot, palmas_ot, twtws_ot, bubble_prices_ns, wealth_ends_ns, ginis_ot_ns, palmas_ot_ns, twtws_ot_ns