import pandas as pd
import numpy as np
import json
from functions.find_bubbles import *
from functions.stylizedfacts import *
from functions.helpers import hypothetical_series


BOOTSTRAPS = 480
shiller_block_size = 75

# Collect data from website of Shiller
shiller_data = pd.read_excel('http://www.econ.yale.edu/~shiller/data/ie_data.xls', header=7)[:-3]
p = pd.Series(np.array(shiller_data.iloc[1174:-1]['Price']))
y = pd.Series(np.array(shiller_data.iloc[1174:-1]['CAPE']))  

# set parameters for bubble detection algorithm 
obs = len(y)
r0 = 0.01 + 1.8/np.sqrt(obs)
swindow0 = int(math.floor(r0*obs))
dim = obs - swindow0 + 1
IC = 2
adflag = 6
yr = 2
Tb = 12*yr + swindow0 - 1
nboot = 199

bsadfs = PSY(y, swindow0, IC, adflag)
quantilesBsadf = cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot)

monitorDates = y.iloc[swindow0-1:obs].index
quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
ind95 =(bsadfs.T[0] > quantile95[1, ])
periods = monitorDates[ind95]

bubbly_dates = find_sequences_ints(periods, monitorDates)

p_returns = pd.Series(np.array(shiller_data.iloc[1174:]['Price'])).pct_change()[1:]
pd_returns = pd.Series(np.array(shiller_data.iloc[1174:]['CAPE'])).pct_change()[1:]

# Formulate empirical moments and dump into json file
percentage_bubbles = len(periods) / float(len(monitorDates))
lenghts_of_bubbles = []
for row in range(len(bubbly_dates)):
    lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)
stdev_lenght_bubbles = np.std(lenghts_of_bubbles)
av_lenght_bubbles = np.mean(lenghts_of_bubbles)

emp_moments = np.array([
    autocorrelation_returns(p_returns, 25),
    autocorrelation_abs_returns(p_returns, 25),
    kurtosis(p_returns),
    percentage_bubbles,
    av_lenght_bubbles,
    stdev_lenght_bubbles,
    pd.Series(lenghts_of_bubbles).skew(),
    pd.Series(lenghts_of_bubbles).kurtosis()
    ])
	
with open('emp_moments.json', 'w') as fp:
    json.dump(list(emp_moments), fp)

# divide the price to dividend returns into blocks
pd_data_blocks = []
p_data_blocks = []
for x in range(0, len(pd_returns[:-3]), shiller_block_size):
    pd_data_blocks.append(pd_returns[x:x+shiller_block_size])
    p_data_blocks.append(p_returns[x:x+shiller_block_size])

# draw BOOTSTRAPS random series 
bootstrapped_pd_series = []
bootstrapped_p_returns = []
for i in range(BOOTSTRAPS):
    # bootstrap p returns
    sim_data_p = [random.choice(p_data_blocks) for _ in p_data_blocks]
    sim_data2_p = [j for i in sim_data_p for j in i]
    bootstrapped_p_returns.append(sim_data2_p)
    
    # first sample the data for pd returns
    sim_data = [random.choice(pd_data_blocks) for _ in pd_data_blocks] # choose a random set of blocks
    sim_data_fundamental_returns = [pair for pair in sim_data]
    
    # merge the list of lists
    sim_data_fundamental_returns1 = [item for sublist in sim_data_fundamental_returns for item in sublist]
    
    # calculate the new time_series
    sim_data_fundamentals = hypothetical_series(y[0], sim_data_fundamental_returns1[1:])
    bootstrapped_pd_series.append(sim_data_fundamentals) # used to also be price returns.. perhaps re-implement
	
# dump bootstrapped price to dividend ratio and returns in json files

with open('boostr_pd_rets.json', 'w') as fp:
    json.dump(bootstrapped_pd_series, fp)
	
with open('boostr_p_rets.json', 'w') as fp:
    json.dump(bootstrapped_p_returns, fp)