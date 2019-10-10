import pandas as pd
import gym
import numpy as np
from sklearn import preprocessing
import enum
import random
from gym import spaces


scaler = preprocessing.StandardScaler()

class CryptoEnv(gym.Env):

    metadata = {'render.modes': ['human', 'system', 'none']}
    scaler = preprocessing.StandardScaler()
    viewer = None

    def __init__(self,  initial_balance, anomalies, arr, n_crypto, crypto_names, his_size, n_retards=8, add_sers = 0,comission = 0.005, buy_decrease=0.01):
        '''
        Docstring: This initializes our environment.
        Parameters:
            - comission: com percentage of the exchange
            - anomalies: list of tuples with the beginning and end of anomaly periods for splitting the dataset
            - arr: 3-d array with the data
            - n_crypto: number of cryptos to operate with
            - n_retards: number of retards considered
            - add_sers: number of additional series apart from the prices and volumes
            - his_size: NI PUTA IDEA DE POR QUÉ CREASTE ESTA VARIABLE HERMANO.
            - crypto_names: list containing the crypto names in the correct order. It's crucial that the order
            of cryptocurrencies is maintained during all the analysis, as this ensures that there's no misleading
            information. This is true for positions, values needed for reward calculation, and building of the
            positions dataframe.
        '''
        super(CryptoEnv, self).__init__()
        scaler = preprocessing.StandardScaler()

        self.comission=np.array(comission, dtype=np.float64)
        self.anomalies = anomalies
        self.anomalies_beginning = [a[0] for a in self.anomalies]
        self.anomalies_ending = [a[1] for a in self.anomalies]
        self.arr = arr
        self.n_crypto = n_crypto
        self.crypto_names = crypto_names
        self.n_retards = n_retards
        #self.add_sers=add_sers
        self.sh = self.arr.shape
        self.his_size = his_size
        self.scaled_arr = scaler.fit_transform(self.arr.reshape(self.arr.shape[0]*self.arr.shape[1], self.arr.shape[2])).reshape(self.sh)
        self.prices_positions = [i for i in range(0, 21, 3)]
        '''
        As we will add sentiments probably later, it makes more sense to set it to 21, as it is the number of columns we have without sentiments.
        This way we take columns 0, 3, 6, 9, 12, 15, 18, which correspond to the cryptocurrencies positions.
        '''
        self.initial_balance = initial_balance
        self.wallet_value = initial_balance
        self.cash = initial_balance
        self.inactive_counts=0
        '''
        Action Space Needs to be Well Defined!!!!!!!!!!!!
        '''
        self.action_space = spaces.MultiDiscrete(np.repeat([[3, 3]], self.n_crypto, axis=0))
        #self.action_space = spaces.MultiDiscrete(np.repeat([[-0.05, 0, 0.05]], self.n_crypto, axis = 0))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.sh[1], self.sh[2]),dtype=np.float64)
        self.dfPositions = pd.DataFrame(np.zeros((self.n_retards + 1, self.n_crypto)), columns = self.crypto_names)
        self.buy_decrease = buy_decrease
        '''
        This variable above is created so that in case we do not have enough cash to buy what we want, we can decrease
        the quantity we want to buy by this value (buy_decrease), so that we can fulfill the operation.
        '''
        self.quantities_held = np.array([0]*self.n_crypto)
        self.cache_ = {'initial_balance': initial_balance,
                        'trades':{},
                        'wallet_composition': {}}

    def _next_observation(self):
        '''
        This function returns the agent the next observation from the environment
        '''
        self.current_step += 1
        obs = self.active_arr_scaled[self.current_step, :, :]
        obs = np.append(obs, scaler.fit_transform(self.dfPositions.iloc[:self.n_retards+1, :].values), axis=1)
        return obs


    def _reset_session(self):
        '''
        Docstring: This function takes an element of the tuples representing anomalies,
        decides in which non-anomaly period it will start, and the end of the period
        is automatically set to the start of a new anomaly. So for example if from 4th may
        to 10th may data was lost, the agent could train up to 4th may since the beginning of
        the period, and then stop training. Next time, it may start on may 10th, and so on.
        Current positions is an array representing the quantity of each cryptocurrency we have
        in our wallet.
        '''
        self.current_step = 0
        self.frame_start = random.choice(self.anomalies_beginning)
        self.frame_end = self.anomalies_ending[self.anomalies_beginning.index(self.frame_start)]
        self.steps_left = self.frame_end - self.frame_start - 1
        self.active_arr_scaled = self.scaled_arr[self.frame_start:self.frame_end, :, :]
        self.active_arr = self.arr[self.frame_start:self.frame_end, :, :]
        self.current_positions = np.array([0]*self.n_crypto, dtype=np.float64)


    def reset(self):
        '''
        Docstring: This resets the environment to re-start training again. This calls reset session,
        but also initializes some values. We get the positions history from the dataframe; we want the
        last rows by above, so that the number of rows of the sub-df that we'll use is the same that the
        number of rows that each of the timestamps of the observation contains.
        '''
        self._reset_session()
        self.positions_history=scaler.fit_transform(self.dfPositions.loc[:self.n_retards + 1, :].values)
        self.trades = []
        self.cash = self.initial_balance
        self.wallet_value = self._get_wallet_value()
        return self._next_observation()

    def _get_current_prices(self):
        '''
        For that we need to take the timestamp corresponding to our current step,
        the first row in which retard0 is (retard0 represents no retard, that is,
        t0), and from that first row we take the columns in which the crypto prices
        are.
        '''
        return self.active_arr[self.current_step, 0, self.prices_positions]

    def _rebuild_positions_df(self):
        '''
        At each step, we insert the new positions into the dataframe in the first row,
        and then returns the new positions_df.
        '''
        d = []
        d.insert(0, {k:v for k, v in zip(self.crypto_names, self.current_positions)})
        self.dfPositions = pd.concat([pd.DataFrame(d), self.dfPositions], ignore_index=True, axis = 0)
        return self.dfPositions

    def _get_wallet_value(self):

        return np.dot(np.array(self.current_positions), self._get_current_prices()) + self.cash

    def take_action(self, actions):
        '''
        actions is a list of tuples in which the first element is the decision to make:
        0: buy
        1: sell
        2: hold
        The second element of the tuples is the quantity to buy/sell.
        '''
        current_prices = self._get_current_prices()

        act_types = np.array([action[0] for action in actions])

        act_q = np.array([action[1]/10 for action in actions])

        self.cache_['trades']['step_'+str(self.current_step)] = {}

        if np.any(np.array(act_types) == 1):
            ''' SELLING OPERATIONS'''

            positions_selling = list(np.where(np.array(act_types) == 1))

            quantities_selling = np.array(np.array(act_q[tuple(positions_selling)])*self.current_positions[tuple(positions_selling)], dtype=np.float64)

            #self.quantities_selling = np.array(act_q[positions_selling])*self.current_positions[positions_selling]

            prices_sold = np.array(current_prices[tuple(positions_selling)], dtype=np.float64)

            cryptos_selling = np.array(self.crypto_names)[tuple(positions_selling)]

            sales = np.dot(quantities_selling, np.array((prices_sold*(np.array(1.0, dtype=np.float64)-self.comission)), dtype=np.float64))

            self.current_positions[tuple(positions_selling)] = self.current_positions[tuple(positions_selling)].astype('float64') - np.float64(quantities_selling)

            self.cash += sales

            self.cache_['trades']['step_'+str(self.current_step)]['sell'] = [(c, v) for c, v in zip(cryptos_selling, quantities_selling)]

        if np.any(np.array(act_types) == 0):
            ''' BUYING OPERATIONS'''

            positions_buying = list(np.where(np.array(act_types) == 0))

            quantities_buying = np.array(act_q)[tuple(positions_buying)] #*self.current_positions[positions_buying]

            prices_bought = current_prices[tuple(positions_buying)]

            cryptos_buying = np.array(self.crypto_names)[tuple(positions_buying)]

            buys = np.dot(quantities_buying, (prices_bought*(1+self.comission)))

            if buys > self.cash:

                quantities_buying = np.array([max(0, q - self.buy_decrease) for q in quantities_buying])

                buys = np.dot(quantities_buying, (prices_bought*(1+self.comission)))

            if buys > self.cash:

                quantities_buying = np.array([max(0, q-self.buy_decrease) for q in quantities_buying])

                buys = np.dot(quantities_buying, (prices_bought*(1+self.comission)))

            if buys > self.cash:

                quantities_buying = np.array([max(0, q-self.buy_decrease) for q in quantities_buying])

                buys = np.dot(quantities_buying, (prices_bought*(1+self.comission)))

            if buys > self.cash:

                buys = 0

                quantities_buying = np.array([0]*quantities_buying.shape[0])

            self.current_positions[tuple(positions_buying)] = np.float64(quantities_buying)

            self.cash -= buys

            self.cache_['trades']['step_'+str(self.current_step)]['buy'] = [(c,v) for c, v in zip(cryptos_buying, quantities_buying)]

        if np.all(np.array(act_types) == 2):

            self.inactive_counts += 1

        self.wallet_value = self._get_wallet_value()

        self.cache_['wallet_composition']['step_'+str(self.current_step)] = list(self.current_positions) + [1.0 - self.current_positions.sum()]

        self.dfPositions = self._rebuild_positions_df()

    def step(self, actions):

        prev_net_worth = self.wallet_value

        self.take_action(actions)

        self.steps_left -= 1

        self.current_step += 1

        if self.steps_left == 0:
            self.cash += np.dot(self.current_positions, self._get_current_prices())
            self.current_positions = np.array([0]*self.n_crypto)

            self._reset_session()

        obs = self._next_observation()
        reward = self.wallet_value - prev_net_worth
        done = self.wallet_value <= 0

        return obs, reward, done, {}
