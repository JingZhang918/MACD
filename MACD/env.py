import gym
import numpy as np
import pandas as pd
import itertools
import random
from operator import add, sub
from datetime import datetime
# from config import config
import matplotlib.pyplot as plt

# initial_balance = config.BALANCE
# symbols = config.SYMBOLS
# max_share_buy = config.MAX_SHARE_BUY
# max_share_sell = config.MAX_SHARE_SELL
# column_names_order = config.COLUMN_NAMES_ORDER
random.seed(0)
#transaction unit: 1 hand = 100 shares
initial_balance = 1000000

class StockTradingEnv(gym.Env):
    def __init__(self, data, index=0):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.index = index
        self.index_dates = data.index
        self.date_number = len(self.index_dates.unique())

        self.n_state = 1 + 1 + data.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_state,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3) #buy, hold, sell

        self.state = self.reset()

        self.balance_record = []
        self.share_owning_record = []
        self.transaction_price_record = []
        self.transaction_cost_record = []
        self.action_record = []
        self.reward_record = []
        self.asset_record = []

    def step(self, action):
        action = action - 1
        # print(action)
        transaction_share = 10 * 100

        current_balance = self.state[0]
        current_own_share = self.state[1]
        #columns: ohlcv
        high_price = self.state[3]
        low_price = self.state[4]
        close_price = self.state[5]
        # current_available_price = [random.uniform(a, b) for a, b in zip(low_price, high_price)]
        transaction_price = high_price if action == 1 else low_price

        # 10 hands transaction
        #commision charges/transfer fee/stamp tax
        transaction_amount = transaction_price * action * transaction_share
        if action == 0:
            transaction_cost = 0
        else:
            commision_charges = max(abs(transaction_amount) * 0.003, 5)
            transfer_fee = abs(transaction_amount) * 0.00002
            stamp_tax = abs(transaction_amount) * 0.001 if action == -1 else 0
            transaction_cost = commision_charges+transfer_fee+stamp_tax

        #会使股票股数变为负数和使账户余额变为负数的action都不会产生任何后果

        taken_action_balance = current_balance - transaction_cost - transaction_amount #sell +/ buy -
        taken_action_own_share = current_own_share + action * transaction_share

        if (taken_action_own_share < 0)|(taken_action_balance<0):
            taken_action_balance = current_balance
            taken_action_own_share = current_own_share
            transaction_cost = 0

        self.action_record.append(action)
        self.transaction_cost_record.append(transaction_cost)
        self.transaction_price_record.append(transaction_price)
        self.balance_record.append(taken_action_balance)
        self.share_owning_record.append(taken_action_own_share)

        self.index += 1

        done = bool(self.index >= self.date_number - 1)
        next_state = np.asarray([taken_action_balance] + [taken_action_own_share] + self._get_market_data())
        self.state = next_state
        asset = taken_action_balance + taken_action_own_share * transaction_price
        reward = asset - initial_balance

        self.reward_record.append(reward)
        self.asset_record.append(asset)

        info = {}
        return self.state, reward, done, info


    def _get_market_data(self) -> list:
        '''

        :return: a list of a state's market data_storage
        '''
        dataframe = self.data.loc[self.index_dates[self.index]].sort_index()
        return list(dataframe.values)

    def reset(self):
        self.index = 0
        self.state = np.asarray([initial_balance] + [0] + self._get_market_data())
        return self.state

    def render(self, mode='detail'):

        if mode == 'detail':
            print(f"The initial balance is {initial_balance} and share holding is {[0]}.")

            for i in range(len(self.action_record)):
                print(f"actions about to take: {self.action_record[i]} at prices: {self.transaction_price_record[i]}"
                      f", which incurs {self.transaction_cost_record[i]} transaction costs.")
                print(
                    f"current balance: {round(self.balance_record[i],2)}, current shares holding: {self.share_owning_record[i]},"
                    f"total asset is {round(self.asset_record[i],2)}")

        elif mode == 'summary':
            print(
                f"The total asset is {round(self.asset_record[-1],2)} after {self.date_number} days of trading. The cash balance"
                f"is {round(self.balance_record[-1],2)}")
            # for i in range(len(symbols)):
            #     print(f"stock {symbols[i]} owned {self.share_owning_record[-1][i]} shares, worth of "
            #           f"{round(self.share_owning_record[-1][i] * self.transaction_price_record[-1][i],2)}")
            print(f"The total transaction cost is {round(np.sum(self.transaction_cost_record),2)}")
            annualized_return = self.asset_record[-1]/initial_balance-1
            print(f"The annualized return is {round(annualized_return*100,2)}%")
            temp_asset_record = self.asset_record[1:] + [0]
            returns = np.array(temp_asset_record)/np.array(self.asset_record) - 1
            standard_deviation = np.std(returns[:-1])
            sharp_ration = (annualized_return - 0.02)/standard_deviation
            print(f"standard deviation is: {round(standard_deviation*100,2)}, and the sharp ration is: {round(sharp_ration,2)}")

        elif mode == 'plot':

            time = range(self.date_number)

            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle('Performance')

            ax1.plot(time, [initial_balance]+self.asset_record, 'o-')
            ax1.set_ylabel('asset record')

            ax2.plot(time, [0]+self.reward_record, '.-')
            ax2.set_xlabel('time (D)')
            ax2.set_ylabel('reward record')

            plt.show()

        # elif mode == 'save_reward':
        #     f1 = open(config.RESULT_SAVE_PATH_SINGLE, "a")
        #     f1.write(f"==============================================================================\n")
        #     f1.write(f"=========================={datetime.now()}===================================\n")
        #     # f.write(f"===========================symbol: {}=====================================")
        #     f1.write(str(np.around(np.asarray(self.reward_record),2))+"\n")
        #     f1.write(str(np.around(np.asarray(self.asset_record),2))+"\n")
        #     f1.close()

        elif mode == "performance":
            print("The reward and asset records are: ")
            print(self.reward_record)
            print(self.asset_record)

        elif mode == 'record':
            return self.reward_record, self.asset_record

        elif mode == "action":
            plt.hist([a[0] for a in self.action_record])
            plt.show()

        elif mode == "asset":
            return self.asset_record

        else:
            raise ValueError("Invalid Mode! Try detail/summary/plot")

#
# from stable_baselines3.common.vec_env import DummyVecEnv
# data = pd.read_csv("~/Documents/Codes/Single-Stock-Trading/data_storage/data_test.csv").set_index('date')
# print(data)
# env_train = DummyVecEnv([lambda: StockTradingEnv(data)])
#
# # env_train.reset()
# env_train.step([0])