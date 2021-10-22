import yfinance as yf
import pandas_ta as ta
import pandas as pd
from env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import HER
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from datetime import datetime


TICKER = '600887.SS'

def run() -> None:
    # download data
    df = yf.Ticker(TICKER).history(period='7y')[['Open','High','Low','Close', 'Volume']]
    # process data
    df.ta.macd(close='close', fast=12, slow=26, append=True)
    df.columns = [x.lower() for x in df.columns]
    # derive features
    #Crossovers (lagging)
    # MACD crossing below the centerline – Downtrend
    co1 = ((df.macd_12_26_9.shift(1) > 0) & (df.macd_12_26_9 < 0)).astype(int).rename('co1')
    # MACD crossing above the centerline – Uptrend
    co2 = ((df.macd_12_26_9.shift(1) < 0) & (df.macd_12_26_9 > 0)).astype(int).rename('co2')
    # MACD crossing below the signal line – Sell
    co3 = ((df.macd_12_26_9.shift(1) < df.macds_12_26_9.shift(1)) & (df.macd_12_26_9 > df.macds_12_26_9)).astype(int).rename('co3')
    # MACD crossing above the signal line – Buy
    co4 = ((df.macd_12_26_9.shift(1) > df.macds_12_26_9.shift(1)) & (df.macd_12_26_9 < df.macds_12_26_9)).astype(int).rename('co4')

    # Histogram Reversals (leading)
    hr1 = ((df.macdh_12_26_9.shift(1) > 0) & (df.macdh_12_26_9 < 0)).astype(int).rename('hr1')
    hr2 = ((df.macdh_12_26_9.shift(1) < 0) & (df.macdh_12_26_9 > 0)).astype(int).rename('hr2')

    # Convergence vs. Divergence (lagging)
    cd = (abs(df.macdh_12_26_9) > abs(df.macdh_12_26_9.shift(1))).astype(int).rename('cd')

    # Zero Crossovers (lagging)
    zc1 = ((df.macds_12_26_9.shift(1) > 0) & (df.macds_12_26_9 < 0)).astype(int).rename('zc1')
    zc2 = ((df.macds_12_26_9.shift(1) < 0) & (df.macds_12_26_9 > 0)).astype(int).rename('zc2')

    #integrate all features
    integrated_data = pd.concat([df, co1, co2, co3, co4, hr1, hr2, cd, zc1, zc2], axis=1, join='inner')
    # print(integrated_data.tail(5))

    # print(integrated_data.Date)
    # integrated_data.to_csv('data.csv')

    training_start_date = '2015-01-01'
    training_end_date = '2020-10-20'
    test_start_date = '2020-10-21'
    test_end_date = '2021-10-21'

    training_data = integrated_data[(integrated_data.index >= training_start_date)*(integrated_data.index <= training_end_date) ]
    test_data = integrated_data[(integrated_data.index >= test_start_date)*(integrated_data.index <= test_end_date) ]

    train_env = DummyVecEnv([lambda: StockTradingEnv(training_data)])
    test_env = DummyVecEnv([lambda: StockTradingEnv(test_data)])

    print("Start training")
    training_start = datetime.now()
    model = PPO('MlpPolicy', train_env, learning_rate=1e-5, gamma=0.98, clip_range=0.15, n_steps=4096, n_epochs=20, batch_size=128,verbose=1)
    model.learn(total_timesteps=2000000)
    # model = A2C('MlpPolicy', train_env)
    # model.learn(total_timesteps=10000)
    # model = DQN('MlpPolicy', train_env)
    # model.learn(total_timesteps=10000)


    training_end = datetime.now()
    print(str(training_end-training_start))


    obs = test_env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)

    obs = train_env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = train_env.step(action)

    # train_env.render('summary')
    test_env.render('detail')


if __name__ == '__main__':
    run()
