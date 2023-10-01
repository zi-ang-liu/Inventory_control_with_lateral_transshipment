import numpy as np
from scipy.stats import poisson

import gymnasium as gym
from gymnasium import spaces


class LateralTransEnv(gym.Env):
    metadata = metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None, 
                 max_onhand_qty_1=10, 
                 max_onhand_qty_2=10, 
                 max_order_qty_1=10,
                 max_order_qty_2=10, 
                 max_trans_qty=5, 
                 lambda_1=1, 
                 lambda_2=2,
                 order_cost_1=5,
                 order_cost_2=5,
                 trans_cost=2,
                 holding_cost_1=1,
                 holding_cost_2=1,
                 lost_sales_cost_1=15,
                 lost_sales_cost_2=15,
                 price=10):

        self.max_onhand_qty_1 = max_onhand_qty_1
        self.max_onhand_qty_2 = max_onhand_qty_2

        self.max_order_qty_1 = max_order_qty_1
        self.max_order_qty_2 = max_order_qty_2

        self.max_trans_qty = max_trans_qty

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.order_cost_1 = order_cost_1
        self.order_cost_2 = order_cost_2

        self.trans_cost = trans_cost

        self.holding_cost_1 = holding_cost_1
        self.holding_cost_2 = holding_cost_2

        self.lost_sales_cost_1 = lost_sales_cost_1
        self.lost_sales_cost_2 = lost_sales_cost_2

        self.price = price

        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array(
            [self.max_onhand_qty_1, self.max_onhand_qty_2]), shape=(2,), dtype=int)
        self.action_space = spaces.MultiDiscrete(
            [max_order_qty_1 + 1, max_order_qty_2 + 1, max_trans_qty * 2 + 1])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # generate the on_hand_inv_level uniformly at random
        on_hand_inv_level = self.np_random.integers(low=np.array([0, 0]), high=np.array(
            [self.max_onhand_qty_1, self.max_onhand_qty_2]), size=(2,), dtype=int)

        self.observation = on_hand_inv_level
        info = {}

        # self.period = 0
        return self.observation, info

    def step(self, action):

        demand_1 = self.np_random.poisson(self.lambda_1)
        demand_2 = self.np_random.poisson(self.lambda_2)
        demand = (demand_1, demand_2)

        # action[2] \in [0, action[2]*2 + 1]
        trans_action = action[2] - 5
        assert trans_action >= -self.max_trans_qty and trans_action <= self.max_trans_qty

        if trans_action > 0:  # transfer from 1 to 2
            transfer_qty_1_2 = trans_action
            transfer_qty_2_1 = 0
        else:  # transfer from 2 to 1
            transfer_qty_1_2 = 0
            transfer_qty_2_1 = -trans_action

        on_hand_inv_level= self.observation

        transfer_qty_1_2 = min(transfer_qty_1_2, on_hand_inv_level[0])
        transfer_qty_2_1 = min(transfer_qty_2_1, on_hand_inv_level[1])

        onhand_inv_1 = max(
            0, on_hand_inv_level[0] - transfer_qty_1_2 - demand[0]) + action[0] + transfer_qty_2_1
        onhand_inv_2 = max(
            0, on_hand_inv_level[1] - transfer_qty_2_1 - demand[1]) + action[1] + transfer_qty_1_2

        if onhand_inv_1 > self.max_onhand_qty_1:
            onhand_inv_1 = self.max_onhand_qty_1
        if onhand_inv_2 > self.max_onhand_qty_2:
            onhand_inv_2 = self.max_onhand_qty_2

        self.observation = np.array([onhand_inv_1, onhand_inv_2], dtype=int)

        total_cost = self.order_cost_1 * action[0] + \
            self.order_cost_2 * action[1] + \
            self.trans_cost * abs(trans_action) + \
            self.holding_cost_1 * on_hand_inv_level[0] + \
            self.holding_cost_2 * on_hand_inv_level[1] + \
            self.lost_sales_cost_1 * max(0, demand[0] - on_hand_inv_level[0] + transfer_qty_1_2) + \
            self.lost_sales_cost_2 * \
            max(0, demand[1] - on_hand_inv_level[1] + transfer_qty_2_1)

        reward = self.price * min(on_hand_inv_level[0] - transfer_qty_1_2, demand[0]) + \
            self.price * min(on_hand_inv_level[1] - transfer_qty_2_1,
                             demand[1]) - total_cost

        # self.period += 1

        # if self.period == 100:
        #     truncation = True

        return self.observation, reward, False, False, {}

    def render(self):
        pass

    def render(self):
        pass
