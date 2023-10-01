import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson
from tqdm import tqdm


def build_dynamics():
    print('Building dynamics...')

    EPSILON = 1e-4

    max_order_qty_1 = 10
    max_order_qty_2 = 10

    max_onhand_qty_1 = 10
    max_onhand_qty_2 = 10

    max_trans_qty = 5

    lambda_1 = 1
    lambda_2 = 2

    order_cost_1 = 5
    order_cost_2 = 5

    trans_cost = 2

    holding_cost_1 = 1
    holding_cost_2 = 1

    lost_sales_cost_1 = 15
    lost_sales_cost_2 = 15

    price = 10

    # initialize state space and action space
    state_space = [(i, j) for i in range(max_onhand_qty_1 + 1)
                   for j in range(max_onhand_qty_2 + 1)]

    action_space = [(i, j, k) for i in range(max_order_qty_1 + 1)
                    for j in range(max_order_qty_2 + 1)
                    for k in range(-max_trans_qty, max_trans_qty + 1)]

    # build probability dictionary
    prob_dict = {}

    for i in range(max_onhand_qty_1 + 1):
        if poisson.pmf(i, lambda_1) < EPSILON:
            max_demand_1 = i
            break

    for i in range(max_onhand_qty_2 + 1):
        if poisson.pmf(i, lambda_2) < EPSILON:
            max_demand_2 = i
            break

    demand_space = [(i, j) for i in range(max_demand_1 + 1)
                    for j in range(max_demand_2 + 1)]

    for demand_1 in range(max_demand_1 + 1):
        prob_dict[demand_1, lambda_1] = poisson.pmf(
            demand_1, lambda_1)

    for demand_2 in range(max_demand_2 + 1):
        prob_dict[demand_2, lambda_2] = poisson.pmf(
            demand_2, lambda_2)

    # build dynamics
    dynamics = {}
    for state in tqdm(state_space):
        for action in action_space:
            dynamics[state, action] = {}

            if action[2] > 0:  # transfer from 1 to 2
                transfer_qty_1_2 = action[2]
                transfer_qty_2_1 = 0
            else:  # transfer from 2 to 1
                transfer_qty_1_2 = 0
                transfer_qty_2_1 = -action[2]

            transfer_qty_1_2 = min(transfer_qty_1_2, state[0])
            transfer_qty_2_1 = min(transfer_qty_2_1, state[1])

            for demand in demand_space:

                onhand_inv_1 = max(
                    0, state[0] - transfer_qty_1_2 - demand[0]) + action[0] + transfer_qty_2_1
                onhand_inv_2 = max(
                    0, state[1] - transfer_qty_2_1 - demand[1]) + action[1] + transfer_qty_1_2

                if onhand_inv_1 > max_onhand_qty_1:
                    onhand_inv_1 = max_onhand_qty_1
                if onhand_inv_2 > max_onhand_qty_2:
                    onhand_inv_2 = max_onhand_qty_2

                next_state = (onhand_inv_1, onhand_inv_2)

                total_cost = order_cost_1 * action[0] + \
                    order_cost_2 * action[1] + \
                    trans_cost * abs(action[2]) + \
                    holding_cost_1 * state[0] + \
                    holding_cost_2 * state[1] + \
                    lost_sales_cost_1 * max(0, demand[0] - state[0] + transfer_qty_1_2) + \
                    lost_sales_cost_2 * \
                    max(0, demand[1] - state[1] + transfer_qty_2_1)

                reward = price * min(state[0] - transfer_qty_1_2, demand[0]) + \
                    price * min(state[1] - transfer_qty_2_1,
                                demand[1]) - total_cost

                prob = prob_dict[demand[0], lambda_1] * \
                    prob_dict[demand[1], lambda_2]

                if (next_state, reward) in dynamics[state, action]:
                    dynamics[state, action][next_state, reward] += prob
                else:
                    dynamics[state, action][next_state, reward] = prob

            assert np.isclose(np.sum(list(dynamics[state, action].values())), 1, atol=1.0e-3)

    init_value = np.zeros((max_onhand_qty_1 + 1, max_onhand_qty_2 + 1))
    init_policy = np.zeros(
        (max_onhand_qty_1 + 1, max_onhand_qty_2 + 1, 3), dtype=int)

    return dynamics, state_space, action_space, init_value, init_policy


def value_iteration(dynamics, state_space, action_space, value, policy, theta=1e-4, gamma=0.9):
    print('Value iteration...')

    # initialize value
    delta = np.inf
    k = 0
    while delta >= theta:
        k = k + 1
        value_old = value.copy()
        for state in state_space:
            # Update V[s].
            value[state] = max([sum([prob * (reward + gamma * value_old[next_state]) for (
                next_state, reward), prob in dynamics[state, action].items()]) for action in action_space])
            # print('State {}, value = {}'.format(state, value[state]))
        delta = np.max(np.abs(value - value_old))
        print('Iteration {}, delta = {}'.format(k, delta))

    for state in state_space:
        best_value = -np.inf
        for action in action_space:
            value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if value_temp > best_value:
                best_value = value_temp
                policy[state[0], state[1], :] = action
    return value, policy

if __name__ == '__main__':

    dynamics, state_space, action_space, init_value, init_policy = build_dynamics()
    value, policy = value_iteration(
        dynamics, state_space, action_space, init_value, init_policy)

    np.save('policy', policy)
    np.save('value',value)

    # plot
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(policy[:, :, i], cmap='viridis', annot=True, fmt='d', ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel('Location 2', fontsize=16)
        ax.set_ylabel('Location 1', fontsize=16)
        ax.set_title('Policy', fontsize=20)
        plt.savefig('policy_' + str(i) + '.png')
        plt.close()

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(value, cmap='viridis', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Location 2', fontsize=16)
    ax.set_ylabel('Location 1', fontsize=16)
    ax.set_title('Value', fontsize=20)
    plt.savefig('Value.png')
    plt.close()
