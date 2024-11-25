import csv
from collections import defaultdict


# Function to perform policy evaluation
def policy_evaluation(V, policy, gamma, markov_chain, markov_chain_reward):
    '''
    INPUTS

    OUTPUTS

    '''
    # what is this
    epsilon = 1e-6
    # what does loop do
    while True:
        delta = 0
        # what does this do
        for s in markov_chain:
            v = V[s]
            # make this more descriptive of a variable name
            total = 0 
            # FIXME Convert policy[s] to a tuple
            action = policy[s]
            for s1, prob in markov_chain[s][action].items():
                reward = markov_chain_reward[s][action][s1]
                total += prob * (reward + gamma * V[s1])
            V[s] = total
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break


def policy_improvement(V, policy, gamma, markov_chain, markov_chain_reward):
    policy_stable = True
    for s in markov_chain:
        old_action = policy[s]
        max_value = float('-inf')
        best_action = None
        for a in markov_chain[s]:
            action_value = 0
            for s1, prob in markov_chain[s][a].items():
                action_value += prob * (markov_chain_reward[s][a][s1] + gamma * V[s1])
            if action_value > max_value:
                max_value = action_value
                best_action = a
        policy[s] = best_action  # Ensure hashable type for keys
        if old_action != policy[s]:
            policy_stable = False
    return policy_stable





# Define a defaultdict to store the three-level dictionary
markov_chain = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
markov_chain_reward = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
# Open the CSV file and read its contents
with open('Markov_Chain_Probs.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skip the header row if it exists
    next(csv_reader, None)
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Extract data from the row
        current_state = row[0].strip()
        action = row[1].strip()
        next_state = row[2].strip()
        probability = float(row[3])
        reward = float(row[4])
        comment = row[5].strip()
        # Populate the three-level dictionary
        markov_chain[current_state][action][next_state] += probability
        markov_chain_reward[current_state][action][next_state] = reward
# Printing the populated dictionary (for verification)
for current_state, actions in markov_chain.items():
    for action, next_states in actions.items():
        sum_probs = 0
        for next_state, probability in next_states.items():
            sum_probs += probability
            # print(
            #     f"Current State: {current_state}, Action: {action}, Next State: {next_state}, Probability: {probability}")
        if sum_probs < 1:
            for next_state, probability in next_states.items():
                markov_chain[current_state][action][next_state] = probability / sum_probs



# Printing the populated dictionary (for verification)
counter = 0
for current_state, actions in markov_chain.items():
    print('')
    print(f'Current State {current_state}')
    for action, next_states in actions.items():
        print(f'Action {action} for {current_state}')
        for next_state, probability in next_states.items():
            print(f'Next state {next_state} given Action {action}')
            reward = markov_chain_reward[current_state][action][next_state]
            counter += 1
            # print(
            #     f"Current State: {current_state}, Action: {action}, Next State: {next_state}, Probability: {probability}, reward: {reward}")


# Define the discount factor
gamma = 0.90

# Define the initial value function and policy
V = defaultdict(float)
policy = {}

# Printing the populated dictionary (for verification)
for current_state, actions in markov_chain.items():
    for action, next_states in actions.items():
        policy[current_state] = action
        break


# Policy Iteration
while True:
    policy_evaluation(V, policy, gamma, markov_chain, markov_chain_reward)
    if policy_improvement(V, policy, gamma, markov_chain, markov_chain_reward):
        policy_evaluation(V, policy, gamma, markov_chain, markov_chain_reward)
        break




# Printing the optimal policy and value function
print("Optimal Policy:")
for state, action in policy.items():
    print(f"State: {state}, Action: {action}")

print("\nOptimal Value Function:")
for state, value in V.items():
    print(f"State: {state}, Value: {value}")