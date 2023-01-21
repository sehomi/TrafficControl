import torch
from torch import nn
import numpy as np
from collections import deque,namedtuple
import matplotlib.pyplot as plt

from Base import SUMOHandler 
from SingleEnv import SingleLightEnv

from tqdm import tqdm

def choose_action_epsilon_greedy(net, state, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if np.random.random() < epsilon:
        # List of non-optimal actions (this list includes all the actions but the optimal one)
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly from non_optimal_actions
        action = np.random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()

def choose_action_softmax(net, state, temperature):
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
    softmax_out = softmax_out/np.sum(softmax_out)
                
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    action = np.random.choice(all_possible_actions,p=softmax_out)
    
    return action, net_out.cpu().numpy()



def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # Sample from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32, device='cpu')
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64, device='cpu')
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32, device='cpu')

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device='cpu') # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute Q values 
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device='cpu')
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)# Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping 
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) 
        # print(self.memory, batch_size)
        idxs = np.random.choice(np.array(range(len(self))), batch_size)
        idxs = np.unique(idxs).tolist()
        return [self.memory[i] for i in idxs]

    def __len__(self):
        return len(self.memory) 

class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                  nn.Linear(state_space_dim,64),
                  nn.ReLU(),
                  nn.Linear(64,64*2),
                  nn.ReLU(),
                  nn.Linear(64*2,action_space_dim)
                )

    def forward(self, x):
        x = x
        return self.linear(x)
    
sh = SUMOHandler()
sh.getNumLights()
tl = sh.trafficLights[10]
env = SingleLightEnv(sh, tl['traffic_light'], tl['link_index'])


# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

gamma = 0.99  
lr = 1e-3
target_net_update_steps = 10   
batch_size = 256   
bad_state_penalty = 0   
min_samples_for_training = 100   

# replay memory
replay_mem = ReplayMemory(100)    

policy_net = DQN(env.observation_space.shape[1], env.action_space.n)

target_net = DQN(env.observation_space.shape[1], env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

loss_fn = nn.SmoothL1Loss()

### Define exploration profile
initial_value = 5
num_iterations = 800
exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) 
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

plotting_rewards=[]

fig, ax = plt.subplots()

for episode_num, tau in enumerate(tqdm(exploration_profile)):
    
    state = env.zero_state()
    if episode_num != 0:
        state = env.reset()
    score = 0
    done = False
    
    
    while not done:
        
      # Choose the action following the policy
      action, q_values = choose_action_softmax(policy_net, state, temperature=tau)
      
      next_state, reward, done, info = env.step(action)
      
      score += reward
      
      if done:  
          reward += bad_state_penalty
          next_state = None
      

      replay_mem.push(state, action, next_state, reward)
      
      # Update the network
      if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
          update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

      # Visually render the environment 
      env.render()

      # Set the current state for the next iteration
      state = next_state
      
      plotting_rewards.append(reward)
      
      if len(plotting_rewards) % 100 == 0:
          ax.cla()
          ax.plot(range(len(plotting_rewards)), plotting_rewards)
          plt.savefig('reward.pdf', format='PDF')

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
   
    torch.save(target_net.state_dict(), "model.pt")
    # plotting_rewards.append(score)
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score

      
    #print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}") 