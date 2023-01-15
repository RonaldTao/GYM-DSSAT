import gym
import numpy as np
from gym import spaces

def dict2array(state):
    new_state = []
    for key  in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])        
    state = np.asarray(new_state)
    return state
   
def get_reward_costant_k1(state, action, next_state, done, k1, k2, action_amount):
    #if done, return Yield (topwt) - k*cost
    penalty = 0
    #k1=10**(-3)*(i_episode*2)
    # base cost is current input action
    #if action_amount > 160:
    #    if action!=0:
    #        penalty = (action_amount - 160)
    if done:
        reward = state[29] - k2*action - k1*penalty
        return reward
    #if done, return Yield (topwt) - k*cost
    else:
        reward = -k2*(action + state[25]) - k1*penalty
        return reward
    # otherwise, return -k*(action+leaching)

class myDssat(gym.Env):
  """Custom Environment that follows gym interface"""


  def __init__(self):
    super(myDssat, self).__init__()    # Define action and observation space
    # They must be gym.spaces objects    # Example when using discrete actions:
    self.action_space = spaces.Discrete(5)    # Example for using image as input:
    self.observation_space = spaces.Box(low=-1000, high=1000, shape=
                    (35,), dtype=np.float32)
    env_args = {
    'run_dssat_location': '/opt/dssat_pdi/run_dssat',  # assuming (modified) DSSAT has been installed in /opt/dssat_pdi
    'log_saving_path': './logs/dssat-pdi.log',  # if you want to save DSSAT outputs for inspection
    # 'mode': 'irrigation',  # you can choose one of those 3 modes
    # 'mode': 'fertilization',
    'mode': 'all',
    'seed': 123456,
    'random_weather': False,  # if you want stochastic weather
    }
    self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

    self.N_total_amount = 0


  def step(self, action):
    # Execute one time step within the environment
    self.N_total_amount += action*40
    action_dict = {
                    'anfer': action*40,  # if mode == fertilization or mode == all ; nitrogen to fertilize in kg/ha
                    'amir': 0,  # if mode == irrigation or mode == all ; water to irrigate in L/ha
            }
    state = dict2array(self.env.observation)
    next_state, reward, done, info = self.env.step(action_dict)
    if done:
        next_state = np.zeros((35,))
        reward = get_reward_costant_k1(state, action_dict['anfer'], next_state, done, 1, 0.1, self.N_total_amount)
    else:
        next_state = dict2array(next_state)
        reward = get_reward_costant_k1(state, action_dict['anfer'], next_state, done, 1, 0.1, self.N_total_amount)
    return next_state, reward, done, {}
  def reset(self):
    # Reset the state of the environment to an initial state
    self.env.reset()
    self.N_total_amount = 0



  def render(self, close=False):
    # Render the environment to the screen
     self.env.render