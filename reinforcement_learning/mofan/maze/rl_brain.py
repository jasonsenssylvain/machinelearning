import numpy as np
import pandas as pd

class QLearningTable:
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    self.actions = actions
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon = e_greedy # 决策率
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
  # 选择动作
  def choose_action(self, observation):
    self.check_state_exist(observation)
    # action selection
    if np.random.uniform() < self.epsilon:
      # choose best action
      state_action = self.q_table.loc[observation, :]
      print('q_table: ')
      print(self.q_table)
      
      # some actions have the same value
      state_action = state_action.reindex(np.random.permutation(state_action.index))
      print('state_action: ')
      print(state_action)
      
      action = state_action.idxmax()
    else:
      # choose random action
      action = np.random.choice(self.actions)
    return action
  
  def learn(self, s, a, reward, s_):
    self.check_state_exist(s_)
    q_predict = self.q_table.loc[s, a]
    if s_ != 'terminal':
      q_target = reward + self.gamma * self.q_table.loc[s_, :].max()
    else:
      q_target = reward
    self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
    print('q_table after learn: ')
    print(self.q_table)
  
  # 检查状态是否存在，若不存在将作为索引添加在 q-table中，行为的值初始化为0
  def check_state_exist(self, state):
    if state not in self.q_table.index:
      self.q_table = self.q_table.append(
        pd.Series(
          [0]*len(self.actions),
          index=self.q_table.columns,
          name=state,
        )
      )
      print('after update, q_table is: \n ', self.q_table)