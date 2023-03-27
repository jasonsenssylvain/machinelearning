import numpy as np
import pandas as pd
import time 

np.random.seed(2)  # reproducible

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.01  # fresh time for one move

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    return table
  
q_table = build_q_table(N_STATES, ACTIONS)
print(q_table)

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.idxmax()    
    return action_name
  
def get_env_feedback(S, A):
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R
  
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        
# 定义强化学习函数
def rl():
    # 初始化Q表格
    q_table = build_q_table(N_STATES, ACTIONS)  
    # 循环MAX_EPISODES次
    for episode in range(MAX_EPISODES):  
        # 初始化步数计数器
        step_counter = 0
        # 初始化状态
        S = 0  
        # 是否终止
        is_terminated = False  
        # 更新环境
        update_env(S, episode, step_counter)  
        # 当前状态未终止
        while not is_terminated:
            # 选择动作
            A = choose_action(S, q_table)  
            # 获取下一个状态和奖励
            S_, R = get_env_feedback(S, A)  
            # 预测Q值
            q_predict = q_table.loc[S, A]  
            # 如果下一个状态不是终止状态
            if S_ != 'terminal':
                # 计算真实Q值
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  
            else:
                # 下一个状态是终止状态
                q_target = R  
                # 终止本次循环
                is_terminated = True  
            # 更新Q表格
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
            # 移动到下一个状态
            S = S_  
            # 更新环境
            update_env(S, episode, step_counter+1)  
            # 步数计数器加1
            step_counter += 1
    # 返回Q表格
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    