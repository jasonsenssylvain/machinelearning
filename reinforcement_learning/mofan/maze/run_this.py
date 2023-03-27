from maze_env import Maze
from rl_brain import QLearningTable

def update():
  for episode in range(100):
    # initial observation
    observation = env.reset()
    print('observation: ', observation)
    
    while True:
      # fresh env
      env.render()
      
      # RL choose action based on observation
      action = rl.choose_action(str(observation))
      print('action: ', action)
      
      # RL take action and get next observation and reward
      observation_, reward, done = env.step(action)
      print('observation_: , reward: , done: ', observation_, reward, done)
      
      rl.learn(str(observation), action, reward, str(observation_))
      
      # swap observation
      observation = observation_
      
      # # break while loop when end of this episode
      # if done:
      #   break
      # break while loop when end of this episode
      if done:
        break

  # end of game
  print('game over')
  env.destroy()
    
if __name__ == "__main__":
  env = Maze()
  rl = QLearningTable(actions=list(range(env.n_actions)))
  
  env.after(100, update)
  env.mainloop()
  