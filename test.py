from os import truncate
from pettingzoo.mpe import simple_adversary_v3
import numpy as np
import time

env = simple_adversary_v3.parallel_env(render_mode="human")
observations, infos= env.reset(seed=42)

#print("Num of agents:", len(env.agents))
#print("Agents: ",env.agents)
#for i in env.agents:
    #print("Observation space for ",i,":", env.observation_space(i).shape)
    #print("Num of actions: ", env.action_space(i).n)
    #print()
print("Action space: ", env.action_spaces)
#print("INFOS:",infos)
#for i in env.agents:
    #print(observations[i])



actions= {i: 0 for i in env.agents}

print(actions)


obs_, reward, done,truncated,info= env.step(actions)

for i in env.agents:
    print("OBS",i,":", obs_[i])
    print("Reward",i,":", reward[i])
    print("Done",i,":", done[i])
    print("Trunc",i,":", truncated[i])
    print("Info",i,":", info[i])
    print()




time.sleep(3)






