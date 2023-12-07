import numpy as np
from MADDPG import MADDPG
from MultiAgentReplayBuffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3,simple_v3
import numpy as np
import time
from typing import List, Tuple, Dict


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    scenario= "simple"
    env = simple_adversary_v3.parallel_env(render_mode=None)
    observations, infos = env.reset()

    num_agents= len(env.agents)
    actor_dims= []

    for i in env.agents:
        actor_dims.append(env.observation_space(i).shape[0])

    critic_dims= sum(actor_dims)

    num_actions= env.action_space(env.agents[0]).n

    print("Num agents:", num_agents)
    print("Critic dims: ", critic_dims)
    print("Actor dims: ", actor_dims)
    print("Action space: ",num_actions)

    maddpg= MADDPG(actor_dims, critic_dims, num_agents, num_actions,
                    fc1= 64, fc2= 64, alpha= 0.01, beta= 0.01, scenario=scenario, chkpt_dir= 'tmp/maddpg/')

    memory= MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, num_actions, num_agents, batch_size= 1024)

    PRINT_INTERVAL= 50

    N_EPISODES= 8000
    MAX_STEPS= 23
    total_steps= 0
    scores= []
    evaluate= False
    best_score= -10

    if evaluate:
        maddpg.load_checkpoint()

    for i in range(N_EPISODES):
        dict_obs,info= env.reset()
        obs = [value for value in dict_obs.values()]
        score= 0
        done= [False]*num_agents
        episode_step= 0

        while not any(done):
            if evaluate:
                env.render()
            #setof actions for each agent

            actions:List[np.ndarray]= maddpg.choose_action(obs)

            discrete_actions: Dict[str,int]= {agent: np.argmax(actions[i]) for i,agent in enumerate(env.agents)}

            dict_obs_, dict_reward, dict_done,truncations, info= env.step(discrete_actions)
            obs_ = [value for value in dict_obs_.values()]
            reward= [value for value in dict_reward.values()]
            done= [value for value in dict_done.values()]

            #Convert For passing in enviroment
            state= obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step > MAX_STEPS:
                done= [True]* num_agents

            #Store the samples of each agents in the memory replay buffer
            memory.store_transition(obs,state,actions,reward,obs_,state_,done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg.learn(memory)

            obs= obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        scores.append(score)
        avg_score= np.mean(scores[-100:])

        if not evaluate:
            if avg_score > best_score:
                maddpg.save_checkpoint()
                best_score= avg_score

        if i % PRINT_INTERVAL== 0 and i>0:
            print("EPISODE",i,"average score {:.1f}".format(avg_score))


