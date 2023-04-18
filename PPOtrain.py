from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.environment import ActionTuple
import argparse
import numpy as np
from parl.utils import logger, summary

from storage import RolloutStorage
from parl.algorithms import PPO
from agent import PPOAgent
from genenal_model import GenenalModel_Continuous_Divide
from genenal_config import genenal_config_continuous
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name="UnityEnvironment", seed=1, side_channels=[channel])
channel.set_configuration_parameters(time_scale = 3.0)

env.reset()
behavior_names = list(env.behavior_specs.keys())
behavior_value = list(env.behavior_specs.values())
for i in range(len(behavior_names)):
    print(behavior_names[i])
    print("obs:",behavior_value[i].observation_specs[0], "   act:", behavior_value[0].action_spec)
discrete_actions = None
total_steps = 0
stepsNum = 0
obs_space = behavior_value[i].observation_specs[0]
act_space = behavior_value[i].action_spec.continuous_size

model = GenenalModel_Continuous_Divide(obs_space, act_space, [256,128], [256,128])
config = genenal_config_continuous
config['batch_size'] = int(config['env_num'] * config['step_nums'])
config['num_updates'] = int(
    config['train_total_steps'] // config['batch_size'])
ppo = PPO(
        model,
        clip_param=config['clip_param'],
        entropy_coef=config['entropy_coef'],
        initial_lr=config['initial_lr'],
        continuous_action=config['continuous_action'])
agent = PPOAgent(ppo, config)

rollout = RolloutStorage(config['step_nums'], config['env_num'], obs_space,
                         act_space)
DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
obs = DecisionSteps.obs[0]
agentsNum = len(DecisionSteps)
done = np.zeros(agentsNum, dtype='float32')
total_reward = np.zeros(agentsNum, dtype='float32')
this_action = np.zeros((agentsNum, act_space), dtype='float32')
next_obs = np.zeros((agentsNum, obs_space.shape[0]), dtype='float32')
for update in range(1, config['num_updates'] + 1):
    for step in range(0, config['step_nums']):
        value, action, logprob, _ = agent.sample(obs)
        agentsNumNow = len(DecisionSteps)
        if agentsNumNow == 0:
            action = np.random.rand(0, 2)
        else:
            action = action.reshape(agentsNumNow, act_space)
            this_action = action
        actions = ActionTuple(action, discrete_actions)
        env.set_actions(behavior_names[0], actions)
        env.step()
        DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
        next_obs_Decision = DecisionSteps.obs[0]
        next_obs_Terminal = TerminalSteps.obs[0]
        if(len(next_obs_Terminal) != 0):
            next_obs = np.zeros((agentsNum, obs_space.shape[-1]))
            rewards = np.zeros(agentsNum, dtype=float)
            next_done = np.zeros(agentsNum, dtype=bool)
            j = 0
            for i in TerminalSteps.agent_id:
                next_obs[i] = next_obs_Terminal[j]
                rewards[i] = TerminalSteps.reward[j]
                next_done[i] = True
                j += 1
            rollout.append(obs, this_action, logprob, rewards, done, value.flatten())
            obs, done = next_obs, next_done
            total_reward += rewards

        if(len(next_obs_Decision) != 0):
            step += 1
            next_obs = next_obs_Decision
            rewards = DecisionSteps.reward
            next_done = np.zeros(agentsNum, dtype=bool)

            rollout.append(obs, this_action, logprob, rewards, done, value.flatten())
            obs, done = next_obs, next_done
            total_reward += rewards

        total_steps += 1
        stepsNum += 1
        if(stepsNum % 200 == 199):
            arv_reward = total_reward / 200
            print("total_steps:{0}".format(total_steps))
            print("arv_reward:", arv_reward)
            stepsNum = 0
            total_reward = 0

    value = agent.value(obs)
    rollout.compute_returns(value, done)
    #print("compute_returns , learn...")
    value_loss, action_loss, entropy_loss, lr = agent.learn(rollout)


env.close()