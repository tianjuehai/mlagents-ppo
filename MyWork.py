from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.environment import ActionTuple
import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


# if file_name=None ,  use editor to train
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name="UnityEnvironment", seed=1, side_channels=[channel])
# 环境运行速度调整为3倍
channel.set_configuration_parameters(time_scale = 3.0)

env.reset()
behavior_names = list(env.behavior_specs.keys())
behavior_value = list(env.behavior_specs.values())
# 这里可以得到observation和action的shape
for i in range(len(behavior_names)):
    print(behavior_names[i])
    print("obs:",behavior_value[i].observation_specs, "   act:", behavior_value[0].action_spec)
# 这里可以得到智能体组的数据，包括状态，奖励，ID等，DecisionSteps包含回合未结束的智能体信息
# TerminalSteps包含回合已结束的智能体信息，包括结束原因等
# 智能体回合结束后进入TerminalSteps，并且下一回合进行重置并进入DecisionSteps，也有可能此回合同时进入DecisionSteps
DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
print("agentIds:",DecisionSteps.agent_id)
print("rewards:",DecisionSteps.reward)
print("obs:",DecisionSteps.obs)
values = list(DecisionSteps.values())
for i in range(len(values)):
    print(values[i])
print("ter_reward:",TerminalSteps.reward)
print("interrupted:", TerminalSteps.interrupted)     # 是否由于步数到了而停止
# 重置环境的条件以及外部奖励函数都在环境中的C#脚本中编写，这里只负责收集数据进行训练，实现训练和数据收集的分离处理
discrete_actions = None
this_time = time.time()
for i in range(1000000):
    print("step:",i,"  last_step_cost:",time.time() - this_time)
    this_time = time.time()
    DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
    agentsNum = len(DecisionSteps.agent_id)
    print("exist:",DecisionSteps.agent_id,"   Dead:",TerminalSteps.agent_id)
    #print("reward:",DecisionSteps.reward,"reward_dead:",TerminalSteps.reward)
    #print("obs:",DecisionSteps.obs,"DeadObs:",TerminalSteps.obs)
    continuous_actions = (np.random.rand(agentsNum, 2) - 0.5) * 2
    actions = ActionTuple(continuous_actions, discrete_actions)
    #print("actions:", actions.continuous)
    env.set_actions(behavior_names[0],actions)
    env.step()


env.close()