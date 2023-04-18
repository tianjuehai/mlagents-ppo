#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import numpy as np

class GenenalModel(parl.Model):
    """ The Model for Atari env
    Args:
        obs_space (Box): observation space.
        act_space (Discrete): action space.
    """

    def __init__(self, obs_space, act_space, hidden_layers):
        super(GenenalModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.hiddens = []
        self.inputs = nn.Linear(obs_space.shape[0], hidden_layers[0])
        if(len(hidden_layers) > 1):
            for i in range(len(hidden_layers) - 1):
                self.hiddens.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.outputV = nn.Linear(hidden_layers[-1], 1)
        self.outputLogitPre = nn.Linear(hidden_layers[-1], hidden_layers[-1])
        self.outputLogit = nn.Linear(hidden_layers[-1], act_space.n)




    def value(self, obs):
        out = F.relu(self.inputs(obs))
        if (len(self.hiddens) > 0):
            for i in range(len(self.hiddens)):
                out = F.relu(self.hiddens[i](out))
        value = self.outputV(out)
        return value

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        out = F.relu(self.inputs(obs))
        if (len(self.hiddens) > 0):
            for i in range(len(self.hiddens)):
                out = F.relu(self.hiddens[i](out))
        out = F.relu(self.outputLogitPre(out))
        logits = self.outputLogit(out)
        return logits

class GenenalModel_Continuous(parl.Model):
    """ The Model for Atari env
    Args:
        obs_space (Box): observation space.
        act_space (Discrete): action space.
    """

    def __init__(self, obs_space, act_space, hidden_layers):
        super(GenenalModel_Continuous, self).__init__()
        self.hidden_layers = hidden_layers
        self.hiddens = []
        self.inputs = nn.Linear(obs_space.shape[0], hidden_layers[0])
        if(len(hidden_layers) > 1):
            for i in range(len(hidden_layers) - 1):
                self.hiddens.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.outputV = nn.Linear(hidden_layers[-1], 1)
        self.outputpolicyPre = nn.Linear(hidden_layers[-1], hidden_layers[-1])
        self.fc_policy = nn.Linear(hidden_layers[-1], np.prod(act_space.shape))

        self.fc_pi_std = paddle.static.create_parameter(
            [1, np.prod(act_space.shape)],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

    def value(self, obs):
        out = F.relu(self.inputs(obs))
        if (len(self.hiddens) > 0):
            for i in range(len(self.hiddens)):
                out = F.relu(self.hiddens[i](out))
        value = self.outputV(out)
        return value

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        out = F.relu(self.inputs(obs))
        if (len(self.hiddens) > 0):
            for i in range(len(self.hiddens)):
                out = F.relu(self.hiddens[i](out))
        out = F.relu(self.outputpolicyPre(out))
        action_mean = self.fc_policy(out)

        action_logstd = self.fc_pi_std
        action_std = paddle.exp(action_logstd)
        return action_mean, action_std

class GenenalModel_Continuous_Divide(parl.Model):
    """ The Model for Atari env
    Args:
        obs_space (Box): observation space.
        act_space (Discrete): action space.
    """

    def __init__(self, obs_space, act_space, hidden_layers_actor, hidden_layers_critic):
        super(GenenalModel_Continuous_Divide, self).__init__()
        if (type(obs_space) == int):
            obs_space = np.zeros(obs_space)
        if (type(act_space) == int):
            act_space = np.zeros(act_space)
        self.hidden_layers_actor = hidden_layers_actor
        self.hidden_layers_critic = hidden_layers_critic
        self.hiddens_actor = []
        self.hiddens_critic = []
        self.inputs_actor = nn.Linear(obs_space.shape[0], hidden_layers_actor[0])
        self.inputs_critic = nn.Linear(obs_space.shape[0], hidden_layers_critic[0])

        if(len(hidden_layers_actor) > 1):
            for i in range(len(hidden_layers_actor) - 1):
                self.hiddens_actor.append(nn.Linear(hidden_layers_actor[i], hidden_layers_actor[i + 1]))
        if (len(hidden_layers_critic) > 1):
            for i in range(len(hidden_layers_critic) - 1):
                self.hiddens_critic.append(nn.Linear(hidden_layers_critic[i], hidden_layers_critic[i + 1]))

        self.outputV = nn.Linear(hidden_layers_critic[-1], 1)
        self.fc_policy = nn.Linear(hidden_layers_actor[-1], np.prod(act_space.shape))

        self.fc_pi_std = paddle.static.create_parameter(
            [1, np.prod(act_space.shape)],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

    def value(self, obs):
        out = F.relu(self.inputs_critic(obs))
        if (len(self.hiddens_critic) > 0):
            for i in range(len(self.hiddens_critic)):
                out = F.relu(self.hiddens_critic[i](out))
        value = self.outputV(out)
        return value

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        out = F.relu(self.inputs_actor(obs))
        if (len(self.hiddens_actor) > 0):
            for i in range(len(self.hiddens_actor)):
                out = F.relu(self.hiddens_actor[i](out))
        action_mean = self.fc_policy(out)

        action_logstd = self.fc_pi_std
        action_std = paddle.exp(action_logstd)
        return action_mean, action_std