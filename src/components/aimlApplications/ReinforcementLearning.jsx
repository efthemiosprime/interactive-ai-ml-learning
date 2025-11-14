import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ReinforcementLearning() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 1. Q-Network for Deep Q-Learning
class DQN(nn.Module):
    """Deep Q-Network for Q-learning"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# 3. DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-networks (main and target)
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer()
    
    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        """Train on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# 4. Policy Gradient (REINFORCE Algorithm)
class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def act(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        self.episode_log_probs.append(log_prob)
        return action.item()
    
    def update(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate returns (discounted rewards)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset episode data
        self.episode_rewards = []
        self.episode_log_probs = []
        
        return policy_loss.item()

# 5. Actor-Critic Agent
class ActorCritic(nn.Module):
    """Actor-Critic network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic (value function)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy distribution
        policy = F.softmax(self.actor(x), dim=-1)
        
        # State value
        value = self.critic(x)
        
        return policy, value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def act(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update(self, state, action, reward, next_state, done, log_prob):
        """Update using TD error"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Get values
        _, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)
        
        # TD target
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value
        
        # Losses
        actor_loss = -log_prob * td_error.detach()
        critic_loss = F.mse_loss(value, td_target.detach())
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Example Usage: Training Loop
def train_dqn_agent(env, agent, episodes=1000):
    """Train DQN agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Select action
            action = agent.act(state)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > 32:
                agent.replay(batch_size=32)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        print(f"Episode {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return scores

# Example Usage
if __name__ == "__main__":
    # Initialize DQN agent
    agent = DQNAgent(state_size=4, action_size=2)
    
    # Example: CartPole environment
    # state = [cart_position, cart_velocity, pole_angle, pole_velocity]
    state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Get action
    action = agent.act(state)
    print(f"Selected action: {action}")
    
    # Simulate step
    next_state = np.array([0.1, 0.05, 0.02, 0.01])
    reward = 1.0
    done = False
    
    # Store experience
    agent.remember(state, action, reward, next_state, done)
    
    # Train
    loss = agent.replay(batch_size=1)
    print(f"Training loss: {loss}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque

# 1. Q-Network for Deep Q-Learning
class DQN(keras.Model):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = layers.Dense(hidden_size, activation='relu')
        self.fc2 = layers.Dense(hidden_size, activation='relu')
        self.fc3 = layers.Dense(action_size)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# 3. DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.memory = ReplayBuffer()
    
    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network(np.array([state]))
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Train on a batch"""
        with tf.GradientTape() as tape:
            # Current Q values
            current_q_values = self.q_network(states)
            current_q_values = tf.gather_nd(
                current_q_values,
                tf.stack([tf.range(len(actions)), actions], axis=1)
            )
            
            # Next Q values from target network
            next_q_values = self.target_network(next_states)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # Loss
            loss = keras.losses.mse(current_q_values, target_q_values)
        
        # Optimize
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        
        return loss
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        loss = self.train_step(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.numpy()
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

# 4. Policy Gradient (REINFORCE)
class PolicyNetwork(keras.Model):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = layers.Dense(hidden_size, activation='relu')
        self.fc2 = layers.Dense(hidden_size, activation='relu')
        self.fc3 = layers.Dense(action_size, activation='softmax')
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def act(self, state):
        state_tensor = tf.constant([state], dtype=tf.float32)
        probs = self.policy(state_tensor)
        action = tf.random.categorical(tf.math.log(probs), 1)[0, 0]
        log_prob = tf.math.log(probs[0, action])
        
        self.episode_log_probs.append(log_prob)
        return action.numpy()
    
    @tf.function
    def update(self):
        # Calculate returns
        returns = []
        G = 0.0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = tf.constant(returns, dtype=tf.float32)
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-9)
        
        # Policy loss
        log_probs = tf.stack(self.episode_log_probs)
        policy_loss = -tf.reduce_sum(log_probs * returns)
        
        # Optimize
        with tf.GradientTape() as tape:
            loss = policy_loss
        
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.policy.trainable_variables)
        )
        
        # Reset
        self.episode_rewards = []
        self.episode_log_probs = []
        
        return loss.numpy()

# 5. Actor-Critic
class ActorCritic(keras.Model):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = layers.Dense(hidden_size, activation='relu')
        self.fc2 = layers.Dense(hidden_size, activation='relu')
        self.actor = layers.Dense(action_size, activation='softmax')
        self.critic = layers.Dense(1)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.actor(x), self.critic(x)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    def act(self, state):
        state_tensor = tf.constant([state], dtype=tf.float32)
        policy, _ = self.model(state_tensor)
        action = tf.random.categorical(tf.math.log(policy), 1)[0, 0]
        return action.numpy(), tf.math.log(policy[0, action])
    
    @tf.function
    def update(self, state, action, reward, next_state, done, log_prob):
        state_tensor = tf.constant([state], dtype=tf.float32)
        next_state_tensor = tf.constant([next_state], dtype=tf.float32)
        
        _, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)
        
        td_target = reward + self.gamma * next_value[0, 0] * (1 - done)
        td_error = td_target - value[0, 0]
        
        actor_loss = -log_prob * td_error
        critic_loss = keras.losses.mse(value[0, 0], td_target)
        
        loss = actor_loss + critic_loss
        
        with tf.GradientTape() as tape:
            loss = actor_loss + critic_loss
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        return loss.numpy()

# Example Usage
if __name__ == "__main__":
    # Initialize DQN agent
    agent = DQNAgent(state_size=4, action_size=2)
    
    # Example state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = agent.act(state)
    print(f"Selected action: {action}")
    
    # Simulate step
    next_state = np.array([0.1, 0.05, 0.02, 0.01])
    reward = 1.0
    done = False
    
    agent.remember(state, action, reward, next_state, done)
    loss = agent.replay(batch_size=1)
    print(f"Training loss: {loss}")`;

  return (
    <div className="space-y-6">
      <div className="flex gap-4 mb-4">
        <button
          onClick={() => setSelectedFramework('pytorch')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'pytorch'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          PyTorch
        </button>
        <button
          onClick={() => setSelectedFramework('tensorflow')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'tensorflow'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          TensorFlow
        </button>
      </div>

      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, borderRadius: '0.5rem' }}
          showLineNumbers
        >
          {selectedFramework === 'pytorch' ? pytorchCode : tensorflowCode}
        </SyntaxHighlighter>
      </div>

      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Key Concepts:</h3>
        <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
          <li><strong>Q-Learning:</strong> Learn action-value function Q(s,a)</li>
          <li><strong>Policy Gradient:</strong> Directly optimize policy using gradients</li>
          <li><strong>Actor-Critic:</strong> Combines policy and value function learning</li>
          <li><strong>Experience Replay:</strong> Store and sample past experiences</li>
          <li><strong>Exploration vs Exploitation:</strong> Balance using epsilon-greedy</li>
        </ul>
      </div>
    </div>
  );
}

