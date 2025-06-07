import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size=4, action_size=2, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size=4, action_size=2, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, target_update_freq=100):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        
        # Initialize target network with same weights
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Store loss for plotting
        self.losses.append(loss.item())
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

def train_dqn(episodes=1000, max_steps_per_episode=500, render_every=100):
    """Train the DQN agent"""
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    
    # Training metrics
    rewards = []
    epsilons = []
    avg_losses = []
    
    print("Starting DQN training...")
    print(f"Network architecture: {agent.q_network}")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps_per_episode):
            # Render occasionally
            if episode % render_every == 0 and episode > 0:
                env.render()
            
            # Choose and take action
            action = agent.choose_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            done = done or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Store metrics
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Average loss over recent training steps
        recent_losses = agent.losses[-100:] if agent.losses else [0]
        avg_losses.append(np.mean(recent_losses))
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_loss = np.mean(recent_losses)
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Avg Loss: {avg_loss:.4f} | "
                  f"Buffer Size: {len(agent.memory)}")
    
    env.close()
    return agent, rewards, epsilons, avg_losses

def test_dqn(agent, episodes=10, render=True, max_steps=500):
    """Test the trained DQN agent"""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    test_rewards = []
    
    print("\nTesting trained DQN agent...")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Use trained policy (no exploration)
            action = agent.choose_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            done = done or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1:2d}: Reward = {total_reward:3.0f}, Steps = {step + 1}")
    
    env.close()
    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage test reward: {avg_test_reward:.2f}")
    print(f"Success rate (≥195 steps): {sum(r >= 195 for r in test_rewards)/len(test_rewards)*100:.1f}%")
    return test_rewards

def plot_training_progress(rewards, epsilons, losses):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards plot
    axes[0, 0].plot(rewards, alpha=0.6, color='blue', linewidth=0.8)
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(rewards)), moving_avg, 'r-', linewidth=2, label='100-episode MA')
        axes[0, 0].legend()
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Solved threshold')
    
    # Epsilon decay
    axes[0, 1].plot(epsilons, color='orange')
    axes[0, 1].set_title('Exploration Rate (Epsilon)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1, 0].plot(losses, alpha=0.7, color='red')
    if len(losses) >= 50:
        loss_ma = np.convolve(losses, np.ones(50)/50, mode='valid')
        axes[1, 0].plot(range(49, len(losses)), loss_ma, 'darkred', linewidth=2)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Performance histogram
    recent_rewards = rewards[-200:] if len(rewards) >= 200 else rewards
    axes[1, 1].hist(recent_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(x=195, color='green', linestyle='--', linewidth=2, label='Solved threshold')
    axes[1, 1].set_title('Recent Performance Distribution')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_network(agent):
    """Analyze the trained network"""
    print(f"\nDQN Network Analysis:")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}")
    print(f"Replay buffer size: {len(agent.memory)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Sample some Q-values for different states
    print(f"\nSample Q-values for different states:")
    sample_states = [
        [0.0, 0.0, 0.0, 0.0],      # Balanced
        [1.0, 0.0, 0.1, 0.0],      # Cart right, pole tilted
        [-1.0, 0.0, -0.1, 0.0],    # Cart left, pole tilted opposite
        [0.0, 1.0, 0.0, 1.0],      # Moving right
        [0.0, -1.0, 0.0, -1.0],    # Moving left
    ]
    
    state_descriptions = [
        "Balanced state",
        "Cart right, pole right",
        "Cart left, pole left", 
        "Moving right",
        "Moving left"
    ]
    
    agent.q_network.eval()
    with torch.no_grad():
        for state, desc in zip(sample_states, state_descriptions):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.q_network(state_tensor).cpu().numpy()[0]
            action = np.argmax(q_values)
            print(f"{desc:20}: Q=[{q_values[0]:6.3f}, {q_values[1]:6.3f}] -> Action {action}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the DQN agent
    print("="*60)
    print("DEEP Q-NETWORK (DQN) TRAINING")
    print("="*60)
    
    trained_agent, training_rewards, training_epsilons, training_losses = train_dqn(episodes=1000)
    
    # Plot training progress
    plot_training_progress(training_rewards, training_epsilons, training_losses)
    
    # Analyze the trained network
    analyze_network(trained_agent)
    
    # Test the trained agent
    test_rewards = test_dqn(trained_agent, episodes=10, render=False)
    
    # Save the trained model
    trained_agent.save_model('cartpole_dqn_model.pth')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Show final performance
    final_avg = np.mean(training_rewards[-100:])
    print(f"Final 100-episode average: {final_avg:.2f}")
    print(f"Environment solved: {'Yes' if final_avg >= 195 else 'No'}")
    
    # Example of loading and testing saved model
    # new_agent = DQNAgent()
    # new_agent.load_model('cartpole_dqn_model.pth')
    # test_dqn(new_agent, episodes=5, render=False)