import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import json
import time
from typing import Dict, List, Any, Tuple

# Memory item with associated metadata and priority
class MemoryItem:
    def __init__(self, content: str, metadata: Dict[str, Any] = None, timestamp: float = None):
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.priority = 0.5  # Default priority (will be updated by the model)
        self.access_count = 0
        self.last_accessed = self.timestamp
    
    def access(self):
        """Update access statistics when memory is accessed"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_features(self) -> np.ndarray:
        """Convert memory to feature vector for model input"""
        # Features that might indicate importance
        age = time.time() - self.timestamp
        recency = time.time() - self.last_accessed
        
        # Extract text-based features (simplified version)
        has_date = any(word in self.content.lower() for word in ["today", "tomorrow", "date", "deadline", "schedule"])
        has_name = any(char.isupper() for char in self.content.split()) and len(self.content.split()) > 1
        has_number = any(char.isdigit() for char in self.content)
        
        # Create feature vector
        features = np.array([
            age / 86400,  # Age in days (normalized)
            recency / 86400,  # Recency in days (normalized)
            self.access_count / 10,  # Access count (normalized)
            len(self.content) / 1000,  # Content length (normalized)
            float(has_date),
            float(has_name),
            float(has_number),
            float(self.metadata.get("is_question", 0)),
            float(self.metadata.get("is_instruction", 0)),
            float(self.metadata.get("is_fact", 0))
        ])
        
        return features

    def __str__(self) -> str:
        return f"Memory(priority={self.priority:.2f}, content={self.content[:50]}...)"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        """Create from dictionary"""
        item = cls(content=data["content"], metadata=data["metadata"], timestamp=data["timestamp"])
        item.priority = data["priority"]
        item.access_count = data["access_count"]
        item.last_accessed = data["last_accessed"]
        return item


# Deep Q-Network for memory prioritization
class MemoryPrioritizer:
    def __init__(self, feature_size: int = 10, learning_rate: float = 0.001):
        self.feature_size = feature_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.memory_buffer = deque(maxlen=2000)  # Replay buffer for training
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
    
    def _build_model(self):
        """Build a neural network model for Q-value prediction"""
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.feature_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Output is priority between 0 and 1
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        """Update target network with weights from the main network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def predict_priority(self, memory_item: MemoryItem) -> float:
        """Predict priority for a memory item"""
        features = memory_item.to_features().reshape(1, -1)
        return self.model.predict(features, verbose=0)[0][0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory_buffer.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action (priority value) based on state"""
        if np.random.rand() <= self.epsilon:
            # Exploration: return random priority
            return np.random.random()
        
        # Exploitation: predict priority using model
        return self.model.predict(state.reshape(1, -1), verbose=0)[0][0]
    
    def replay(self):
        """Train model using experience replay"""
        if len(self.memory_buffer) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory_buffer, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0][0]
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][0] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)


# Priority-based short-term memory using reinforcement learning
class ReinforcedShortTermMemory:
    def __init__(self, max_size: int = 10, feature_extractor=None):
        self.max_size = max_size
        self.memory_queue = deque(maxlen=max_size)
        self.prioritizer = MemoryPrioritizer()
        self.feature_extractor = feature_extractor
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, MemoryItem]:
        """Add new memory item if it's important enough"""
        new_item = MemoryItem(content, metadata)
        
        # Use prioritizer to determine if we should add this memory
        new_item.priority = self.prioritizer.predict_priority(new_item)
        
        # If memory is full, compare with lowest priority item
        should_add = True
        removed_item = None
        
        if len(self.memory_queue) >= self.max_size:
            lowest_priority_item = min(self.memory_queue, key=lambda x: x.priority)
            
            # Only add if new item has higher priority
            if new_item.priority > lowest_priority_item.priority:
                # Remove lowest priority item and remember as negative example
                self.memory_queue.remove(lowest_priority_item)
                self.memory_queue.append(new_item)
                
                # Store in replay buffer for reinforcement learning
                self._reinforce_decision(new_item, lowest_priority_item, True)
            else:
                # Memory rejected (store as learning example)
                should_add = False
                self._reinforce_decision(lowest_priority_item, new_item, False)
        else:
            # Memory not full, simply add
            self.memory_queue.append(new_item)
        
        # Train the prioritizer
        self.prioritizer.replay()
        
        return should_add, new_item

    def _reinforce_decision(self, kept_item, discarded_item, added_new):
        """Record decision for reinforcement learning"""
        # Current state: features of the kept item
        state = kept_item.to_features()
        
        # Action: priority assigned to kept_item
        action = kept_item.priority
        
        # Next state: features of the memory after the decision
        next_state = discarded_item.to_features()
        
        # Reward: positive if we made the right decision (to be refined with feedback)
        # This is a simplified reward function, ideally would be based on actual usefulness
        reward = 0.5  # Neutral baseline
        
        # Store experience for replay
        self.prioritizer.remember(state, action, reward, next_state, False)
    
    def get_memories(self) -> List[MemoryItem]:
        """Get all memories sorted by priority"""
        return sorted(self.memory_queue, key=lambda x: x.priority, reverse=True)

    def update_priority(self, content_substring: str, feedback_score: float):
        """Update priority based on external feedback"""
        updated = False
        for item in self.memory_queue:
            if content_substring in item.content:
                # Record old state and action
                old_state = item.to_features()
                old_priority = item.priority
                
                # Update priority based on feedback (weighted average that gradually shifts toward feedback)
                # Giving more weight to user feedback over time
                item.priority = (item.priority * 0.7 + feedback_score * 0.3)
                
                # Store in replay buffer with feedback as reward
                new_state = item.to_features()
                self.prioritizer.remember(old_state, old_priority, feedback_score, new_state, False)
                updated = True
                
        # Return whether any memories were updated
        return updated
    
    def search(self, query: str) -> List[MemoryItem]:
        """Search for memories based on content match"""
        results = []
        for item in self.memory_queue:
            if query.lower() in item.content.lower():
                item.access()  # Record access
                results.append(item)
        return results
    
    def clear(self):
        """Clear all memories"""
        self.memory_queue.clear()
    
    def save(self, filepath: str):
        """Save memories and model to file"""
        data = {
            "memories": [item.to_dict() for item in self.memory_queue],
            "max_size": self.max_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        # Save model separately
        self.prioritizer.save(f"{filepath}_model")
    
    def load(self, filepath: str):
        """Load memories and model from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.max_size = data.get("max_size", self.max_size)
        self.memory_queue = deque(maxlen=self.max_size)
        
        for item_data in data.get("memories", []):
            self.memory_queue.append(MemoryItem.from_dict(item_data))
        
        # Load model separately
        try:
            self.prioritizer.load(f"{filepath}_model")
        except:
            print("No model file found, using default model")


# Example usage
if __name__ == "__main__":
    # Create the reinforced short-term memory
    memory = ReinforcedShortTermMemory(max_size=5)
    
    # Add some memories with different importance levels
    memory.add_memory("Remember to submit the report by Friday", {"is_instruction": 1})
    memory.add_memory("The coffee machine is broken", {"is_fact": 1})
    memory.add_memory("What was the name of that restaurant again?", {"is_question": 1})
    memory.add_memory("User prefers dark mode for all applications", {"is_fact": 1})
    memory.add_memory("Meeting with John at 3pm tomorrow", {"is_fact": 1, "is_instruction": 0.5})
    
    # Try to add one more (should replace lowest priority)
    added, item = memory.add_memory("Critical deadline for project X is next Monday!", 
                              {"is_instruction": 1, "is_fact": 1})
    
    print(f"New memory was added: {added}")
    
    # Print all memories with priorities
    print("\nCurrent memories:")
    for item in memory.get_memories():
        print(f"Priority: {item.priority:.2f} - {item.content}")
    
    # Update priority based on feedback
    memory.update_priority("coffee machine", 0.1)  # Lower priority
    memory.update_priority("deadline", 0.9)  # Higher priority
    
    print("\nMemories after feedback:")
    for item in memory.get_memories():
        print(f"Priority: {item.priority:.2f} - {item.content}")



