
# AI Memory Chatbot with Reinforcement Learning

This project implements an intelligent chatbot with a reinforcement learning-based short-term memory system. The chatbot learns from user interactions and sentiment feedback to prioritize important memories and improve its responses over time.

## Features

- **Short-term memory** with priority-based retention using deep Q-learning
- **Sentiment analysis** of user feedback to improve memory prioritization
- **Dynamic prioritization** of memories based on relevance and importance
- **Automatic cleanup** of less important memories when capacity is reached
- **Memory persistence** - save and load memory states between sessions

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy

Optional but recommended packages:
- OpenAI Python package (for improved response generation)
- TextBlob (for better sentiment analysis)

### Installation

1. Install the required dependencies:
   ```
   pip install tensorflow numpy
   ```

2. For improved functionality, install optional packages:
   ```
   pip install openai textblob
   ```

3. If using OpenAI API, you'll need to provide your API key.

### Usage

To start the interactive chatbot:

```python
from memory_chatbot import interactive_chat

# Replace None with your OpenAI API key if available
interactive_chat(api_key=None)
```

## How It Works

### Memory System

The chatbot uses a deque-based short-term memory system enhanced with deep Q-learning:

1. New information is evaluated for importance using a neural network
2. If memory is full, low-priority items are replaced with more important ones
3. Feedback from the user updates the priority values through reinforcement learning

### Chatbot Interaction

1. **Basic interaction**: The chatbot responds to user queries by using its memory
2. **Feedback**: Type `feedback: [your feedback]` to provide sentiment feedback
3. **Memory management**: Use commands like `clear memory`, `save memory` and `load memory`

### Learning Process

1. User interacts with the chatbot
2. Chatbot stores important information in short-term memory
3. User provides feedback on chatbot responses
4. Reinforcement learning model adjusts memory priorities based on feedback
5. Over time, the chatbot improves at remembering what's important

## Example Interaction

```
You: Hello, my name is Alice
Chatbot: Hello! Nice to meet you Alice. How can I help you today?

You: I prefer dark mode interfaces
Chatbot: I'll remember that you prefer dark mode interfaces. Is there anything specific you're looking for?

You: What's my name?
Chatbot: Based on what I remember, your name is Alice. Is there anything else you'd like to know?

You: feedback: That's correct, good job!
Thank you for your feedback! I'll learn from it.

...later in the conversation...

You: Do you remember my interface preference?
Chatbot: Yes, I recall that you prefer dark mode interfaces. Would you like me to help you with something related to that?
```

## Customization

You can modify various parameters in the `ReinforcedShortTermMemory` class to customize the memory behavior:

- `max_size`: Maximum number of items to keep in memory
- `feature_size`: Dimensionality of the feature vector for memory items
- Learning parameters in the `MemoryPrioritizer` class (epsilon, decay, etc.)

## Extending the System

The modular design allows for easy extension:
- Add new feature extractors for better memory prioritization
- Integrate with more sophisticated LLMs
- Implement long-term memory storage
- Add multi-modal memory capabilities
