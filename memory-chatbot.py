import numpy as np
import tensorflow as tf
from reinforcememoy import ReinforcedShortTermMemory, MemoryItem
from typing import Dict, Any, List, Tuple
import time
import re
import random

# Try to import OpenAI - provide instructions if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Sentiment analysis using TextBlob if available, otherwise use a simple rule-based approach
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

class SentimentAnalyzer:
    """Simple sentiment analyzer that can be used with or without TextBlob"""
    
    def __init__(self):
        self.positive_words = set([
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
            "helpful", "useful", "thanks", "thank", "appreciate", "like", "love",
            "correct", "right", "yes", "perfect", "awesome", "brilliant"
        ])
        
        self.negative_words = set([
            "bad", "poor", "terrible", "awful", "useless", "unhelpful", "wrong",
            "incorrect", "no", "not", "don't", "hate", "dislike", "stupid",
            "confused", "confusing", "error", "mistake", "fail"
        ])
    
    def analyze(self, text: str) -> float:
        """Analyze sentiment of text and return score from -1.0 to 1.0"""
        if TEXTBLOB_AVAILABLE:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            return blob.sentiment.polarity
        else:
            # Use simple rule-based approach
            text = text.lower()
            words = re.findall(r'\b\w+\b', text)
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            
            # Calculate sentiment score between -1 and 1
            total_count = positive_count + negative_count
            if total_count == 0:
                return 0  # Neutral if no sentiment words found
            
            return (positive_count - negative_count) / total_count


class AIMemoryChatbot:
    """An AI chatbot with short-term memory that improves based on sentiment feedback"""
    
    def __init__(self, api_key=None, memory_size=10):
        """Initialize the chatbot with a short-term memory system"""
        self.memory = ReinforcedShortTermMemory(max_size=memory_size)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_history = []
        self.user_feedback_count = 0
        self.api_key = api_key
        
        # Initialize OpenAI client if API key is provided
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("NOTE: OpenAI API not available. Using simple response generation.")
    
    def add_to_memory(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Add user message to short-term memory with appropriate metadata"""
        if metadata is None:
            metadata = {}
        
        # Detect if message is a question
        is_question = "?" in message
        
        # Detect if message contains instructions
        instruction_indicators = ["please", "could you", "would you", "can you", "help me"]
        is_instruction = any(indicator in message.lower() for indicator in instruction_indicators)
        
        # Set metadata flags for the memory item
        metadata.update({
            "is_question": float(is_question),
            "is_instruction": float(is_instruction),
            "is_fact": float(not (is_question or is_instruction)),  # Simple assumption
            "timestamp": time.time()
        })
        
        # Add to memory system
        added, memory_item = self.memory.add_memory(message, metadata)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time()
        })
    
    def process_feedback(self, feedback: str) -> None:
        """Process user feedback and update memory priorities"""
        # Get sentiment score from feedback
        sentiment_score = self.sentiment_analyzer.analyze(feedback)
        
        # Normalize sentiment to 0-1 range for priority
        priority = (sentiment_score + 1) / 2
        
        # If we have conversation history
        if self.conversation_history:
            # Find the most recent user message
            for i in range(len(self.conversation_history) - 1, -1, -1):
                if self.conversation_history[i]["role"] == "user":
                    recent_message = self.conversation_history[i]["content"]
                    # Update priority of this message in memory
                    self.memory.update_priority(recent_message[:20], priority)
                    break
        
        # Increment feedback counter
        self.user_feedback_count += 1
        
        # Every 5 feedback instances, update the prioritizer's model
        if self.user_feedback_count % 5 == 0:
            self.memory.prioritizer.update_target_network()
    
    def generate_response(self, user_message: str) -> str:
        """Generate response based on user message and memory"""
        # Add user message to memory first
        self.add_to_memory(user_message)
        
        # Get relevant memories for context
        relevant_memories = self.memory.search(user_message)
        if not relevant_memories:
            relevant_memories = self.memory.get_memories()[:3]  # Get top 3 by priority
        
        # Create context from memories
        memory_context = "\n".join([f"- {mem.content}" for mem in relevant_memories])
        
        if self.client:  # If OpenAI is available
            try:
                # Create prompt with memory context
                messages = [
                    {"role": "system", "content": f"""You are a helpful assistant with short-term memory. 
                    Use the following relevant information from your memory to inform your response:
                    
                    {memory_context}
                    
                    If the memory contains relevant information, use it in your response.
                    If not, respond based on your general knowledge."""},
                    {"role": "user", "content": user_message}
                ]
                
                # Add a subset of conversation history for context
                history_subset = self.conversation_history[-6:]  # Last 3 exchanges (6 messages)
                for item in history_subset:
                    messages.append({"role": item["role"], "content": item["content"]})
                
                # Generate response using OpenAI
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=150
                )
                
                response_text = response.choices[0].message.content
                
            except Exception as e:
                # Fallback in case of API error
                response_text = f"I encountered an issue processing your request. {str(e)}"
        
        else:  # Simple response generator if OpenAI not available
            # Create a simple response based on pattern matching
            response_text = self._simple_response_generator(user_message, relevant_memories)
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": time.time()
        })
        
        return response_text
    
    def _simple_response_generator(self, message: str, relevant_memories: List[MemoryItem]) -> str:
        """Generate simple responses without OpenAI API"""
        # Check for greetings
        if any(greeting in message.lower() for greeting in ["hello", "hi", "hey"]):
            return "Hello! How can I assist you today?"
        
        # Check for questions
        if "?" in message:
            if relevant_memories:
                memory_text = relevant_memories[0].content
                return f"Based on what I remember: {memory_text}. Does that help?"
            else:
                return "I'm not sure about that. Could you provide more information?"
        
        # Check for thanks
        if any(thanks in message.lower() for thanks in ["thank", "thanks", "appreciate"]):
            return "You're welcome! Glad I could help."
        
        # Default response with memory incorporation if available
        if relevant_memories:
            return f"I recall that {relevant_memories[0].content}. Is there anything specific you'd like to know about that?"
        else:
            responses = [
                "I'm here to help. What would you like to know?",
                "Could you provide more details about what you're looking for?",
                "I'm listening. Feel free to ask me anything.",
                "I'm your AI assistant. How can I assist you today?"
            ]
            return random.choice(responses)
    
    def clear_memory(self):
        """Clear all memories and conversation history"""
        self.memory.clear()
        self.conversation_history = []
        print("Memory and conversation history cleared.")
    
    def save_state(self, filepath: str):
        """Save the memory state to file"""
        self.memory.save(filepath)
        print(f"Memory state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the memory state from file"""
        self.memory.load(filepath)
        print(f"Memory state loaded from {filepath}")


# Interactive chat loop for testing the chatbot
def interactive_chat(api_key=None):
    print("Initializing AI Memory Chatbot...")
    chatbot = AIMemoryChatbot(api_key=api_key)
    
    print("\n==== AI Memory Chatbot ====")
    print("- Type 'quit' or 'exit' to end the conversation")
    print("- Type 'feedback: [your feedback]' to provide feedback")
    print("- Type 'clear memory' to clear the chatbot's memory")
    print("- Type 'save memory' to save the current memory state")
    print("- Type 'load memory' to load a previously saved memory state")
    print("================================\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        elif user_input.lower() == "clear memory":
            chatbot.clear_memory()
            continue
        
        elif user_input.lower() == "save memory":
            filepath = input("Enter filepath to save memory: ").strip()
            chatbot.save_state(filepath)
            continue
        
        elif user_input.lower() == "load memory":
            filepath = input("Enter filepath to load memory from: ").strip()
            chatbot.load_state(filepath)
            continue
        
        elif user_input.lower().startswith("feedback:"):
            feedback = user_input[9:].strip()  # Extract the feedback text
            chatbot.process_feedback(feedback)
            print("Thank you for your feedback! I'll learn from it.")
            continue
        
        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    # If you have an OpenAI API key, replace None with your API key as a string
    api_key = "YOUR-API-KEY"
    if not OPENAI_AVAILABLE:
        print("OpenAI package not found. You can install it with: pip install openai")
    if not TEXTBLOB_AVAILABLE:
        print("TextBlob package not found. You can install it with: pip install textblob")
    
    # Start interactive chat
    interactive_chat(api_key)
