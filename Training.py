import keras, tensorflow

# print TensorFlow and Keras versions
print("TensorFlow version:", tensorflow.__version__)
print("Keras version:", keras.__version__)

# Simple Dataset for Questions
questions = [
    "What is AI?",
    "How does machine learning work?",
    "What is deep learning?",
    "Explain neural networks.",
    "What is natural language processing?",
    "How do attention mechanisms work?",
    "What is reinforcement learning?",
    "Explain supervised learning.",
    "What is unsupervised learning?",
    "What are generative models?"
]

# Corresponding Answers
answers = [
    "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.",
    "Machine learning works by using algorithms to parse data, learn from it, and make decisions.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Neural networks are computing systems inspired by the human brain that can learn from data.",
    "Natural language processing is a field of AI that focuses on the interaction between computers and human language.",
    "Attention mechanisms allow models to focus on specific parts of the input data when making predictions.",
    "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
    "Supervised learning is a type of machine learning where the model is trained on labeled data",
    "Unsupervised learning is a type of machine learning where the model is trained on unlabeled data to find hidden patterns.",
    "Generative models are a type of model that can generate new data instances similar to the training data."
]

# Add start and end tokens to answers
answers = ["<start> " + answer + " <end>" for answer in answers]