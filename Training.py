import keras, tensorflow, numpy

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
    "What are generative models?",
    "What is transfer learning?"
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
    "Generative models are a type of model that can generate new data instances similar to the training data.",
    "Transfer learning is a technique where a pre-trained model is adapted to a new, but related, task."
]

# Add start and end tokens to answers
answers = ["<start> " + answer + " <end>" for answer in answers]

# Create a TextVectorization layer for questions - Vectorization used for tokenization and integer encoding of text data
vectorize_question = keras.layers.TextVectorization(standardize=None, split="whitespace", output_mode="int", output_sequence_length=None, pad_to_max_tokens=False )
# Adapt the vectorization layer to the questions dataset
vectorize_question.adapt(questions)
# Get the vocabulary size for questions
vocab_count_questions = vectorize_question.vocabulary_size()
# vocab count is the sum of unique tokens + 1 for padding token. Repeated tokens are counted once.
print("Vocabulary size for questions:\n", vocab_count_questions)
# Vectorize the questions to integer sequences
questions_sequences = vectorize_question(questions).numpy().tolist()
print("Vectorized questions (integer sequences):\n", questions_sequences)
maximum_length_questions = max(len(seq) for seq in questions_sequences)
print("Maximum length of questions (in tokens):\n", maximum_length_questions)
# Pad the sequences to ensure uniform length for batching - Padding is done post-sequence with 0s
question_padding = keras.preprocessing.sequence.pad_sequences(questions_sequences, maxlen=maximum_length_questions, padding='post', value=0)
print("Padded questions (uniform length):\n", question_padding)

# Create a TextVectorization layer for answers - Vectorization used for tokenization and integer encoding of text data
vectorize_answer = keras.layers.TextVectorization(standardize=None, split="whitespace", output_mode="int", output_sequence_length=None, pad_to_max_tokens=False)
# Adapt the vectorization layer to the answers dataset
vectorize_answer.adapt(answers)
# Get the vocabulary size for answers
vocab_count_answers = vectorize_answer.vocabulary_size()
# vocab count is the sum of unique tokens + 1 for padding token. Repeated tokens are counted once.
print("Vocabulary size for answers:\n", vocab_count_answers)
# Vectorize the answers to integer sequences
answers_sequences = vectorize_answer(answers).numpy().tolist()
print("Vectorized answers (integer sequences):\n", answers_sequences)
maximum_length_answers = max(len(seq) for seq in answers_sequences)
print("Maximum length of answers (in tokens):\n", maximum_length_answers)
# Pad the sequences to ensure uniform length for batching - Padding is done post-sequence with 0s
answer_padding = keras.preprocessing.sequence.pad_sequences(answers_sequences, maxlen=maximum_length_answers, padding='post', value=0)
print("Padded answers (uniform length):\n", answer_padding)
