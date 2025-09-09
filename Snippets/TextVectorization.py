import keras

# Long Sentences to simulate a small dataset for demonstration purposes
Sentence = [
    "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.",
    "Machine learning works by using algorithms to parse data, learn from it, and make decisions.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers."
]

Vector = keras.layers.TextVectorization(standardize=None, split="whitespace", output_mode="int", output_sequence_length=None, pad_to_max_tokens=False)
Vector.adapt(Sentence)
print(f"Vectorized Sentence: {Vector("AI algorithms machine")}")
#Result: Vectorized Sentence: [41 35 22]