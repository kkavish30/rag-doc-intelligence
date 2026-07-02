EVAL_QA_PAIRS = [
    # --- Factual lookup (5) ---
    {
        "question": "What is the Transformer architecture based on?",
        "ground_truth": "The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    },
    {
        "question": "What dataset was used to train the base Transformer model for English-to-German translation?",
        "ground_truth": "The model was trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."
    },
    {
        "question": "How many attention heads does the base Transformer model use?",
        "ground_truth": "The base Transformer model uses 8 attention heads."
    },
    {
        "question": "What is the dimensionality of the model in the base Transformer configuration?",
        "ground_truth": "The base model uses a dimensionality of 512 (d_model = 512)."
    },
    {
        "question": "What optimizer was used to train the Transformer?",
        "ground_truth": "The Adam optimizer was used, with beta1=0.9, beta2=0.98, and epsilon=1e-9."
    },

    # --- Multi-hop / requires combining info (5) ---
    {
        "question": "How does the Transformer handle positional information without recurrence?",
        "ground_truth": "Since the Transformer has no recurrence or convolution, it adds positional encodings to the input embeddings using sine and cosine functions of different frequencies to inject information about token positions."
    },
    {
        "question": "What is the difference between the encoder and decoder self-attention layers?",
        "ground_truth": "The encoder uses standard multi-head self-attention over all positions, while the decoder uses masked multi-head self-attention to prevent positions from attending to subsequent positions, preserving the auto-regressive property."
    },
    {
        "question": "How does multi-head attention improve over single-head attention?",
        "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which a single attention head would average out and prevent."
    },
    {
        "question": "Why does the Transformer use scaled dot-product attention instead of regular dot-product attention?",
        "ground_truth": "The dot products are scaled by 1/sqrt(d_k) to counteract the effect of large values pushing the softmax function into regions with extremely small gradients when d_k is large."
    },
    {
        "question": "What are the three different ways multi-head attention is used in the Transformer model?",
        "ground_truth": "Multi-head attention is used in encoder-decoder attention (queries from decoder, keys/values from encoder), encoder self-attention, and decoder self-attention (masked)."
    },

    # --- Definition / explanation (5) ---
    {
        "question": "What is self-attention?",
        "ground_truth": "Self-attention, also called intra-attention, is an attention mechanism relating different positions of a single sequence to compute a representation of that sequence."
    },
    {
        "question": "What is the purpose of layer normalization in the Transformer?",
        "ground_truth": "Layer normalization normalizes inputs across the features to stabilize and accelerate training."
    },
    {
        "question": "What is a residual connection and why is it used?",
        "ground_truth": "A residual connection adds the input of a layer to its output before applying further processing, which helps in training deeper networks by mitigating the vanishing gradient problem."
    },
    {
        "question": "What is the feedforward network in each Transformer layer?",
        "ground_truth": "Each layer contains a position-wise fully connected feedforward network applied independently to each position, typically using ReLU activation."
    },
    {
        "question": "What evaluation metric was used to measure translation quality?",
        "ground_truth": "BLEU score was used to evaluate translation quality on the WMT 2014 English-German and English-French tasks."
    },

    # --- Out of scope (5) — guardrail should reject these ---
    {
        "question": "What is the capital of France?",
        "ground_truth": None  # Not in document — guardrail should reject
    },
    {
        "question": "How do I bake a chocolate cake?",
        "ground_truth": None
    },
    {
        "question": "What is the stock price of Google today?",
        "ground_truth": None
    },
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "ground_truth": None
    },
    {
        "question": "What is the best programming language for web development?",
        "ground_truth": None
    },
]