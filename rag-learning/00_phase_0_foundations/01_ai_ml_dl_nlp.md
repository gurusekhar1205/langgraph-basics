# Lesson 1: AI, ML, Deep Learning, NLP, Generative AI, and LLMs

## 1. Learning objectives

By the end of this lesson, you should be able to:

1. Explain Artificial Intelligence (AI).
2. Distinguish Machine Learning (ML) from traditional programming.
3. Explain Deep Learning (DL) and neural networks.
4. Explain Natural Language Processing (NLP).
5. Explain Generative AI and Large Language Models (LLMs).
6. Understand how these concepts are related.
7. Explain the difference between training and inference.
8. Understand why an LLM can produce incorrect answers.
9. Explain why RAG can improve answers that depend on external documents.

---

## 2. What is Artificial Intelligence?

Artificial Intelligence is a broad area of computer science concerned
with building systems that perform tasks normally associated with
human intelligence.

Examples of these tasks include:

- Understanding language
- Recognizing objects in images
- Planning a route
- Making predictions
- Recommending products
- Solving problems
- Generating text or images

AI is the broad category. It does not refer to only chatbots or LLMs.

### Real-world example

Consider an email application that identifies spam.

The application receives an email and decides whether it should be
classified as spam or legitimate.

A traditional program could use manually written rules. A machine
learning system could instead learn patterns from examples of spam
and legitimate messages.

Both are possible approaches to an AI-related problem, but they work
differently.

### Important point

Not every AI system learns from data. Some systems use explicit rules,
search algorithms, planning, or other techniques.

---

## 3. Traditional programming versus Machine Learning

### 3.1 Traditional programming

In traditional rule-based programming, a developer writes the rules
that determine the program's behavior.

A simplified spam detector might use rules such as:

- If an email contains a known suspicious phrase, increase its spam score.
- If an email comes from a blocked sender, classify it as spam.
- Otherwise, classify it as legitimate.

The program follows the rules supplied by the developer.

### 3.2 Machine Learning

Machine Learning is a field of AI in which algorithms learn patterns
from data and use those patterns to make predictions or decisions.

Instead of writing every decision rule manually, we provide examples
and a learning algorithm.

For example, a spam classifier may be trained using thousands of
emails labelled as spam or legitimate.

During training, the algorithm adjusts its internal parameters to
reduce prediction errors on the training objective.

After training, the model can predict labels for new emails.

### 3.3 Main difference

Traditional programming:

    Manually written rules + input -> output

Supervised machine learning:

    Training examples + learning algorithm -> trained model

    Trained model + new input -> prediction

These are simplified representations. Real machine learning systems
often involve preprocessing, feature representations, optimization,
validation, and post-processing.

### 3.4 Where machine learning is useful

Machine learning is useful when patterns are too complex, numerous,
or changing to describe reliably with hand-written rules.

Examples include:

- Predicting customer churn
- Detecting fraudulent transactions
- Recognizing speech
- Ranking search results
- Classifying documents
- Recommending products

### 3.5 Limitations

A machine learning model is not automatically correct.

Its behavior depends on factors such as:

- The quality and representativeness of training data
- The learning algorithm
- The model's capacity
- The training objective
- The evaluation process
- Changes in the real-world data distribution

A model can perform well on familiar examples and poorly on new cases.

---

## 4. What is Deep Learning?

Deep Learning is a subfield of Machine Learning that uses neural
networks with multiple layers of learned representations.

A neural network contains connected computational units. During
training, the model adjusts numerical parameters so that its outputs
better match the training objective.

The term "deep" refers to the use of multiple layers of computation.

Deep learning has been especially influential in:

- Image recognition
- Speech recognition
- Machine translation
- Text generation
- Multimodal AI
- Modern language models

### 4.1 What is a neural network?

A neural network transforms numerical inputs through a sequence of
learned operations.

A simplified view is:

    Input -> Layer 1 -> Layer 2 -> Layer 3 -> Output

Each layer processes information and produces a representation for
the next layer.

During training, an optimization algorithm updates model parameters.
The details depend on the architecture and training method.

### 4.2 Deep Learning versus Machine Learning

Machine Learning is the broader category.

Deep Learning is one approach within Machine Learning.

Traditional ML can use methods such as decision trees, linear models,
and support vector machines. Deep learning uses multilayer neural
networks.

Deep learning often benefits from large datasets and substantial
computational resources, although requirements vary by task and model.

---

## 5. What is Natural Language Processing (NLP)?

Natural Language Processing is the area of AI concerned with
processing and working with human language.

Language can be written or spoken.

NLP tasks include:

- Text classification
- Sentiment analysis
- Named entity recognition
- Language translation
- Question answering
- Summarization
- Information extraction
- Text search

### Example: sentiment analysis

Input:

    "The product works well, and I am very happy with it."

Output:

    Positive sentiment

A sentiment classifier attempts to identify the sentiment expressed
by the text.

### Example: named entity recognition

Input:

    "An employee joined Wissen in Bengaluru."

Possible extracted entities:

    Wissen -> Organization
    Bengaluru -> Location

The precise output depends on the model and its entity-label schema.

### NLP and Deep Learning

NLP existed before modern deep learning. Earlier systems used
hand-written rules and statistical techniques.

Modern NLP frequently uses neural networks and transformer models.

Therefore, NLP describes the language-processing problem area, while
deep learning describes one family of methods used to solve it.

---

## 6. What is Generative AI?

Generative AI refers to AI systems that generate new content.

Depending on the model, generated content may include:

- Text
- Images
- Audio
- Video
- Music
- Code
- Structured data

For example, a generative text model can create an explanation,
draft an email, summarize a document, or produce a Python function.

### Generative AI versus classification

A classification model may receive a message and return:

    Category: Technical Support

A generative model may receive the same message and produce:

    This request concerns a technical support issue. The user is
    reporting a problem and needs troubleshooting guidance.

The first output selects a category. The second generates text.

This distinction is illustrative. Generative models can also perform
classification, and classifiers can return more than a single label.

### Does generative AI always produce correct content?

No.

A generated response may be fluent, convincing, incomplete, outdated,
or incorrect. The quality depends on the task, model, available
information, and generation process.

---

## 7. What is a Large Language Model (LLM)?

A Large Language Model is a language model with a large number of
learned parameters, trained to model patterns in language and perform
language-related tasks.

Many modern LLMs use the Transformer architecture.

Examples of tasks they can perform include:

- Answering questions
- Summarizing text
- Generating code
- Explaining technical concepts
- Extracting information
- Rewriting content
- Following instructions

Different models have different capabilities, training procedures,
context limits, costs, and performance characteristics.

### 7.1 What is a token?

A token is a unit of text processed by a language model.

A token may correspond to:

- A complete word
- Part of a word
- Punctuation
- A space-associated text fragment
- Another unit defined by the tokenizer

Tokens are not necessarily equal to words.

For example, a long technical identifier may be split into several
tokens. Tokenization depends on the tokenizer used by the model.

### 7.2 What are model parameters?

Parameters are learned numerical values inside a model.

During training, the model's parameters are adjusted according to
the training objective.

The number of parameters is one aspect of model size, but it does
not alone determine the quality or suitability of a model.

### 7.3 How does an autoregressive LLM generate text?

Many widely used LLMs generate text autoregressively.

At a simplified level:

1. The model receives input tokens.
2. It processes the available context.
3. It calculates a probability distribution over possible next tokens.
4. A decoding strategy selects the next token.
5. The selected token is added to the context.
6. The process repeats until a stopping condition is reached.

For example, given:

    The capital of France is

A model may assign a high probability to the token "Paris".

The exact token sequence and probability distribution depend on
the model and tokenizer.

The model does not need to search the internet to generate this
answer. It may generate it using patterns learned during training.

### 7.4 What is a Transformer?

A Transformer is a neural network architecture widely used in
modern language models.

One of its important mechanisms is attention, which helps the model
process relationships between elements of the input sequence.

A detailed study of tokenization, embeddings, attention, and
transformer internals will come in later lessons.

For now, remember:

    Text -> Tokenization -> Neural network processing
         -> Next-token probabilities -> Generated text

This is a simplified conceptual pipeline, not a complete description
of every modern model architecture.

---

## 8. Training versus inference

These terms are essential when working with LLM applications.

### Training

Training is the process of adjusting model parameters using data
and a learning objective.

For a language model, training may involve predicting tokens from
training sequences and updating parameters to improve the objective.

Training can require large datasets and substantial computing
resources.

### Inference

Inference is the process of using an already trained model to produce
an output for a given input.

When you send a question to an LLM and receive a response, that is
typically inference.

### Comparison

Training:
- Learns model parameters.
- Uses a training procedure and training data.
- Updates parameters during optimization.

Inference:
- Uses the trained model.
- Processes new input.
- Produces predictions or generated content.
- Usually does not update the model's learned parameters.

A RAG application normally uses a trained model during inference.
It retrieves external evidence and includes that evidence in the
model's input. This is different from retraining the LLM.

---

## 9. Why can an LLM produce incorrect answers?

An LLM is not a guaranteed database of verified facts.

Potential reasons for incorrect answers include:

1. The required information was not learned during training.
2. The information is outdated.
3. The prompt is ambiguous.
4. The model misinterprets the question.
5. The model generates a plausible but unsupported statement.
6. The input exceeds or poorly uses the available context.
7. The task requires precise evidence that the model does not have.
8. The model combines related facts incorrectly.

A model can produce confident wording even when the answer is wrong.

This is why important outputs should be evaluated and checked against
reliable evidence.

---

## 10. How does RAG fit into this picture?

Retrieval-Augmented Generation combines information retrieval with
language generation.

A basic RAG application has two workflows.

### 10.1 Indexing workflow

    Documents
        |
        v
    Extract and clean text
        |
        v
    Split text into chunks
        |
        v
    Convert chunks into embeddings
        |
        v
    Store vectors and associated document information

This workflow prepares the knowledge base before a question is asked.

### 10.2 Query workflow

    User question
        |
        v
    Convert question into an embedding
        |
        v
    Retrieve relevant document chunks
        |
        v
    Put the question and evidence into a prompt
        |
        v
    LLM generates an answer
        |
        v
    Return the answer and source references

RAG does not guarantee correctness. Retrieval can return the wrong
passages, documents can be incomplete, and an LLM can misinterpret
the evidence.

The system therefore needs testing and evaluation.

### Example

Suppose a company's leave policy says:

    Employees may carry forward up to five unused leave days.

A general-purpose LLM may not know this private company policy.

A RAG system can retrieve the relevant policy passage and provide it
to the model as evidence.

The model can then answer the question using that passage.

The key difference is that the answer can be grounded in the
provided company document rather than relying only on the model's
previously learned knowledge.

---

## 11. Important distinctions to remember

| Concept | Meaning |
|---|---|
| AI | Broad field of intelligent computational systems |
| ML | Methods that learn patterns from data |
| Deep Learning | ML using multilayer neural networks |
| NLP | Processing and understanding human language |
| Generative AI | Systems that generate content |
| LLM | A language model with large learned parameter sets |
| Training | Learning or adjusting model parameters |
| Inference | Using a trained model to produce outputs |
| RAG | Retrieving external information and using it during generation |

These categories overlap. They are not seven competing technologies.

For example, an LLM may use deep learning, belong to generative AI,
and perform NLP tasks. A RAG application can use that LLM at inference
time alongside a retrieval system.

---

## 12. Common misconceptions

### Misconception 1: AI and ML are the same thing

Correction: ML is one area within AI. Not every AI system learns
from examples.

### Misconception 2: Every NLP system is an LLM

Correction: NLP includes rule-based, statistical, and neural methods,
including many systems that are not LLMs.

### Misconception 3: A larger model is always better

Correction: Model quality depends on the task, evaluation, latency,
cost, context limits, and other requirements.

### Misconception 4: An LLM always knows the latest information

Correction: A model's learned knowledge may be incomplete or outdated.
Some applications provide current information through retrieval or tools.

### Misconception 5: RAG eliminates hallucinations

Correction: RAG can improve grounding, but retrieval and generation
can still fail.

### Misconception 6: RAG and fine-tuning are the same

Correction: RAG supplies external evidence at inference time.
Fine-tuning changes model parameters through an additional training
process. They can be combined when appropriate.

---

## 13. Interview questions

1. What is Artificial Intelligence?
2. How is Machine Learning different from traditional programming?
3. What is Deep Learning?
4. What is Natural Language Processing?
5. What is Generative AI?
6. What is an LLM?
7. What is the difference between training and inference?
8. What is a token?
9. How does an autoregressive LLM generate text?
10. Why can an LLM hallucinate?
11. What is RAG, and why is it useful?
12. How is RAG different from fine-tuning?
13. Does RAG guarantee a correct answer? Why or why not?
14. Why should a RAG system return source references?

Write your own answers before reading external material or asking for
the answers. We will review them together.

---

## 14. Lesson completion checklist

- [ ] I can explain AI, ML, and Deep Learning.
- [ ] I can distinguish NLP from Generative AI.
- [ ] I can explain what an LLM does.
- [ ] I understand tokens and model parameters at a conceptual level.
- [ ] I can distinguish training from inference.
- [ ] I understand why an LLM can produce incorrect answers.
- [ ] I can explain the two main RAG workflows.
- [ ] I can explain how retrieved evidence can support an answer.

Do not move ahead simply because you have read the lesson. Complete
the exercises and explain the ideas in your own words first.
