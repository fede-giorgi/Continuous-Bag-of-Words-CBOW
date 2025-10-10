# Continuous Bag-of-Words (CBOW) with Negative Sampling  
---

## Overview
This project implements the **CBOW (Continuous Bag-of-Words)** architecture from the **Word2Vec** family of models, trained using **negative sampling** to efficiently learn dense word embeddings from large corpora. The implementation is written in **PyTorch** and trained on the **text8** Wikipedia corpus (≈100 million characters). The model predicts a target (center) word given its surrounding context and learns meaningful distributed representations that capture semantic and syntactic relationships between words.  

---

## Background
In the CBOW formulation, the goal is to predict a **center word** $w_o$ from its **context words** $w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}$:

$$
p(w_o \mid w_{\text{context}}) =
\frac{\exp(b_{w_o}^\top a_{\text{avg}})}
{\sum_{w \in V} \exp(b_w^\top a_{\text{avg}})}
$$

where $A$ and $B$ are the **input** and **output** embedding matrices, and $a_{\text{avg}}$ is the mean of the context embeddings.  
To avoid the $O(|V|)$ cost of the softmax denominator, this probability is approximated using **negative sampling**:

$$
L = -\log \sigma(b_{w_o}^\top a_{\text{avg}}) -
\sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} \log \sigma(-b_{w_k}^\top a_{\text{avg}})
$$

This objective encourages the model to assign high similarity to true context–target pairs and low similarity to randomly sampled (negative) pairs, enabling efficient training even with large vocabularies.  

---

## Implementation
The project begins by reading and preprocessing the text8 corpus, lowercasing the text, removing punctuation, and splitting it into tokens. Rare words are filtered out, and very frequent words are subsampled according to  

$$
p_{\text{keep}}(w) = \left(\sqrt{\frac{z(w)}{0.001}} + 1\right)\frac{0.0001}{z(w)}
$$

where $z(w)$ is the empirical frequency of word $w$.  

A custom `Vocab` class constructs bidirectional mappings between words and integers (`stoi` and `itos`) and rebuilds the vocabulary after subsampling. Training samples are generated using a fixed context window $(m = 1)$, producing triplets of the form `[left_context, right_context, center_word]`. Each sample is represented by integer indices and grouped into batches for training.  

The neural network `CBOWNegativeSampling` defines two embedding matrices:  
`A` for context embeddings and `B` for target embeddings. For each batch, the model computes the average of the two context embeddings, takes the dot product with the corresponding target embedding, and applies the sigmoid activation to produce logits. The loss combines the positive and negative components through the numerically stable `softplus` formulation.  

Training is performed with **stochastic gradient descent** (learning rate 10.0, batch size 512, embedding dimension 300, negative samples K=4). Gradients are clipped to 0.1 to prevent explosion, and a step-decay scheduler halves the learning rate every three epochs. Periodically, the model evaluates the learned representations by printing the most similar words to a fixed set of probe terms such as “money”, “lion”, “africa”, “musician”, and “dance”.  

---

## Results
After ten epochs, the model converges to a stable representation with average loss around **0.94**. The nearest-neighbor queries clearly reveal semantic and syntactic clusters, confirming that the embeddings capture meaningful linguistic relationships. The final validation output is shown below:

| epoch 10 | 32500/32580 batches | loss 0.942

- money: goods, taxes, prices, payment, buying, millions, funds, compensation, gift, cash

- lion: sands, bearded, saguinus, crab, goat, swan, alder, serpent, eagle, calf

- africa: korea, borneo, albania, eurasia, mozambique, yemen, americas, africans, sumatra, morocco

- musician: singer, guitarist, pianist, drummer, comedian, songwriter, dancer, songwriters, bassist, singers

- dance: dancing, jazz, blues, dances, ballroom, pop, bluegrass, rap, swing, techno

The model successfully groups words according to their semantic similarity: “money” relates to economic terms, “lion” to animals, “africa” to countries and regions, and “musician” and “dance” to artistic domains.  

These results empirically confirm the **distributional hypothesis**: that words occurring in similar contexts acquire similar meanings in vector space. Despite its simplicity, the CBOW model with negative sampling efficiently captures meaningful structure in natural language, producing robust embeddings that align closely with human linguistic intuition.  


