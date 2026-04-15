= Experiments & Ablations for the Project

= Experiments
1. Reconstruction Fidelity: with T5 frozen check how well the vae reconstruct the hidden states and preserve information.

2. Emotion classification accuracy (\* stand by): after shifting the emotion vector to a target label and decoding back to text, run an emotion classifier on the output and measure hit rate per emotion class.

3. Robustness to emotion ambiguity: test on sentences that have multiple labels and check whether the model produces more uncertain or unstable translations on these.

4. Semantic preservation under translation (\* stand by): measure through some metrics the semantic similarity between the input and the shifted output.

5. Latent space geometry visualization: apply umap separately on the emotion dimensions and meaning dimensions of the latent space.



= Ablations (we want to do)
1. T5 fine-tuning: compare frozen vs fine-tuned T5 model
2. Number of attention pooling heads: check over possible number of heads (1, 2, 4, etc) and plot emotion classification accuracy vs head count to identify plateau point
3. VAE bottleneck size: vary total latent dimensionality relative to T5 hidden size to find the optimal compression
4. Attention Pooling weights (Q^TK): compare using the emotion-only vector, the meaning-only vector, both concatenated, or the raw pre-VAE encoder embeddings as queries/keys, to determine the best option.
5. Attention Pooling Values: compare using only the emotion latent dimensions vs the full VAE latent vector as values.
6. Skip Connection across VAE: add residual connection to measure whether it improves stability.
7. Human guided LLM evaluation


= Ablation (it is not our priority for the moment)
1. Try the vae without the factor (I think it changes disentagled stuff etc)


= Evaluation

- *T5 Finetuning*: Exact Match Rate, Blue Score, Rouge Score. We do not want complete reconstruction, otherwise the model will collapse.
- *Reconstruction metrics*: grid search for coefficients in VAE loss, 
- Disentanglement metrics: DCI and MIG
- *Emotion classification*: entropy, but we need weighted loss in order to have uniformly capable model due to class imbalances.
  - _Multi-hot_: Binary Cross Entropy applied to the softmax activations of each label. Ensures each error in classification has the same unit of measure.
  - _One-hot_: (Weighted) Cross Entropy. 
  - After training, we can check *F1 score* and *(Balanced) Accuracy* and *per-class recall* on the validation dataset.
  - *Plots of error*: (normalized) confusion matrix, 

- *Human-guided LLM assessment*: use ollama {qwen3.5:9b, gemma4:e4b} to answer questions if two phrases match.
  - *Predict Emotion*: finetune, emotion prediction.
    - Get a raw semantic-similarity score from the LLM, then adjust it with embedding cosine and your  human labels.
    - Calibrate that score with a simple regressor, then add a conformal interval for error bounds.
  - *Compare Meaning*: finetune, meaning comparison. Platt scaling
    - Get raw probabilities over emotion labels from the LLM, using fixed label definitions and examples.
    - Calibrate those probabilities on human labels, then output a conformal label set for uncertainty.