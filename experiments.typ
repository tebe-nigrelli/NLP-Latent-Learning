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


