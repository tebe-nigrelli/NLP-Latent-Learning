


#set document(
  title: [Evaluating Transformer Embedding  Disentanglement using Emotion Translation
  ],
)

#set page(
  paper: "a4",
  margin: (
    left: 5em,
    right: 6em,
    top: 4em,
    bottom: 5em,
  ),
)

#place(
  top + center,
  float: true,
  scope: "parent",
)[
  #align(center)[
    #title()

    #grid(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr),
      gutter: 0.5em,
      align: center,
      [*Tebe Nigrelli*], [*Federico Pezzoli*], [*Alessandro Reali*], [*Luca Ricci*], [*Ali Emre Senel*],
    )

    #line(length: 100%)
  ]
]

#set par(justify: true)
#set heading(numbering: "1.a.")

Encoder-Decoder models based on the transformer are capable of performing a wide range of tasks by first mapping tokens to a sequence of embeddings, before remapping them autoregressively to a new sequence. Without additional inductive biases, embedding spaces suffer from a lack of interpretability, despite having a complete encoding of information. In this work, we investigate whether such embeddings can be refined and disentangled by a Variational Autoencoder (VAE). To do so, we train the VAE to reproduce the original sequence through an information bottleneck. We use a factor-VAE architecture to decorrelate the latent dimensions, which helps in smoothing the representation, but we also include task-reserved dimensions for performing emotion-translating tasks. Particularly, we use an attention pooling component to learn a two-way mapping between some dimensions of the latents and the emotion labels, producing an emotion-translating sequence-to-sequence model. We evaluate our model using human-guided LLM labelling, which complements our reconstruction error in the translation from embedding and latent representations.

#grid(
  align: left + horizon,
  columns: (1fr, 1fr),
  column-gutter: 1em,
  [
    In the architecture, we place the VAE between the encoder and decoder, forcing it to map token embedding to an interpretable latent space and then back to the original encoding. We use a VAE because it allows for new sample generation, and it can be set to enforce strong inductive biases in the latent space through a custom loss. More precisely, we set the number of latent dimensions to be half of the original embedding dimension, reserving $m$ dimensions for encoding emotion. We then use an attention pooling block to aggregate the latent representations of the whole sequence to quantify the overall emotional tone of the sentence. During training, VAE first learns to compress and reconstruct the original token embeddings, while the attention pooling mechanism trains to correctly identify emotion. At evaluation, the VAE can reconstruct and classify input sequences internally, and the attention pooling mechanism can be inverted to alter the emotion of a document and obtain a new tone.
  ],

  [
    #image("img/MAI Y42 NLP Project.png", width: 120%)
  ],
)

We use a T5 model as the embedding backbone because it is an encoder-decoder model and its general pre-training makes the embedding space better suited for general purpose tasks. We use two human-labelled datasets for training: GoEmotions and SemEval-2018 Task 1: Affect in Tweets (EI-reg). The first provides 58k Reddit comments annotated with 27 fine-grained emotion categories plus Neutral, for categorical emotion detection; the latter contributes tweet-level real-valued emotion intensity scores for anger, fear, joy, and sadness, which lets us model not only which emotion is present but also how strongly it is expressed.

For evaluation, we quantify two relevant metrics to the translation process: emotional accuracy and meaning preservation. We use human-guided LLM labelling for both metrics: we manually label some samples (in the order of \~100), and tweak temperature and distribution parameters of the model to obtain human-like responses on our complete dataset. We fix the softmax probability values of the emotion classifier, treating them as the ground truth of the world. Finally, we can compare performance of the model against a human baseline, comparing the emotion-translating task performance of human volunteers against our model. 
