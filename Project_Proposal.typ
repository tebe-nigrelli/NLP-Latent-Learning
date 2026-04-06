


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

Transformer based Encoder-Decoder models are capable of performing a wide range of tasks by first mapping sequences to a lower-dimensional representation, before using the encoding to autoregressively continue a new sequence. The highly irregular geometry of latent spaces results in complete, yet uninterpretable representations of information. In this work, we investigate whether such a pathological latent space can still be disentangled, by training a variational autoencoder (VAE) to perform the task.
#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,

  [
    We place the VAE between the encoder and decoder, forcing it to map to an interpretable embedding space then back to the original encoding. We opted for a VAE because it allows for sampling and generation of new samples, but we also explicitly enforce strong inductive biases in the loss, imposing that the first dimensions of the interpretable latent space must correspond to a desired labelling of the text sample, while enforcing low relative correlation in the remaining features. While the VAE will be trained, the encoder-decoder model will be frozen: as backbone, we will use the T5 model, as its unified text-to-text framing makes its latent space more general-purpose and better studied across a wider range of tasks.

  ],

  [
    #image("img/MAI Y42 NLP Project.jpg", width: 120%)
  ],
)

We ground our task with performance at encoding emotion, and we perform latent space traversal to obtain a model that can “translate” text samples from one emotion to another in a directly interpretable way. For this scope, we chose two datasets, GoEmotions and SemEval-2018 Task 1: Affect in Tweets (EI-reg). The first provides 58k Reddit comments annotated with 27 fine-grained emotion categories plus Neutral, for categorical emotion detection; the latter contributes tweet-level real-valued emotion intensity scores for anger, fear, joy, and sadness (between 0 and 1), which lets us model not only which emotion is present but also how strongly it is expressed.

For evaluation, we quantify two relevant metrics to the translation process: emotional accuracy and meaning preservation. We use human-guided LLM labelling for both metrics: we manually label some samples (a few hundred will suffice), and tweak temperature and distribution parameters of the model to obtain human-like responses on our complete dataset. Next, we use the softmax emotion probability values of the classifier as the ground truth for the interpretable representation. Finally, we balance loss between the two metrics, to obtain a model which is both emotion-translating and meaning-preserving. We conclude by providing a law relation between emotion and meaning loss. For this task, we use as base models DistilRoBERTa for emotional accuracy and DeBERTa-Large-MNLI for meaning preservation.
