#set document(
  title: [Evaluating Transformer Embedding  Disentanglement using Emotion Translation
  ],
)

#set heading(numbering: "1.a.")

#set page(
  paper: "a4",
  margin: (
    left: 6em,
    right: 6em,
    top: 5em,
    bottom: 5em,
  ),
  numbering: "1",
)

#align(center)[
  #title()

  #grid(
    columns: (1fr, 1fr, 1.1fr, 1fr, 1fr),
    gutter: 1em,
    align: center,
    [*Tebe Nigrelli*], [*Federico Pezzoli*], [*Alessandro Reali*], [*Luca Ricci*], [*Ali Emre Senel*],
  )

    #line(length: 100%)
  ]

// TODO insert abstract and conclusion into README

#align(center)[
  #par(justify: true)[
    1 Encoder-Decoder models based on the transformer can perform a wide range of tasks
    by first mapping tokens to a sequence of encoder hidden states (embeddings), and then remapping them
    autoregressively to a new sequence.
    2 Without additional inductive biases, embedding spaces suffer from a
    lack of interpretability, despite having a complete encoding of information.
    3 We investigate whether such
    embeddings can be refined and disentangled by a Variational Autoencoder (VAE).
    4 To do so, we train the VAE
    to reproduce the original sequence through an information bottleneck.
    5 We use a factor-VAE architecture
    to decorrelate the latent dimensions, which helps smooth the representation, and we include task-reserved
    dimensions for performing emotion-translating tasks:
    particularly, we use an attention pooling component
    to learn a two-way mapping between some dimensions of the latents and the emotion labels, producing
    an emotion-translating sequence-to-sequence model.
    6 //results
    7 // future implications
  ]
]

= Introduction

// TODO introduce SOTA as keyword since it is used later

== Motivation and Task Relevance

== Methodology

= GoEmotions Dataset

Google-research's `goemotions` dataset contains 58k samples with a true/false indicator over each emotion from a fixed set of 27 cases, in addition to a "Neutral" label @demszky2020goemotions. The dataset originated from Reddit comments, labelled manually for the highest quality. We picked it because of its relatively large size and reasonable coverage of the emotional spectrum. In addition, it has a simple structure and intuitive labelling. More precisely, we used the Huggingface version for all our tasks, because it contains the canonical _train-test-validate_ split, allowing us to compare classification results with publicly available SOTA models.

// TODO insert classifier comparison to the SOTA classifiers and link as bib citation in the comparison table

Moreover, since it was developed as a training reference for emotion classification models, we expect it to contain enough information to encode nuances in meaning, which a pre-trained text embedding model can learn. We do not verify this claim; we assume the dataset induces sufficient coverage of emotion in English writing, and use it for all our experiments. We do not verify these claims, but we expect the trends identified in this work to hold at a larger scale, even with better quality data.

== Exploration

= Architecture

== Backbone Model

== VAE

= Evaluation

== Metrics

== Human-Guided LLM Annotation

== Ablations

= Results

== Emotion Translation Performance

== Meaning Preservation Performance

= Evaluation

// TODO include limitations
// The dataset is in the English language, comes from Reddit
// each LLM has a different pretraining dataset - Data Leakage?

= Conclusion



#bibliography("sources.bib")


#pagebreak()

*What Meaning Preservation Verification Model we pick and why*

Regarding the meaning preservation we think about using a textual entailment model since, after emotion translation, the output should still preserve the propositional content of the source and the problem can be viewed as a textual entailment task. We plan to use DeBERTa-Large-MNLI architecture (https://huggingface.co/microsoft/deberta-large-mnli). Since DeBERTa-Large-MNLI is fine-tuned on the MultiNLI benchmark, which contains over 400k inference pairs across ten textual genres, it is trained to distinguish entailment, neutrality, and contradiction.

In practice, we would compute the model's entailment scores in both directions (source → generated and generated → source), and analyze them; in fact, we would get P1=P(entailment|premise=A, hypothesis=B) , P2=P(entailment|premise=B, hypothesis=A). What we expect is that if both scores are low there is no implication between the original and the generated phrase; if P1 is high and P2 is low we have that the model correctly transmits the meaning of the phrase but forgets some information; if only P2 is high we have that the model transferred the general content but adding some information that was not defined before; if both P1 and P2 are high we have that the model correctly transferred all the information.



*How human-guided LLM labelling works mathematically
*

Mathematically, human-guided LLM annotation can be viewed as a calibration procedure for a labelling function. In particular, given a set of inputs x_i and labels y_i produced by a human, we can tweak the annotator model using a simple method like temperature scaling, which adjusts the confidence without changing label ranking. Alternatively, we can run affine calibration, which also corrects systematic class-specific biases. The prompt, decoding settings, and calibration parameters are selected by minimizing disagreement between LLM annotations and human labels. We can then use comparison scores such as F1, mean squared error (MSE), and cross-entropy (CE) loss to quantify agreement between calibrated LLM annotations and human labels.


*What Encoder-Decoder Model we pick and why*

Regarding the encoder-decoder model for our project, we thought of selecting the T5 model. T5 frames NLP tasks as text-to-text problems by mapping input sequences to a shared latent representation via a standard transformer encoder before autoregressively decoding an output sequence. These steps make it the natural fit for our context that happens exactly at the encoder-decoder interface.
There are other encoder-decoder architectures, most notably BART, but T5's unified text-to-text framing makes its latent space more general-purpose and better studied across a wider range of tasks.
Also, compared to just decoder-only models, T5's encoder produces a well-defined fixed-dimensional sequence representation that can be directly used by our VAE. Compared to encoder-only family architectures (like Bert) instead, it keeps generative capability, which are essential for producing the translated output text in our scenario.
Finally, T5 has different variants that can be integrated in our project based on the necessity: some  examples are mT5 which extends pretraining to 101 languages, and FLAN-T5, which is an instruction-tuned version of T5 which leads to better generalization.


*What Emotion Classification Model we pick and why*

For emotion classification, we thought of selecting emotion-english-distilroberta-base, a DistilRoBERTa model trained on 6 diverse datasets that predicts Ekman's 6 basic emotions plus a neutral class. This compact 7-class taxonomy leads to sharp and consistent softmax probability vectors, providing a clean training signal for our VAE's disentanglement objective and potentially making latent space traversal directly interpretable.

*2 Datasets picked and why*

We pick GoEmotions and SemEval-2018 Task 1: Affect in Tweets (EI-reg) as our two datasets: GoEmotions provides 58k Reddit comments annotated with 27 fine-grained emotion categories plus Neutral, giving us rich, multi-label categorical supervision for learning disentangled, interpretable emotion dimensions in our VAE's latent space; SemEval-2018 EI-reg contributes tweet-level real-valued emotion intensity scores for anger, fear, joy, and sadness (between 0 and 1), which lets us model not only which emotion is present but also how strongly it is expressed, aligning directly with our goal of quantifying emotion strength changes and relating them to meaning preservation in social-media-style text.

*Task Relevance, why it matters*

Emotion translation is a highly relevant task for evaluating latent space disentanglement because it requires modifying one specific and identifiable factor of a text sample, namely its emotional tone, while preserving the underlying propositional content; this makes it an ideal setting for testing whether the latent representation learned by the VAE is truly interpretable and factorized. If the latent space is successfully disentangled, then moving along the emotion-related dimensions should change the expressed emotion in a controlled way without altering the semantic meaning of the sentence, whereas unintended semantic drift would indicate that the representation does not cleanly separate emotion from content. This is important not only from a representation-learning perspective, but also because it grounds disentanglement in a concrete generative task rather than in purely abstract metrics such as latent independence or visualization quality. In our case, success can be measured directly through emotional accuracy and meaning preservation, making the evaluation both rigorous and practically meaningful. More broadly, the task matters because controllable emotion-aware text generation has potential applications in dialogue systems, writing assistance, education, accessibility, entertainment, and affect-sensitive human-computer interaction, all of which benefit from models that are not only effective but also transparent and steerable.
