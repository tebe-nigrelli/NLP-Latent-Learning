#set document(
  title: [Evaluating Transformer Embedding  Disentanglement using Emotion Translation
  ],
)

#set heading(numbering: "1")

#set page(
  paper: "a4",
  margin: (
    left: 6em,
    right: 6em,
    top: 5em,
    bottom: 5em,
  ),
  numbering: "1",
  columns: 2
)

#place(
  top + center,
  scope: "parent",
  float: true,
)[
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
      by first mapping tokens to a sequence of encoder hidden states, and then remapping them
      autoregressively to a new sequence.
      2 Without additional inductive biases, latent spaces suffer from a
      lack of interpretability, despite having a complete encoding of information.
      3 Most models circumvent this issue by using a classification head on top of the encoder, which is trained to extract the relevant information for the task at hand: however, this does not necessarily lead to a disentangled representation. 
      4 We investigate whether such
      hidden representations can be refined and disentangled by a Variational Autoencoder (VAE).
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
]

= Introduction

Transformer-based encoder–decoder architectures have become a central paradigm for sequence-to-sequence learning, extending earlier neural encoder–decoder models by using attention mechanisms to map an input sequence into contextual hidden states and then autoregressively decode an output sequence. //(Sutskever et al., 2014; Bahdanau et al., 2015; Vaswani et al., 2017)
While these hidden representations often contain rich task-relevant information, they are not necessarily interpretable: without explicit structural constraints, the latent factors encoded by the model remain entangled across dimensions, making it difficult to identify which components correspond to semantic, stylistic, or affective properties of the input. This issue is particularly important for controllable text generation and style transfer, where prior work has shown that successful attribute manipulation requires separating content from controllable attributes such as sentiment, tense, or style  //(Hu et al., 2017; Shen et al., 2017). 
Standard supervised approaches often address downstream prediction by adding a classification head on top of the encoder, but such heads are trained only to extract task-discriminative information and do not guarantee that the underlying representation is factorized or disentangled. 
Variational Autoencoders provide a natural framework for imposing structure on latent spaces through probabilistic encoding and reconstruction objectives, // (Kingma & Welling, 2014), 
while later variants such as β-VAE and FactorVAE explicitly encourage more interpretable and independent latent factors. // (Higgins et al., 2017; Kim & Mnih, 2018).
 Building on this line of work, we investigate whether the hidden representations of a Transformer encoder–decoder model can be refined through a FactorVAE objective so that latent dimensions become smoother, less correlated, and more useful for controlled generation. 
 To have a concrete testbed, we ground our analysis in the domain of emotion classification and translation. Specifically, we reserve a subset of latent dimensions for emotion-related information and introduce an attention pooling mechanism that learns a direct mapping between these dimensions and emotion labels. Thus, we build a sequence-to-sequence model that preserves input content while modifying the affective attribute of the generated output. *MODIFY LAST STATEMENT, THIS IS A LIE*

 *PLACEHOLDER* results and contributions

= Experiments

== Dataset

Google-research's `goemotions` dataset contains 54k samples with a true/false indicator over each emotion from a fixed set of 27 cases, in addition to a "Neutral" label @demszky2020goemotions. The dataset originated from English Reddit comments, labelled manually for the highest quality. We selected it because of its relatively large size and reasonable coverage of the emotional spectrum; in addition, it has a simple structure and intuitive labelling. We used the Huggingface version for all our tasks, because it contains the canonical _train-test-validate_ split, allowing us to compare classification results with publicly available SOTA models.

// TODO insert classifier comparison to the SOTA classifiers and link as bib citation in the comparison table

Moreover, since it was developed as a training reference for emotion classification models, we expect it to contain enough information to encode nuances in meaning, which a pre-trained text embedding model can learn. We do not verify this claim; we assume the dataset induces sufficient coverage of emotion in English writing, and use it for all our experiments. 

*Exploration* Among all the phrases present in the dataset, the label that appears most times is "neutral" with 17772 samples, followed by "admiration" with 5122 samples, while the least common is "grief" with 96 samples.
In the plot A of figure @fig:plots_EDA, it is shown the distribution of the labels in the dataset (removing the lable neutral in order to have a more readable plot).

Among all the documents, 83.75% of them are one-hot-encoded (only 1 label), while only one input has 5 labels (highest number of labels for a single input). The distribution of the number of labels per input is shown in the plot B of figure @fig:plots_EDA with logarithmic scale.

*DO WE WANT TO PUT THIS?*
The input with 5 labels is "Yeah I probably would've started crying on the spot. Loud, sudden and especially shrill noises are extremely \*"cringey"\* and uncomfortable and stressful" and the associated labels are "curiosity", "disapproval", "embarrassment", "joy", "relief".


The number of words between the input text ranges from 1 to 33. In particular, the distribution is almost symmetrical as the mean is 12.8 and the median is 12.0 with a standard deviation of 6.70. The first and third quartiles are 7.0 and 18.0 respectively, confirming the relative symmetry of the distribution. The distribution of the number of words in the text column is shown in plot C of figure @fig:plots_EDA.


Regarding the number of characters in each input, the distribution is concentrated mostly between 2 and 180 and is almost uniform; however, there are some outliers with a high number of characters (up to 703). 
The distribution is slightly left-skewed as the mean is 68.3 and the median is 65.0 with a standard deviation of 36.7.

The plot of the distribution is visible in plot D of figure @fig:plots_EDA, where the logarithmic scale is used to make the plot more readable.


#figure(
  image("/0. Exploratory/plots/image4.png", width: 100%),
  caption: [Plot A: Distribution of the lables in the dataset (removing the label "neutral"); Plot B: Distribution of the number of labels per input (in logarithmic scale); Plot C: Distribution of the number of words in the inpouts; Plot D: Distribution of the number of characters in the text column (in logarithmic scale).],
) <fig:plots_EDA>


== Model

The model uses a T5 encoder-decoder as a controllable text-generation backbone. The T5 encoder maps each input sentence into token-level hidden states: a variational bottleneck then compresses each hidden state into a latent representation. The latent representation separates into scalar emotion factors and vector content factors. The scalar factors encode emotion signals, while the vector factors preserve semantic content. An attention-based classifier reads the scalar factors and predicts GoEmotions labels. A FactorVAE discriminator penalizes dependence between latent dimensions and encourages disentanglement. Adversarial classifiers further remove emotion information from the vector and residual pathways. The T5 decoder reconstructs text from the modified latent states. Latent editing changes selected scalar factors, so the model can shift the target emotion while retaining the original meaning.

The T5 backbone encoder maps input sentences into a 512-dimensional latent space. Our latent bottleneck adds 28 dimensions, assigning them to scalar emotion factors, and reserves 512 dimensions to vector content factors. The scalar factors match the 28 GoEmotions labels. The training setup freezes most T5 parameters and trains LoRA adapters on the q and v attention projections. The LoRA configuration uses rank 16, alpha 32, and dropout 0.1. The objective combines emotion classification, hidden-state reconstruction, KL regularization, total-correlation regularization, copy loss, and adversarial losses. The schedule delays harder objectives. *IS THIS TRUE IN THE END?* The model first learns classification and reconstruction, then activates LoRA, KL, total-correlation, and copy losses. This staged training choice stabilizes the latent space before the model learns stronger disentanglement and generation constraints.

== Task 1: Emotion Classification

The model performs emotion classification from the scalar latent factors, not from the full T5 hidden state. An attention-pooling module reads token-level scalar latents and selects the tokens that carry the strongest evidence for each emotion. A per-emotion classifier then predicts multilabel emotion scores with a sigmoid output. This design forces the classifier to use the intended emotion subspace and supports later latent editing. The evaluation uses multilabel metrics because each sentence can express more than one emotion: the main used metrics include micro-F1, macro-F1, weighted F1, hamming accuracy, average precision, and label-ranking average precision. Micro-F1 measures overall label prediction quality and favors frequent labels. Macro-F1 gives each emotion equal weight and therefore exposes poor performance on rare labels. This distinction matters for GoEmotions, because labels such as neutral dominate the dataset while labels such as grief appear rarely. 

== Task 2: Emotion Translation

The model performs emotion translation by editing the scalar emotion factors while keeping the vector content factors fixed. The T5 encoder first maps the source sentence into token-level hidden states. The variational bottleneck then separates these states into emotion and content latents. The editing procedure changes the scalar factors toward a target GoEmotions label and preserves the vector factors as the content anchor. The T5 decoder then generates a new sentence from the edited latent representation. This design makes emotion translation a controlled intervention in the latent space rather than a direct prompt-based rewrite. The evaluation checks two goals at once: target-emotion success and meaning preservation. Target-emotion success measures whether the rewritten sentence expresses the requested emotion.*PLACEHOLDER*
 Meaning preservation measures whether the rewrite keeps the original topic, entities, and factual content. For this purpose, we used a combination of BERTScore and masked semantic similarity. The first approach measures token-to-token similarity, while the second copmutes a sentence-level similarity, removing emotion-bearing words so the metric does not punish the model for changing the intended emotional tone. For the masking we use NRCLex, a lexicon of emotion-bearing words *add reference*. 

= Results

== Classification

== Emotion Translation

= Discussion

= Related Work

= Conclusion

= Limitations and Future Work

#pagebreak()

#bibliography("sources.bib")


#pagebreak()