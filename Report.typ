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
      TODO ABSTRACT
    ]
  ]
]
= Introduction


= Dataset

Google-research's `goemotions` dataset contains 58k samples with a true/false indicator over each emotion from a fixed set of 27 cases, in addition to a "Neutral" label @demszky2020goemotions. The dataset originated from Reddit comments, labelled manually for the highest quality. We picked it because of its relatively large size and reasonable coverage of the emotional spectrum. In addition, it has a simple structure and intuitive labelling. More precisely, we used the Huggingface version for all our tasks, because it contains the canonical _train-test-validate_ split, allowing us to compare classification results with publicly available SOTA models.

// TODO insert classifier comparison to the SOTA classifiers and link as bib citation in the comparison table

Moreover, since it was developed as a training reference for emotion classification models, we expect it to contain enough information to encode nuances in meaning, which a pre-trained text embedding model can learn. We do not verify this claim; we assume the dataset induces sufficient coverage of emotion in English writing, and use it for all our experiments. We do not verify these claims, but we expect the trends identified in this work to hold at a larger scale, even with better quality data.

*Exploration* 
Among all the phrases present in the dataset, most frequent label is "neutral" with 17772 samples, followed by "admiration" with 5122 samples, while the least common is "grief" with 96 samples.
Plot A of figure @fig:plots_EDA in Appendix, it is shows the distribution of the labels in the dataset (removing label "neutral").

Among all the 54263 input texts, around 83.75% of them is one-hot-encoded, while the highest number of labels is 5. The distribution of the number of labels per input is shown in the plot B of figure @fig:plots_EDA with logarithmic scale.

The number of words of the input text ranges from 1 to 33. In particular, the distribution is almost symmetrical as the mean is 12.8 and the median is 12. The distribution of the number of words in the text column is shown in plot C of figure @fig:plots_EDA.


Regarding the number of characters in each input, the distribution is concentrated mostly between 2 and 180 and is almost uniform; however, there are some outliers with a high number of characters (up to 703). 
The distribution is slightly left-skewed as the mean is 68.3 and the median is 65.
The plot of the distribution is visible in plot D of figure @fig:plots_EDA, whith logarithmic scale.


= Experiments and Results

For the experiments we focus on a T5 based encoder-decoder, where we we insert a VAE bottleneck between the encoder and the decoder to compress the representation stored in the latent space after the encoder and encourage a more organized latent space.

*Starting architecture* The input of the model are the sentences of the two datasets with the emotions labels and, after tokenizing it with the T5 tokenizer, we pass it through the T5 encoder. The T5 encoder receives the tokenized input and produces a sequence of contextualized hidden states; however, this representation is not disentangled and meaning, emotions carried, syntax and all the other information are all mixed together in the hidden states.

After that we have te full sequence of embedding tokens, we need a compact size representation to be able to perform the tasks. For this, our model uses attention pooling and, in particular, we used two different strategies: either we have one attention head for each emotion label and we pool the sequence of hidden states into a vector of size equal to the number of labels, or we have a unique joint attention head that pools the sequence into a single vector. 


After attention pooling, the pooled vector is passed to the Factor-VAE encoder which compresses the input latent space into a low dimensional representation passing from a 786 dimensional space to a 32 dimensional latent space. The objective of this is to add an additional strong bottleneck to the model pushing different latent dimensions to be statistically independent penalizing total correlation between each dimension.

From the latent reppresentation, the model branches into a first MLP classifier head predicting emotion label for each input. 

The model continues with the decoder of the VAE wich receives the latent representation and produces a reconstruction of the original representation of the input sequence. 

Finally, the output of the VAE decoder is passed to the T5 decoder which produces the final output reconstructed sequence.

The complete scheme of the architecture is shown in figure @fig:architecture_1 in Appendix.


From this first architecture, the performance of the model was not satisfactory: the emotion classification head produced good results similar to BERT level, but the disentanglement, the recosntruction and the translation were not good.




*Stability and reconstruction updates* 
To stabilize the training of the VAE, the first update we added to the architecture is to add skip connections over the VAE bottleneck. In this way the trainig stability is improved and performance increased as emotion information is carried not only through the latent space but also through the skip connections.
Moreover, to control the contribution of the skip connection, we added an MLP over it to regulate the flow giving more weight to the VAE bottleneck.

With this changes the emotion classification head produces good results and the reconstruction task improved, but still did not reach a good level; instead, the disentanglement and the translation still failed.

In order to improve these tasks and increase the bottleneck around the VAE we added dropout regularization to the skip connections; however, the performance decreased significatly so we removed it.

The sequent passage we made to increase the reconstruction performance was to train the entire model end-to-end. Howver, this apprach was highly inefficient and the preformance deacreased showing catastrophic forgetting of the emotion classification task. To solve these problems we made two adjustments: first, since the reconstruction is performed by the decoder part of the model and the classification is performed after the encoders produce the latent vectors, we froze the encoder, both of the T5 and of the VAE, and decided to train only the decoders; second, to tackle the inefficiency of the training, we decided to add LoRA adapters to the decoder part of the model, so that only a small number of parameters are updated during training.
With this new architecture we solved the problems of the previous version and we reached a good performance both in the emotion classification and in the reconstruction tasks, while the disentanglement and the translation still failed.


* Emotions translation updates* 









= Discussion




= Related Work


= Limitations and Future Work







= Conclusion




#bibliography("sources.bib")


#pagebreak()

= Appendix

*Plot 1* Exploratory Plots of GoEmotions dataset.
#figure(
  image("/0. Exploratory/plots/image4.png", width: 100%),
  caption: [*Plot A*: Distribution of the lables in the dataset (except "neutral"); *Plot B*: Distribution of the number of labels per input; *Plot C*: Distribution of the number of words in the inputs; *Plot D*: Distribution of the number of characters in each input.],
) <fig:plots_EDA>


*Plot 2* Starting architecture of the model used for the experiments.
#figure(
  image("/architecture_plots/arch_1.png", width: 100%),
  caption: [Architecture 1: starting T5-based encoder-decoder model with a VAE inserted between the encoder and the decoder. From emotion latent vectors an MLP is used for emotion classification.],
) <fig:architecture_1>














#pagebreak()


= TO BE DELETED

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
