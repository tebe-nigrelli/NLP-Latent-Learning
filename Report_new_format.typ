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
= Motivation

// TODO introduce SOTA as keyword since it is used later

= Dataset

Google-research's `goemotions` dataset contains 58k samples with a true/false indicator over each emotion from a fixed set of 27 cases, in addition to a "Neutral" label @demszky2020goemotions. The dataset originated from Reddit comments, labelled manually for the highest quality. We picked it because of its relatively large size and reasonable coverage of the emotional spectrum. In addition, it has a simple structure and intuitive labelling. More precisely, we used the Huggingface version for all our tasks, because it contains the canonical _train-test-validate_ split, allowing us to compare classification results with publicly available SOTA models.


Moreover, since it was developed as a training reference for emotion classification models, we expect it to contain enough information to encode nuances in meaning, which a pre-trained text embedding model can learn. We do not verify this claim; we assume the dataset induces sufficient coverage of emotion in English writing, and use it for all our experiments. We do not verify these claims, but we expect the trends identified in this work to hold at a larger scale, even with better quality data.

*Exploration* 
Among all the phrases present in the dataset, the label that appears most frequently is "neutral" with 17772 samples, followed by "admiration" with 5122 samples, while the least common is "grief" with 96 samples.
In the plot A of figure @fig:plots_EDA, it is shown the distribution of the labels in the dataset (removing the label neutral in order to have a more readable plot).

Among all the 54263 input texts, the 45446 of them (around 83.75%) of them is one-hot-encoded (only 1 label), while only one input has 5 labels (highest number of labels for a single input). The distribution of the number of labels per input is shown in the plot B of figure @fig:plots_EDA with logarithmic scale.


The number of words between the input text ranges from 1 to 33. In particular, the distribution is almost symmetrical as the mean is 12.8 and the median is 12. The distribution of the number of words in the text column is shown in plot C of figure @fig:plots_EDA.


Regarding the number of characters in each input, the distribution is concentrated mostly between 2 and 180 and is almost uniform; however, there are some outliers with a high number of characters (up to 703). 
The distribution is slightly left-skewed as the mean is 68.3 and the median is 65.
The plot of the distribution is visible in plot D of figure @fig:plots_EDA in Appendix, with logarithmic scale.


= Experiments and results


For this project, we considered T5 model as encoder-decoder architecture and, in order to investigate the latent space, we inserted a Factor VAE between the two parts. Introducing the VAE bottleneck compresses the representation stored in the latent space after the encoder and encourages a more organized latent space.

The input sequences pass the T5 encoder and the Factor VAE compressing the latent representation from a 786 dimensional space to a 32 dimensional latent space producing a sequence of hidden states. 
From these hidden states we can extract the part representing the emotions and, from it, we can use an attention pooling strategy (either one attention head per emotion label or a single joint attention head) and an MLP classifier head to predict the emotion label for each input.


The model continues with the decoder of the Factor VAE and of the T5 wich receive the latent representation and produces a reconstruction of the original representation of the input sequence. 
Finally, the output of the T5 decoder is used to produce the translated sentence with the target emotion.
The complete scheme of the architecture is shown in figure @fig:architecture_1 of the Appendix.


From this first architecture, the performance of the model was not satisfactory: the emotion classification head produced good results similar to BERT level, but the disentanglement, the recosntruction and the translation were not good.




*First updates: changing the transmission of information* 
In order to improve the performance of the model, we made some adjustments to the architecture. The first one was to change the way the information is transmitted from the encoder to the decoder: instead of using only the latent representation produced by the FactorVAE, we concatenated it with the original representation produced by the T5 encoder using a skip connection from the encoder hidden layer to the decoder memory. This way we give more information to the decoder and we make it easier for it to reconstruct the original input and to produce a good translation. 
To guarantee that the content flows not only on the skip connection but also through the latent space, we added a further bottleneck in the skip connection to force the model to use the latent space for the emotion classification and for the translation.
However, this approach did not lead to a significant improvement in performance: while the emotion classification head still produced good results, the disentanglement, the reconstruction and the translation still failed.

Then, our second modification in order to improve reconstruction and translation was to train the decoder of the T5 model. In particular, due to the limitated computational resources, we decided to train it using LoRA approach. Furthermore, to increase the ability of the model to reconstruct the original input, we added a Copy Loss to the training objective, which encourages the model to copy the input sequence in the output.
This second update led to a significant improvement in the performance of the model: the reconstruction improved significantly and works correctly, while the translation is still not good. The emotion classification head still produces good results, but the disentanglement is still not good.

The scheme of the architecture after the first two updates is shown in figure @fig:architecture_2 in the Appendix.


*Second update: ADVERSARIAL MLP and NEW APPROACH*






= Limitations and Future Work

= Conclusion



#bibliography("sources.bib")


#pagebreak()

= Appendix

*EDA plots*
#figure(
  image("/0. Exploratory/plots/image4.png", width: 100%),
  caption: [*Plot A*: Distribution of the lables in the dataset (except "neutral"); *Plot B*: Distribution of the number of labels per input; *Plot C*: Distribution of the number of words in the inputs; *Plot D*: Distribution of the number of characters in each input.],
) <fig:plots_EDA>


*Architecture schemes*
#figure(
  image("/architecture_plots/arch 1.png", width: 100%),
  caption: [Base architecture: T5 encoder-decoder model with Factor VAE],
) <fig:architecture_1>


#figure(
  image("/architecture_plots/arch 2.png", width: 100%),
  caption: [Updated architecture: T5 encoder-decoder model with Factor VAE and improved information flow],
) <fig:architecture_2>
