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

= Motivation

// TODO introduce SOTA as keyword since it is used later

= Dataset

Google-research's `goemotions` dataset contains 58k samples with a true/false indicator over each emotion from a fixed set of 27 cases, in addition to a "Neutral" label @demszky2020goemotions. The dataset originated from Reddit comments, labelled manually for the highest quality. We picked it because of its relatively large size and reasonable coverage of the emotional spectrum. In addition, it has a simple structure and intuitive labelling. More precisely, we used the Huggingface version for all our tasks, because it contains the canonical _train-test-validate_ split, allowing us to compare classification results with publicly available SOTA models.

// TODO insert classifier comparison to the SOTA classifiers and link as bib citation in the comparison table

Moreover, since it was developed as a training reference for emotion classification models, we expect it to contain enough information to encode nuances in meaning, which a pre-trained text embedding model can learn. We do not verify this claim; we assume the dataset induces sufficient coverage of emotion in English writing, and use it for all our experiments. We do not verify these claims, but we expect the trends identified in this work to hold at a larger scale, even with better quality data.

*Exploration* Among all the phrases present in the dataset, the label that appears most times is "neutral" with 17772 samples, followed by "admiration" with 5122 samples, while the least common is "grief" with 96 samples.
In the plot A of figure @fig:plots_EDA, it is shown the distribution of the labels in the dataset (removing the lable neutral in order to have a more readible plot).



Among all the 54263 input texts, the 45446 of them (around 83.75%) of them is one-hot-encoded (only 1 label), while only one input has 5 labels (highest number of labels for a single input). The distribution of the number of labels per input is shown in the plot B of figure @fig:plots_EDA with logarithmic scale.
The input with 5 labels is "Yeah I probably would've started crying on the spot. Loud, sudden and especially shrill noises are extremely \*"cringey"\* and uncomfortable and stressful" and the associated labels are "curiosity", "disapproval", "embarrassment", "joy", "relief".


The number of words between the input text ranges from 1 to 33. In particular, the distribution is almost symmetrical as the mean is 12.8 and the median is 12.0 with a standard deviation of 6.70. The first and third quartiles are 7.0 and 18.0 respectively, confirming the relative symmetry of the distribution. The distribution of the number of words in the text column is shown in plot C of figure @fig:plots_EDA.


Regarding the number of characters in each input, the distribution is concentrated mostly between 2 and 180 and is almost uniform; however, there are some outliers with a high number of characters (up to 703). 
The distribution is slightly left-skewed as the mean is 68.3 and the median is 65.0 with a standard deviation of 36.7.

The plot of the distribution is visible in plot D of figure @fig:plots_EDA, where the logarithmic scale is used to make the plot more readable.
The four inputs with the highest number of characters are the following are the following:
 - "Communism naturally results in dictatorship. When you centralize power, you invite the power-hungry. Every morally-motivated communist seems to think they're immune to the bullets of power-hungry totalitarians." with 211 characters and label "neutral";

 - "here you go: |Games|Home|Away|Team|vs W-L 18-19| |:-|:-|:-|:-|-:| |1|1|0|Golden State|0-1| |2|1|1|Houston|0-0| |2|1|1|Oklahoma City|0-0| |2|0|2|Toronto|1-1| |2|1|1|Milwaukee|0-1| |3|1|2|Indiana|0-1| |2|1|1|Philadelphia|1-1| |2|1|1|Boston|1-0| |1|1|0|Minnesota|0-1| |2|1|1|New Orleans|0-0| |2|1|1|Memphis|0-0| |1|1|0|Dallas|0-1| |1|0|1|Miami|2-1| |3|2|1|Brooklyn|0-0| |2|1|1|Charlotte|0-2| |2|0|2|Detroit|1-1| |2|1|1|Washington|1-1| |2|1|1|New York|2-0| |2|1|1|Cleveland|1-0| |4|2|2|Atlanta|0-0| |1|1|0|Chicago|2-1|" with 514 characters and label "neutral";

 - "For your kindness to mobile users I give a platinum ⠀⠀⠀⠀⠀⣤⣶⣶⡶⠦⠴⠶⠶⠶⠶⡶⠶⠦⠶⠶⠶⠶⠶⠶⠶⣄⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⣿⣀⣀⣀⣀⠀⢀⣤⠄⠀⠀⣶⢤⣄⠀⠀⠀⣤⣤⣄⣿⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠿⣿⣿⣿⣿⡷⠋⠁⠀⠀⠀⠙⠢⠙⠻⣿⡿⠿⠿⠫⠋⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⢀⣤⠞⠉⠀⠀⠀⠀⣴⣶⣄⠀⠀⠀⢀⣕⠦⣀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⢀⣤⠾⠋⠁⠀⠀⠀⠀⢀⣼⣿⠟⢿⣆⠀⢠⡟⠉⠉⠊⠳⢤⣀⠀⠀⠀ ⠀⣠⡾⠛⠁⠀⠀⠀⠀⠀⢀⣀⣾⣿⠃⠀⡀⠹⣧⣘⠀⠀⠀⠀⠀⠀⠉⠳⢤⡀ ⠀⣿⡀⠀⠀⢠⣶⣶⣿⣿⣿⣿⡿⠁⠀⣼⠃⠀⢹⣿⣿⣿⣶⣶⣤⠀⠀⠀⢰⣷ ⠀⢿⣇⠀⠀⠈⠻⡟⠛⠋⠉⠉⠀⠀⡼⠃⠀⢠⣿⠋⠉⠉⠛⠛⠋⠀⢀⢀⣿⡏ ⠀⠘⣿⡄⠀⠀⠀⠈⠢⡀⠀⠀⠀⡼⠁⠀⢠⣿⠇⠀⠀⡀⠀⠀⠀⠀⡜⣼⡿⠀ ⠀⠀⢻⣷⠀⠀⠀⠀⠀⢸⡄⠀⢰⠃⠀⠀⣾⡟⠀⠀⠸⡇⠀⠀⠀⢰⢧⣿⠃⠀ ⠀⠀⠘⣿⣇⠀⠀⠀⠀⣿⠇⠀⠇⠀⠀⣼⠟⠀⠀⠀⠀⣇⠀⠀⢀⡟⣾⡟⠀⠀ ⠀⠀⠀⢹⣿⡄⠀⠀⠀⣿⠀⣀⣠⠴⠚⠛⠶⣤⣀⠀⠀⢻⠀⢀⡾⣹⣿⠃⠀⠀ ⠀⠀⠀⠀⢿⣷⠀⠀⠀⠙⠊⠁⠀⢠⡆⠀⠀⠀⠉⠛⠓⠋⠀⠸⢣⣿⠏⠀⠀⠀ ⠀⠀⠀⠀⠘⣿⣷⣦⣤⣤⣄⣀⣀⣿⣤⣤⣤⣤⣤⣄⣀⣀⣀⣀⣾⡟⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⢹⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠃ " 
 with 542 characters and label "neutral";

 - "This person is the smartest person to play town of salem literally 999999999999999999999999999999999999999999999999999999999999999999999999999999999999999991000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001234567898765432345676543345678987654345678909876543234567898765432345678909876543234567898765432345678987654323456787654345676543456543456434543434343434323456765434567654323454323456543345678987654323456789876565656565656565656565656565454545654565454323456765432345678765456 IQ 
 with 703 characters and label "admiration".



#figure(
  image("/0. Exploratory/plots/image4.png", width: 100%),
  caption: [Plot A: Distribution of the lables in the dataset (removing the label "neutral"); Plot B: Distribution of the number of labels per input (in logarithmic scale); Plot C: Distribution of the number of words in the inpouts; Plot D: Distribution of the number of characters in the text column (in logarithmic scale).],
) <fig:plots_EDA>

= Model Architecture

// Backbone Model + VAE

*Inference*

*Metrics* To evaluate our model, we employ two strategies: for emotion classification we use both a fine tuned model and four queried large language models; for translation and meaning preservation we use four large language models and a human poll.

// TODO write in details the part about the fine-tuned model
// TODO Write in details the part about the poll
// TODO Check the precise size of dataset used for evaluation and the specific large language models employed 

= Classification Task

*Evaluation*

*Baseline*

*Results and Ablations*

== Human-Guided LLM Annotation

For emotion classification evaluation, we select a subset of the GoEmotions dataset and query four large language models via the OpenRouter API. Each model receives a specific prompt containing the original sentence from the dataset plus the full list of 28 GoEmotions labels, and is explicitly told to return a list of the emotions it thinks best express the meaning of the sentence. The predicted emotions are then extracted deterministically through regex over the full response in case structured line is malformed.

For translation evaluation, we also select a subset of the GoEmotions dataset and query the same four language models. Each model receives a specific prompt containing the original sentence from the dataset plus a target emotion and is instructed to rewrite the sentence so that it expresses the target emotion while preserving the original semantic meaning. The final translated sentences are collected and used both for evaluation against our model and for the human poll described above.

= Translation

*Evaluation*

*Baseline*

*Results and Ablations*

= Discussion

= Conclusion

// TODO include limitations
// The dataset is in the English language, comes from Reddit
// each LLM has a different pretraining dataset - Data Leakage?

#pagebreak()

#bibliography("sources.bib")


#pagebreak()

= Appendix


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
