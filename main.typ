#import "tracl.typ": *
#import "tracl-pergamon.typ": *
#import "tracl-titlebox.typ": *
#import "appendix.typ": appendix


#set heading(numbering: "1")

#let backmatter(content) = {
  set heading(numbering: "A.1")
  set page(columns: 1)
  counter(heading).update(0)
  state("backmatter").update(true)
  content
}


#let label(name) = [
  #par("")
  *#name.* #h(1em)
]

// Save Typst's built-in lorem function before using `lorem` as a parameter name.
#let make-lorem = lorem

// Put this near the top of your document.

// Escape regex metacharacters in a literal word.
#let regex-escape(s) = {
  let specials = regex("[\\\\^$.*+?()\\[\\]{}|]")
  s.replace(specials, "\\$0")
}

#let todo(s) = {
  text(
    fill: red,
    size: 20pt,
    s,
  )
}

// Global banned-word highlighter.
// Use as: #show: banned-words.with(("foo", "bar"), enabled: true)
#let banned-words(words, enabled: true, body) = {
  if not enabled or words.len() == 0 {
    body
  } else {
    let pattern = "(?i)\\b(" + words.map(regex-escape).join("|") + ")\\b"

    show regex(pattern): it => text(
      fill: red,
      size: 100pt,
      it,
    )

    body
  }
}
#show: banned-words.with(
  (
    "utilise",
    "utilize",
    "in order to",
    "in spite of",
    "at that point in time",
    "at this point in time",
    "in the event that",
    "until such time as",
    "on account of",
    "in the majority of cases",
    "has the capability of",
    "in spite of the fact that",
    "in the final analysis",
    "a large percentage of",
    "owing to the fact that",
    "need to be established",
    "give consideration to",
    "with the exception of",
    "it would thus appear that",
    "in order to",
    "we aim to do",
    "embedding",
    "embeddings",
    "SOTA",
    "at the end",
  ),
  enabled: true,
)


// Global switch.
#let word-limit-enabled = state("word-limit-enabled", true)

#let word-limit-on() = word-limit-enabled.update(true)
#let word-limit-off() = word-limit-enabled.update(false)

#let plain-text(it) = {
  if type(it) == str {
    it
  } else if it == [ ] {
    " "
  } else if it.has("children") {
    it.children.map(plain-text).join("")
  } else if it.has("body") {
    plain-text(it.body)
  } else if it.has("text") {
    plain-text(it.text)
  } else {
    ""
  }
}

#let word-count(body) = {
  let s = plain-text(body).trim()
  if s == "" {
    0
  } else {
    s.split(regex("\s+")).len()
  }
}


#let limit-text(
  body,
  red-limit: 150,
  show-count: false,
  lorem: false,
) = context {
  let enabled = word-limit-enabled.get()

  // Global disable removes both lorem replacement and coloring.
  if not enabled {
    assert(lorem == false)
    body
  } else {
    let n = word-count(body)
    let content = if lorem { make-lorem(red-limit) } else { body }
    if show-count { [(#n)] }
    if n > red-limit {
      text(fill: red)[#content]
    } else {
      content
    }
  }
}

#show: doc => acl(
  doc,
  anonymous: false,
  titlebox-height: 3cm,
  title: [Emo-DiVA: Emotion Disentanglement \ via Variational Autoencoders],
  authors: make-authors(
    (
      name: [Pietro \ Maran],
      affiliation: [],
    ),
    (
      name: [Tebe \ Nigrelli],
      affiliation: [],
    ),
    (
      name: [Federico \ Pezzoli],
      affiliation: [],
    ),
    (
      name: [Alessandro \ Reali],
      affiliation: [],
    ),
    (
      name: [Luca \ Ricci],
      affiliation: [],
    ),
    (
      name: [Ali Emre \ Senel],
      affiliation: [],
    ),
  ),
)


#abstract(
  limit-text(red-limit: 150)[
    Encoder-decoder transformer models do not have interpretable hidden representations.
    Addressing this limitation can help with model alignment: disentangling the inner representations of different information can make it easier to access, edit and use.
    Current approaches use Variational Autoencoders (VAEs) to produce latents with interpretable components, with limited success.
    We experiment with a T5 + FactorVAE architecture, progressively expanding it to separate emotion from context.
    We test the model on classification and translation of emotion. Classification performs better than the BERT baseline, while translation fails despite our additions.
    The results suggest that we achieve partial disentanglement of emotion, but architecture and dataset are jointly insufficient to implement translation.
  ],
)
= Introduction

#limit-text(red-limit: 300)[
  Pretrained text-to-text encoder-decoder Transformer models such as T5 and FLAN-T5 learn powerful linguistic representations @raffel2020exploring @FLAN-T5, but their internal states are not exposed as latents that users can inspect, edit, or constrain. This limits explainability and alignment, since different aspects of meaning often remain entangled in the hidden representation. Recent work addresses this issue by processing the hidden states of a pretrained sequence-to-sequence model using a Variational Autoencoder @park2021finetuningpretrainedtransformersvariational. This mechanism forces the model to map text into structured latent variables before decoding them back into language.

  We apply this framework by injecting a FactorVAE @kim2019disentanglingfactorising between the encoder and decoder of FLAN-T5, with the goal of separating emotional information from contextual content in the latent space. We evaluate the resulting model on emotion classification, latent-space disentanglement, sentence reconstruction, and emotion translation. By reconstruction, we mean the model’s ability to reproduce the input sentence when the emotion-related latent component is left unchanged. By translation, we mean modifying the emotional tone of a sentence while preserving its contextual meaning, as in @pizza. We outperform the BERT baseline in emotion classification, investigate emotion disentanglement, and achieve reconstruction, but not translation.
]

#place(
  top + right,
  scope: "column",
  float: true,
  block(width: 100%)[
    #figure(
      image("img/pizza.png", width: 90%),
      caption: [Emotion translation example.],
    ) <pizza>
  ],
)


= Experimental Setup

#label[Backbone] #limit-text(red-limit: 50)[
  We use FLAN-T5-base as our base model because it was pretrained on a large dataset, and its weights are publicly available. Also, its encoder–decoder structure allows us to train the entire architecture. Its limited size also enables faster training, thus more experiments @FLAN-T5.
]

#label[VAE] #limit-text(red-limit: 50)[
  We use a FactorVAE because it encourages representations with uncorrelated factors and supports generation from the learned latent space @kim2019disentanglingfactorising. We split latent vectors into two subspaces: emotion and context.
]

#label[Dataset] #limit-text(red-limit: 50)[
  We use GoEmotions, a corpus of 54k English Reddit samples annotated with 27 emotion labels plus 1 for neutral. We select it for its broad coverage, size, multi-label format, and canonical train-validation-test split, which lets us compare our classifier with published results  @demszky2020goemotions.
]

== Tasks and Evaluation

#limit-text(red-limit: 250)[
  Disentanglement is difficult to evaluate intuitively, as it involves high-dimensional spaces and a complex structure for meaning. We evaluate separation using metrics that capture different properties of the latent space, and from two downstream tasks: *emotion classification* and *emotion translation*. Since no models achieved meaningful results in emotion translation, we also evaluate reconstruction as a diagnostic intermediate step.

  The main disentanglement metrics are: *mutual information gap* (MIG), where higher values indicate stronger separation between emotion and context information; *emotion $R^2$*, which measures how much emotion-label variance can be predicted from the emotion latent using linear regression; and *separation $R^2$*, defined as the difference between emotion $R^2$ and the largest non-emotion $R^2$, which may come from the context latent or residual connections depending on the architecture. MIG and emotion $R^2$ range from 0 to 1, with higher values being better. Separation $R^2$ is also better when larger, but unless clipped or rescaled, it can range from -1 to 1.

  We evaluate classification with *macro-F1* and *micro-F1*. Reconstruction is measured with *exact match* and *token F1*. For translation, we only report fluency through *GPT-2 perplexity* (PPL), since outputs are often nonsensical. We also report *median-PPL*, because *mean-PPL* suffers from the presence of large outliers.
]

#place(
  bottom + right,
  scope: "column",
  float: true,
  block(width: 100%)[
    #figure(
      image("img/goemotions_emotion_counts_sorted_excluding_neutral.png", width: 100%),
      caption: [Label frequency in the dataset],
    ) <EDA>
  ],
)

= Exploratory Data Analysis

#limit-text(red-limit: 200)[
  We analyze label frequency, number of emotions per input, input length, and label co-occurrence on 54263 GoEmotions comments GoEmotions comments. Excluding neutral, the three most frequent classes are admiration (5122), approval (3687), and gratitude (3372), while grief (96), pride (142), relief (182), and nervousness (208) are rare. The dataset is therefore strongly imbalanced. Most inputs contain a single emotion: 83.8% of examples have one label, 15.0% have two, 1.2% have three, 0.07% have four, and fewer than 0.01% have five. This confirms that the task is mostly single-label, but still requires sigmoid outputs instead of a softmax classifier. Inputs are short: the median length is 12 words, although the character distribution has a long tail up to 703 characters. These findings motivate reporting macro-$F_1$ together with micro-$F_1$, since micro-$F_1$ can hide poor performance on rare emotions.
]


= Architectures and Results

#place(
  top + right,
  scope: "column",
  float: true,
  block(width: 100%)[
    #figure(
      image("img/archit_1.png", width: 105%),
      caption: [1 $->$ base; 2 $->$ residual bottleneck; \ 3 $->$ LoRA on T5 decoder + copy loss; 4 $->$ adversarial classifier.],
    ) <arch-1-4>
  ],
)

#label[Architecture 1]
#limit-text(red-limit: 1000)[
  Our base architecture (blue in @arch-1-4) inserts a FactorVAE between the T5 encoder and decoder. The FactorVAE encoder maps transformer hidden states into a sequence of latent representations, _separating emotion and context branches_; the FactorVAE decoder then projects these latent representations back to the hidden-state dimension as T5 decoder memory. Depending on the task, the T5 decoder finally reconstructs or translates the input, producing a new sequence of tokens.

  For classification, we pool the sequence of emotion latents using attention weights generated from the T5 encoder hidden states. Pooling can be done either *jointly*, across the emotion subspace, or independently *per emotion factor*. The resulting representation is then passed to an MLP classifier. Similar to pooling, the classifier can either use a *single classification head* or use *separate heads* for individual latent dimensions. For a size comparison, see @tab:arch_1_size.

  We train our model using a combination of different losses: a standard binary classification loss, a reconstruction loss, a KL loss for the VAE, and a total-correlation loss, specific for the disentanglement in the FactorVAE @kim2019disentanglingfactorising. The respective weights are $alpha_("cl")=4.0, alpha_("rec")=2.0, alpha_("KL")=0.05, alpha_("TC")=0.1$.

  We test this architecture by ablating on the dimensionality of the emotion and context latents, the pooling mode, and the classifier mode. Since our dataset contains 28 labels, we set the emotion latent dimensionality to either 28 or 56, while for the context latent we test 64 and 128 dimensions @tab:arch_1_results.

  The results show that joint pooling favors disentanglement, especially in terms of separation $R^2$ ($-0.003 -> 0.009$). A lower-dimensional context latent has a similar effect on separation $R^2$ ($-0.017 -> 0.023$). Moreover, a 56-dimensional emotion latent is better separated from the context latent than a 28-dimensional emotion latent ($-0.007 -> 0.013$). Branch MIG follows the same trend: averaged across ablations, joint pooling improves over per-emotion pooling ($0.227 -> 0.195$), a 64-dimensional context latent improves it over a 128-dimensional context latent ($0.223 -> 0.199$), and a 56-dimensional emotion latent improves it over a 28-dimensional one ($0.220 -> 0.203$). The best Branch MIG value ($0.246$) is obtained by the same configuration that maximizes separation $R^2$ @tab:arch_1_results.

  Classification achieves on average *$tilde$0.5 macro-F1*: since it remains above the BERT baseline (*0.46 macro-F1*, as per @demszky2020goemotions) consistently across ablations, we keep the 56-dimensional emotion latent. Consequently, we configure the MLP with *one classifier head for every two emotion latent dimensions*, since this classifier mode achieves the best results in both micro- and macro-F1. Reconstruction and translation fail for all tested setups.
]

#label[Architecture 2]
#limit-text(red-limit: 150)[
  We add a *residual connection* (red in @arch-1-4) to route features around the VAE. The residual connection is a small MLP bottleneck over T5 hidden states: a linear layer reducing dimension from 768 to 64; GELU activation; 50% dropout; a final linear layer mapping the hidden states back to their original dimension. The goal is to stabilize training and avoid catastrophic forgetting of the T5 decoder. Although classification performance matches Architecture 1, *reconstruction remains poor*, with a token-F1 value of 0.1. However, skip connections do not hurt disentanglement, and in fact they improve it: residual $R^2$ is considerably low (< 0.01) compared to the context $R^2$ (< 0.1, an improvement compared to Architecture 1) and, more importantly, emotion $R^2$ ($approx$ 0.15).
  // #todo[add figures on model architecture size / skip connection dimensions / etc]
]

#label[Architecture 3]
#limit-text(red-limit: 100)[
  Since the VAE's reconstruction loss is too weak to teach the decoder how to read the new decoder memory, we use *LoRA* adapters @hu2022lora to train the T5 decoder efficiently and add a token-level *copy loss* (brown section in @arch-1-4). Copy loss is a teacher-forced cross-entropy loss of the T5 decoder, forcing the output from decoder memory $M$ to match the input:
  $
    L_("copy") = -1/T sum_(t=1)^T log p_theta (x_t | x_(<t), M) .
  $
]

This model preserves the previous disentanglement results, while slightly improving classification, increasing micro-F1 to 0.597 and macro-F1 to 0.517, and having LoRA rank = 16 and LoRA $alpha$ = 32. As intended, copy loss boosts reconstruction to consistently good results, with 90% perfect matches across samples and *$tilde$ 99% token-F1*. Translation becomes more fluent, but still fails, with a median-PPL of 96 and a copy rate of almost 80%. Since modifying the emotion vector does not influence reconstruction, we believe the model *stores emotional information in the context vector*.


#label[Architecture 4]
#limit-text(red-limit: 500)[
  We add two *adversarial MLPs*, one on the context latents and one on the residual branch. Each MLP pools its input and learns to predict emotion. At the same time, the MLPs send inverse gradients backward, which push emotion out of context, *increasing separation $R^2$ to 0.1*. Surprisingly, the adversarial components are not enough to shake translation from mere reconstruction, leading to similar results, which suggests the *model overfits on reconstruction*, using decoder-ready context and residuals.
]

#label[Architecture 5]
#limit-text(red-limit: 500)[
  Since residual and adversarial variants show that regularization alone does not solve the shortcut problem, we rethink the architecture. We replace the residual connection and VAE decoder with a split decoder and introduce pooled emotion injection, inspired by feature-wise linear modulation (FiLM) @perez2017filmvisualreasoninggeneral and emotion modulation in LLM-TTS @wang2025globalemotionfinegrainedemotional.   The split decoder maps context latents to token memory $M_c$ and emotion latents to token memory $M_e$. The pooled emotion summary produces a general bias $b_e$ and FiLM parameters $Delta gamma_e$ and $beta_e$. The decoder memory is computed as:
  $
    M = (1 + op("tanh")(Delta gamma_e)) dot (M_c + M_e + b_e) + beta_e .
  $

  During training, inspired by Park et al. @park2021finetuningpretrainedtransformersvariational, we add a warmup schedule to the FactorVAE. The KL and TC loss weights start at zero, allowing the split-FiLM decoder to learn reconstruction before regularization. The weights are then gradually increased over the training.
  We evaluate how different schedules impact the model @tab:beta_schedule_ablation, varying the number of epochs (0, 2, 4, 6) where KL and TC losses are deactivated, followed by four warm-up epochs. The best configuration, with 4 zero-epochs, improves classification, reaching 59% micro-F1 and *52% macro-F1*, and improves disentanglement, in particular increasing *separation $R^2$ above 0.14*. As for translation, the model shows modest improvements, reducing copy rate down to 65%, without affecting median-PPL; but still fails to achieve emotion translation.
]


#label[Reproducibility]
#limit-text(red-limit: 500)[
  We tune the model by varying the latent bottleneck and the components that most affect task performance, while keeping the optimization setup fixed across comparable runs. We use learning rate $2 times 10^(-4)$, train batch size 8, evaluation batch size 16, weight decay $10^(-2)$, 10 training epochs, and 0.1 warmup ratio. This setup is stable enough for our experiments, so we focus the search on scalar emotion factors, context-vector size, pooling strategy, skip connections, decoder LoRA, adversarial regularization, and KL/TC warmup. The best trade-off uses a compact 64-dimensional context vector, with micro-$F_1$ around 0.59 and macro-$F_1$ around 0.52. The repository includes a _reproducible_scripts/_ folder with the Slurm scripts used to train, evaluate, and summarize the reported runs: _train_eval_\*_, _summarize_\*_, _run_reconstruction_\*_, _run_translation_\*_, and _collect_beta_classification_samples.slurm_. Attention analyses are also reproduced using _analyze_attention_maps.py_, _visualize_attention_examples.py_, and _aggregate_attention_words_by_emotion.py_. Further architectural details and ablations are reported in  @tab:arch_1_size.
]




#place(
  top + right,
  scope: "column",
  float: true,
  block(width: 100%)[
    #figure(
      image("img/archit_2.png", width: 110%),
      caption: [5 $->$ split-FiLM VAE Decoder.],
    ) <final_arch>
  ],
)

= Discussion

#limit-text(red-limit: 500)[
  The final architecture surpasses BERT-level accuracy in emotion classification @harutyunyan2026finegrainedemotiondetectiongoemotions @demszky2020goemotions and successfully  reconstructs sentences, but fails at emotion translation. More specifically, translation experiments show that the decoder rewrites the input with small lexical edits at times, generating fluent but semantically weak rewrites in other cases. In fact, the model never achieves proper emotion translations. However, the model captures emotion well enough for recognition, but not well enough to manipulate it independently from context.

  For example, using as input the sentence _"I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!"_, labeled in the dataset as "remorse" and "sadness", our model correctly labels it and reconstructs it by changing the second name in the list. However, translation to "joy" fails, and outputs the same sentence.


  This inability indicates that the model *does not yet have a stable mechanism* for moving from one emotional region of the latent space to another while preserving sentence context. However, the architecture *can organize emotion information* into a classifiable subspace. This aspect is not surprising, since the FactorVAE forces the latent space of the T5 model into a smaller space, where the classification MLP can successfully recover the information for the emotion recognition task. The success of sentence reconstruction can also be explained by the introduction of the copy loss from architecture 3, which encourages the model to correctly reproduce the input.

  The failure in the emotion translation task, instead, can be attributed to multiple reasons. GoEmotions is a small dataset when compared to the training data of the T5 model. Moreover, it provides emotion labels, without parallel sentence pairs, showing the same context with different emotions. Also, *emotion and context are often deeply entangled in natural language*, so changing the emotion without changing the semantic content is intrinsically difficult.

  Copy loss may have influenced the outcome of the task by pushing the model toward the total reconstruction too strongly, introducing a *bias toward copying the input* instead of modifying emotional content. Nonetheless, the small lexical edits and the few semantically weak rewrites obtained by our model suggest that we are moving in the correct direction.

  We also visualized attention weights, confirming the model attends to plausible lexical cues @fig:qual-attention-examples, and the per-label most attended words are sensible @tab:top-attended-words. We see that the latent pooling attention is most interpretable when the emotion has a conventional lexical realization. Emotions attend to typically relevant words, such as "gratitude" attending to _thanks_ @tab:top-attended-words. Also, KL divergence over the phrases shows the model spreads attention over the tokens, as opposed to focusing only on easy-to-spot single words, which would appear as a concentrated attention map @attention-entropy-analysis.
]

= Related Work

#limit-text(red-limit: 40)[
  Fu et al. @fu2017styletransfertextexploration perform _style transfer_ with non-parallel corpora by disentangling content and style through adversarial training. They evaluate both transfer strength and content preservation; we consider a stricter setting, modifying emotion while preserving contextual meaning.
]

#limit-text(red-limit: 40)[
  Park and Lee @park2021finetuningpretrainedtransformersvariational convert pretrained encoder-decoder transformers into text VAEs using two-stage finetuning: first autoencoding with zero KL weight, followed by gradual KL warmup. We adopt only the warmup strategy as stabilizer in Architecture 5.
]

#limit-text(red-limit: 40)[
  Perez et al. @perez2017filmvisualreasoninggeneral introduce FiLM, which modulates intermediate features through feature-wise affine transformations based on conditioning information. While they condition visual reasoning on language, we use FiLM-style modulation in Architecture 5 to inject pooled emotion information into T5 decoder memory.
]

= Conclusion

#limit-text(red-limit: 150)[
  We investigated whether the latent space of an encoder-decoder model can be disentangled into emotion and semantic context. To this end, we inserted a FactorVAE between the encoder and decoder of a T5 model and refined the architecture toward a FiLM-style design. The final models achieved competitive performance on emotion classification and sentence reconstruction, but failed to perform reliable emotion translation. These results suggest that the proposed research direction may be limited by dataset size, model choice, and an unavoidable emotional entanglement in the latent representation. Future work could explore alternative encoder-decoder models, such as BART, which is pretrained as a denoising autoencoder. Moreover, weakening copy loss through a stochastic mask may help reduce overfitting. Finally, applying dimensionality reduction and latent-space visualization techniques can help clarify why disentanglement and emotion-controlled generation remain challenging.
]

= Limitations

#label[Dataset]
#limit-text(red-limit: 100)[
  We only used the GoEmotions dataset, which is small compared to the 750 GB C4 corpus used for training FLAN-T5-base @raffel2020exploring. Size limits lexical and emotional variation, by not providing a wide coverage of both emotion and context independently. Multiple sources of bias also exist in the Reddit dataset: it is almost completely in English, it contains community-specific patterns, and was labelled by human annotators.
]


#label[Model]
#limit-text(red-limit: 100)[
  All our results were only tested with FLAN-T5-base as backbone. Although we expect our results to carry to other architectures, we only tested the FLAN-T5-base pretrained model, which limits our findings.
]

#label[Ablations]
#limit-text(red-limit: 100)[
  All ablation settings were run once. Consequently, performance metrics may be affected by poor initialization or randomness, although we do not see evidence in our experiments to support this claim.
]


#bibliography("sources.bib", style: "ieee")

#pagebreak()

#show: backmatter
#appendix
