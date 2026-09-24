#let clamp(x, lo, hi) = calc.min(calc.max(x, lo), hi)

#let cell-size = 8pt
#let sep-stroke = 1.1pt

#let att-fill(att, max-att: 0.36) = {
  let t = clamp(att / max-att, 0.0, 1.0)
  let sat = 42% + 42% * t
  let light = 97% - 50% * t
  color.hsl(22deg, sat, light)
}

#let att-token(tok, att) = box(
  fill: att-fill(att),
  inset: (x: 0.3pt, y: 1pt),
  radius: 0.7pt,
)[
  #grid(
    columns: 1,
    row-gutter: 1pt,
    align: center,
    text(size: cell-size)[#tok],
    text(size: cell-size)[#str(att)],
  )
]

#let true-fill = rgb("#86efac")
#let false-fill = rgb("#fca5a5")
#let mismatch-fill = rgb("#fee2e2")

#let label-cell(label, rowspan: 1, mispredicted: false) = table.cell(
  rowspan: rowspan,
  fill: if mispredicted { mismatch-fill } else { none },
  align: left + horizon,
)[#text(size: cell-size, weight: "bold")[#label]]

#let pred-cell(name, pred, gold) = table.cell(
  fill: if pred != gold { mismatch-fill } else { none },
  align: left + horizon,
)[#text(size: cell-size, weight: "bold")[#name]]

#let score-cell(score, threshold) = table.cell(
  fill: if score >= threshold { true-fill } else { false-fill },
  align: center + horizon,
)[#text(size: cell-size, weight: "bold")[#str(score)]]

#let threshold-cell(threshold) = table.cell(
  align: center + horizon,
)[#text(size: cell-size)[#str(threshold)]]

#let phrase-cell(rowspan: 1, body) = table.cell(
  rowspan: rowspan,
  align: left + horizon,
)[
  #block(width: 100%)[
    #set par(justify: false)
    #body
  ]
]

#let comment-cell(rowspan: 1, body) = table.cell(
  rowspan: rowspan,
  align: left + horizon,
)[#text(size: cell-size)[#body]]

#let header-cell(body) = table.cell(fill: rgb("#fff2cc"))[#body]
#let notable(body) = table.cell(fill: rgb("#d9ead3"))[#strong(body)]

#let appendix = [

  #show figure: set block(breakable: true)

  = Architecture 1 Ablations

  The table compares Architecture 1 across pooling strategies, emotion-factor counts, and context dimensions. The strongest overall configuration uses 56 emotion factors. Different settings achieve the best trade-offs across classification, emotion prediction, separation, and branch disentanglement.

  #figure(
    table(
      columns: (2.9fr, 1fr, 1.2fr, 1.4fr, 1.4fr, 1.2fr, 1.2fr, 1.2fr),
      align: (left, center, center, right, right, right, right, right),
      inset: 6pt,
      stroke: 0.5pt,

      table.header(
        header-cell[Pooling mode],
        header-cell[Emo],
        header-cell[Context],
        header-cell[micro-F1],
        header-cell[macro-F1],
        header-cell[Emo R#super[2]],
        header-cell[Sep R#super[2]],
        header-cell[B. MIG],
      ),

      [joint_emotion_context], [28], [64], [0.580], [0.508], [0.149], [0.019], [0.241],
      [joint_emotion_context], [28], [128], [0.579], notable[0.514], [0.151], [-0.017], [0.202],
      [per_emotion_dim], [28], [64], [0.576], [0.508], [0.131], [0.002], [0.195],
      [per_emotion_dim], [28], [128], [0.580], [0.491], [0.130], [-0.032], [0.172],

      [joint_emotion_context], [56], [64], [0.585], [0.504], [0.164], notable[0.041], notable[0.246],
      [joint_emotion_context], [56], [128], [0.587], notable[0.514], notable[0.166], [-0.006], [0.218],
      [per_emotion_dim], [56], [64], notable[0.591], [0.504], [0.155], [0.030], [0.211],
      [per_emotion_dim], [56], [128], [0.589], [0.505], [0.155], [-0.012], [0.203],
    ),
    caption: [Architecture 1 results],
  ) <tab:arch_1_results>
  \
  We report architecture size for the different ablations of Architecture 1, depending on latent dimensions, pooling mode and the number of classifier heads. In the table, B is batch size and T is input sequence length. Note that T5 hidden states have dimension 768.
  #figure(
    table(
      columns: (2fr, 2fr, 4fr, 12fr, 3fr, 8fr),
      inset: 3pt,
      align: (center, center, center, left, center, left),
      table.header(
        header-cell[Emo],
        header-cell[Ctx],
        header-cell[Pooling mode],
        header-cell[Pooling shape \ [B,T,768] $->$ ...],
        header-cell[Class. heads],
        header-cell[Class. shape],
      ),

      [28], [64], [per-factor], [separate emotion pools $->$ [B,28,8]], [multi], [28 × \ MLP(8 $->$ 32 $->$ 1)],
      [28], [64], [joint], [shared joint evidence $->$ [B,28,8]], [multi], [28 × \ MLP(8 $->$ 32 $->$ 1)],
      [28], [128], [per-factor], [28 separate emotion pools $->$ [B,28,8]], [multi], [28 × \ MLP(8 $->$ 32 $->$ 1)],
      [28], [128], [joint], [shared joint evidence $->$ [B,28,8]], [multi], [28 × \ MLP(8 $->$ 32 $->$ 1)],

      [56], [64], [per-factor], [56 separate emotion pools $->$ [B,56,8]], [multi], [28 × \ MLP(16 $->$ 32 $->$ 1)],
      [56], [64], [joint], [shared joint evidence $->$ [B,56,8]], [multi], [28 × \ MLP(16 $->$ 32 $->$ 1)],
      [56], [128], [per-factor], [56 separate emotion pools $->$ [B,56,8]], [multi], [28 × \ MLP(16 $->$ 32 $->$ 1)],
      [56], [128], [joint], [shared joint evidence $->$ [B,56,8]], [multi], [28 × \ MLP(16 $->$ 32 $->$ 1)],
    ),
    caption: [Architecture 1 size comparison],
  )<tab:arch_1_size>

  #pagebreak()

  = Architecture 5

  == Ablation Results


  #figure(
    table(
      columns: (1fr, 1.4fr, 1.4fr, 1.2fr, 1.2fr, 1.2fr),
      align: (center, center, center, center, center, center),
      inset: 6pt,
      stroke: 0.5pt,
      table.header(
        header-cell[Epoch],
        header-cell[micro-F1],
        header-cell[macro-F1],
        header-cell[Sep R#super[2]],
        header-cell[Emo R#super[2]],
        header-cell[B. MIG],
      ),
      [0], [0.5880], [0.5200], [0.1147], [0.1411], [0.2430],
      [2], notable[0.5924], [0.5193], [0.1354], [0.1456], [0.2427],
      [4], [0.5914], notable[0.5240], [0.1414], [0.1486], [0.2516],
      [6], [0.5822], [0.5196], notable[0.1422], notable[0.1543], notable[0.2563],
    ),
    caption: [KL warmup start epoch. Higher warmup start improves all three disentanglement metrics at a small classification cost.],
  ) <tab:beta_schedule_ablation>


  == Attention Maps

  Following multi-hot classification, the model assigns each label an independent probability $hat(p)$. A calibration set is used to identify the threshold value. A label is predicted as active if its score exceeds the label-specific threshold $theta$, i.e. $hat(p) >= theta$. \ \

  #figure(
    table(
      columns: (0.55in, 0.6in, 0.3in, 0.3in, 3.55in, 0.85in),
      stroke: 0.25pt,
      inset: 2pt,

      table.header(
        [#text(size: cell-size, weight: "bold")[Labels]],
        [#text(size: cell-size, weight: "bold")[Predicted]],
        [#text(size: cell-size, weight: "bold")[$hat(p)$]],
        [#text(size: cell-size, weight: "bold")[$theta$]],
        [#text(size: cell-size, weight: "bold")[Phrase token importance - Average Attention Map]],
        [#text(size: cell-size, weight: "bold")[Comment]],
      ),

      // Phrase 1
      label-cell("gratitude"),
      pred-cell("gratitude", true, true),
      score-cell(0.999, 0.921),
      threshold-cell(0.921),
      phrase-cell[
        #att-token("I", 0.03) #att-token("didn", 0.09) #att-token("'", 0.01)
        #att-token("t", 0.09) #att-token("know", 0.13) #att-token("that", 0.02)
        #att-token(",", 0.01) #att-token("thank", 0.34) #att-token("you", 0.03)
        #att-token("for", 0.06) #att-token("teaching", 0.09) #att-token("me", 0.02)
        #att-token("something", 0.04) #att-token("today", 0.02) #att-token("!", 0.04)
      ],
      comment-cell[
        The highest mass falls on the explicit cue "thank".
      ],

      table.hline(stroke: sep-stroke),

      // Phrase 2
      label-cell("amusement"),
      pred-cell("amusement", true, true),
      score-cell(0.997, 0.921),
      threshold-cell(0.921),
      phrase-cell(rowspan: 2)[
        #att-token("Lo", 0.18) #att-token("l", 0.23) #att-token("!", 0.03)
        #att-token("But", 0.05) #att-token("I", 0.02) #att-token("love", 0.28)
        #att-token("your", 0.03) #att-token("last", 0.01) #att-token("name", 0.01)
        #att-token("though", 0.03) #att-token(".", 0.02) #att-token("▁", 0.01)
        #att-token("X", 0.02) #att-token("D", 0.08)
      ],
      comment-cell(rowspan: 2)[
        The map peaks on love, but assigns mass to "Lol" fragments.
      ],

      label-cell("love"),
      pred-cell("love", true, true),
      score-cell(0.965, 0.921),
      threshold-cell(0.921),

      table.hline(stroke: sep-stroke),

      // Phrase 3
      label-cell("—", mispredicted: true),
      pred-cell("approval", true, false),
      score-cell(0.884, 0.804),
      threshold-cell(0.804),
      phrase-cell(rowspan: 2)[
        #att-token("I", 0.02) #att-token("had", 0.06) #att-token("to", 0.05)
        #att-token("write", 0.04) #att-token("in", 0.03) #att-token("the", 0.02)
        #att-token("Swan", 0.02) #att-token("gas", 0.02) #att-token(".", 0.04)
        #att-token("See", 0.08) #att-token("m", 0.11) #att-token("s", 0.02)
        #att-token("far", 0.08) #att-token("better", 0.26) #att-token("than", 0.05)
        #att-token("the", 0.02) #att-token("choices", 0.04) #att-token("given", 0.02)
        #att-token(".", 0.03)
      ],
      comment-cell(rowspan: 2)[
        Attention concentrates on "better", supporting approval, while missing the gold realization label.
      ],

      label-cell("realization", mispredicted: true),
      pred-cell("—", false, true),
      score-cell(0.403, 0.637),
      threshold-cell(0.637),
    ),
    caption: [
      Score cell is green when the score exceeds the threshold, showing classification, and red otherwise.
      Label and predicted-emotion cells are highlighted when prediction and gold labels disagree.
    ],
  ) <fig:qual-attention-examples>
  \
  \
  \
  #figure(
    kind: table,
    supplement: [Table],
    caption: [Top attended words by emotion.],
    text(size: 1.05em)[
      #table(
        columns: (5em, 5em, 5em),
        align: (left, left, right),
        inset: (x: 0.35em, y: 0.2em),
        stroke: none,

        table.hline(stroke: 0.6pt),

        table.header([#strong[Emotion]], [#strong[Word]], [#strong[Mass]]),

        table.hline(stroke: 0.4pt),

        table.cell(rowspan: 5)[gratitude], [thanks], [0.213],
        [thank], [0.182],
        [you], [0.017],
        [appreciate], [0.013],
        [welcome], [0.009],

        table.hline(stroke: 0.35pt),

        table.cell(rowspan: 5)[love], [love], [0.352],
        [like], [0.046],
        [loved], [0.019],
        [favorite], [0.009],
        [loving], [0.007],

        table.hline(stroke: 0.35pt),

        table.cell(rowspan: 5)[realization], [realize], [0.031],
        [realized], [0.029],
        [wow], [0.023],
        [forgot], [0.022],
        [never], [0.019],

        table.hline(stroke: 0.6pt),
      )
    ],
  ) <tab:top-attended-words>

  #pagebreak()

  == Attention Weights Entropy Analysis <attention-entropy-analysis>

  #let qtable-header-cell(body) = table.cell(fill: rgb("#fff2cc"))[#body]
  The quantile table suggests that attention concentration differs systematically
  by predicted emotion. Lexically explicit emotions such as #emph[gratitude],
  #emph[love], and #emph[remorse] show relatively low entropy, indicating sharp
  phrase-level evidence around cue words. #emph[Gratitude] also has the largest
  mean head-template KL, suggesting stronger head-level variation around these
  cues. In contrast, #emph[neutral] has the lowest KL and highest entropy,
  indicating broad, shared attention rather than a localized emotional cue. More
  discourse-dependent labels such as #emph[approval], #emph[realization],
  #emph[confusion], and #emph[curiosity] are also relatively high-entropy,
  consistent with the need to interpret context or stance rather than isolated
  keywords. \

  #figure(
    kind: table,
    supplement: [Table],
    caption: [KL and normalized-entropy distributions of the attention pooling weights by predicted-positive emotion. Quartile columns report Q1 / median / Q3],

    text(size: 12pt)[
      #table(
        columns: (1.5fr, 0.5fr, 0.7fr, 2fr, 0.7fr, 2fr),
        align: (left, right, center, right, center, right),
        inset: (x: 5pt, y: 3pt),
        stroke: 0.25pt,

        [],
        [],
        table.cell(colspan: 2, align: center, fill: rgb("#fff2cc"))[KL Divergence Value],
        table.cell(colspan: 2, align: center, fill: rgb("#fff2cc"))[H Information Entropy Value],

        table.header(
          qtable-header-cell[Emotion],
          qtable-header-cell[#h(1fr) $n$ #h(1fr)],
          qtable-header-cell[#h(1fr)$mu$#h(1fr)],
          qtable-header-cell[Q1#h(1fr)50#h(1fr)Q3],
          qtable-header-cell[#h(1fr)$mu$#h(1fr)],
          qtable-header-cell[Q1#h(1fr)50#h(1fr)Q3],
        ),

        [gratitude], [329], [0.167], [0.128#h(1fr)0.168#h(1fr)0.207], [0.748], [0.715#h(1fr)0.764#h(1fr)0.794],
        [amusement], [317], [0.146], [0.106#h(1fr)0.139#h(1fr)0.176], [0.809], [0.768#h(1fr)0.817#h(1fr)0.852],
        [desire], [37], [0.144], [0.108#h(1fr)0.141#h(1fr)0.175], [0.760], [0.714#h(1fr)0.777#h(1fr)0.820],
        [curiosity], [459], [0.143], [0.115#h(1fr)0.140#h(1fr)0.169], [0.870], [0.840#h(1fr)0.887#h(1fr)0.910],
        [optimism], [184], [0.143], [0.102#h(1fr)0.141#h(1fr)0.173], [0.825], [0.781#h(1fr)0.823#h(1fr)0.875],
        [embarrassment], [24], [0.140], [0.112#h(1fr)0.132#h(1fr)0.159], [0.810], [0.749#h(1fr)0.809#h(1fr)0.844],
        [confusion], [181], [0.140], [0.110#h(1fr)0.137#h(1fr)0.164], [0.864], [0.834#h(1fr)0.877#h(1fr)0.906],
        [remorse], [97], [0.137], [0.108#h(1fr)0.135#h(1fr)0.164], [0.747], [0.692#h(1fr)0.760#h(1fr)0.806],
        [fear], [106], [0.129], [0.098#h(1fr)0.124#h(1fr)0.156], [0.806], [0.771#h(1fr)0.819#h(1fr)0.852],
        [sadness], [276], [0.128], [0.095#h(1fr)0.122#h(1fr)0.157], [0.807], [0.750#h(1fr)0.824#h(1fr)0.868],
        [disgust], [187], [0.127], [0.097#h(1fr)0.116#h(1fr)0.155], [0.831], [0.793#h(1fr)0.831#h(1fr)0.873],
        [annoyance], [286], [0.121], [0.097#h(1fr)0.117#h(1fr)0.147], [0.849], [0.813#h(1fr)0.860#h(1fr)0.898],
        [love], [262], [0.120], [0.091#h(1fr)0.114#h(1fr)0.148], [0.742], [0.694#h(1fr)0.751#h(1fr)0.814],
        [anger], [189], [0.119], [0.095#h(1fr)0.115#h(1fr)0.146], [0.845], [0.809#h(1fr)0.861#h(1fr)0.906],
        [realization], [179], [0.117], [0.078#h(1fr)0.112#h(1fr)0.142], [0.858], [0.831#h(1fr)0.868#h(1fr)0.904],
        [disappointment], [259], [0.115], [0.088#h(1fr)0.110#h(1fr)0.140], [0.831], [0.790#h(1fr)0.848#h(1fr)0.881],
        [grief], [28], [0.113], [0.085#h(1fr)0.103#h(1fr)0.149], [0.848], [0.825#h(1fr)0.854#h(1fr)0.876],
        [admiration], [504], [0.113], [0.076#h(1fr)0.108#h(1fr)0.141], [0.812], [0.769#h(1fr)0.816#h(1fr)0.860],
        [joy], [221], [0.111], [0.078#h(1fr)0.108#h(1fr)0.142], [0.768], [0.726#h(1fr)0.773#h(1fr)0.825],
        [surprise], [165], [0.110], [0.069#h(1fr)0.103#h(1fr)0.140], [0.809], [0.768#h(1fr)0.822#h(1fr)0.855],
        [caring], [287], [0.107], [0.075#h(1fr)0.101#h(1fr)0.134], [0.846], [0.797#h(1fr)0.859#h(1fr)0.907],
        [disapproval], [351], [0.106], [0.083#h(1fr)0.102#h(1fr)0.128], [0.859], [0.833#h(1fr)0.871#h(1fr)0.897],
        [pride], [10], [0.104], [0.086#h(1fr)0.107#h(1fr)0.111], [0.773], [0.743#h(1fr)0.758#h(1fr)0.796],
        [relief], [10], [0.099], [0.081#h(1fr)0.087#h(1fr)0.105], [0.740], [0.677#h(1fr)0.774#h(1fr)0.834],
        [nervousness], [19], [0.099], [0.084#h(1fr)0.100#h(1fr)0.108], [0.784], [0.726#h(1fr)0.778#h(1fr)0.834],
        [excitement], [170], [0.095], [0.058#h(1fr)0.091#h(1fr)0.126], [0.813], [0.761#h(1fr)0.819#h(1fr)0.859],
        [approval], [525], [0.094], [0.066#h(1fr)0.090#h(1fr)0.120], [0.862], [0.830#h(1fr)0.870#h(1fr)0.905],
        [neutral], [2627], [0.088], [0.061#h(1fr)0.084#h(1fr)0.109], [0.899], [0.875#h(1fr)0.906#h(1fr)0.933],
      )
    ],
  ) <tab:predicted-emotion-kl-entropy-quantiles>

  #pagebreak()

  == Translation Samples
  Architecture 5 emotion translation samples: in some cases the model leaves the text unchanged, even when source emotion conflicts with the target. Otherwise, the model edits the sentence in ways unrelated to (or opposite of) the target emotion: eg. substituting #emph[hardly] with #emph[sadly] for a joy target, and deleting a middle clause without emotional restructuring.


  #figure(
    table(
      columns: (1fr, 4fr, 4fr),
      align: (center, left, left),
      inset: 6pt,
      stroke: 0.5pt,

      table.header(header-cell[Target], header-cell[Input], header-cell[Output]),

      [neutral], [Girlfriend weak as well, that jump was pathetic.], [Girlfriend weak as well, that jump was pathetic.],

      [sadness], [Lol! But I love your last name though. XD], [Lol! But I love your last name though. XD],

      [joy],
      [I never wanted to punch osap harder after seeing that However not too #highlight(fill: green)[*hardly*] I cant afford them taking everything away],
      [I never wanted to punch osap harder after seeing that However not too #highlight(fill: green)[*sadly*] I cant afford them taking everything away],

      [sadness],
      [What's that like? Like what's #highlight(fill: yellow)[*the thought process? I dunno. I know what's*] a weird question..i just can't imagine],
      [What's that like? Like what's a weird question#highlight(fill: yellow)[*?*]..i just can't imagine],
    ),
    caption: [Translation (Failure) Examples],
  ) <tab:translation_examples>

]
