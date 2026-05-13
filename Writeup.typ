#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#add-bib-resource(read("sources.bib"))
#set heading(numbering: "1")

#show: doc => acl(
  doc,
  anonymous: false,
  title: [Evaluating Transformer Embedding Disentanglement \ using Emotion Translation],
  authors: make-authors(
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
  [

  ],
)

= Introduction

= Dataset

= Tasks

= Architectures
.

*Classification* 

*Reconstruction* 

*Robust Translation*

= Contributions

= Related Work

= Future Work

= Conclusion


#print-bibliography()