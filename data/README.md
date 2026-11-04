# Data: MELD (text part)

`raw/{train,dev,test}_sent_emo.csv` are the official splits of **MELD - Multimodal EmotionLines
Dataset** (Poria et al., ACL 2019), downloaded unchanged from
<https://github.com/declare-lab/MELD/tree/master/data/MELD> (GPL-3.0 licence).

| split | utterances | dialogues |
|---|---|---|
| train | 9,989 | 1,038 |
| dev | 1,109 | 114 |
| test | 2,610 | 280 |

Columns used: `Utterance` (text), `Speaker`, `Emotion` (label: neutral, joy, surprise, anger,
sadness, disgust, fear), `Dialogue_ID`, `Utterance_ID` (order within the dialogue). The
`Sentiment` and video time-stamp columns are not used.

> S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. *MELD: A Multimodal
> Multi-Party Dataset for Emotion Recognition in Conversations.* ACL 2019.
