# Demo guide (about 3 minutes, plus Q&A)

The rubric asks for "a brief demo of the working application" covering **what it does, why it
works, and limitations**. This page is a script for exactly that.

## Before the demo (do this once, the day before)

```bash
pip install -r requirements.txt
python app.py            # wait for "Running on local URL: http://127.0.0.1:7860", open it
```

The first start takes about 20 s while the model loads. Click one example and check that you
get a prediction. Keep the browser tab open. If you are demoing from a different laptop, copy
the whole repository, including `models/`.

## The 3-minute script

**1. What it does (about 20 s).** Open the *Single message* tab.
> "This app detects the emotion a speaker expresses in a conversation, one of seven classes:
> neutral, joy, surprise, anger, sadness, disgust and fear. It's a RoBERTa transformer that I
> fine-tuned on MELD, a dataset of about 13,000 dialogue turns from the TV show *Friends*."

**2. Basic predictions (about 30 s).** Click the first two examples (the exam and job
sentences from the assignment):
- "I can't believe I failed the exam again..." gives **sadness**.
- "I got the job!! I'm so excited!" gives **joy**.

> "On the right you can see the probability for every emotion, and underneath, what my
> classical baseline (TF-IDF + logistic regression) predicts for the same text."

**3. Why context matters (about 45 s). This is the strongest moment of the demo.** Click the
two **"What?"** examples one after the other.
> "The message is identical, just 'What?', but the previous turns differ. After 'I'm getting
> married next week' the model reads it as surprise. After 'I ate your sandwich and used your
> toothbrush' the emotion shifts toward anger or disgust. The model reads the last two turns
> of the conversation, and that's the main design decision of the project. In my experiments,
> adding context improved the scores most on short replies, and short replies are where
> emotion is most ambiguous."

(Check before the demo what the model actually outputs for these, and phrase it accordingly.)

**4. Whole conversations (about 30 s).** Open the *Whole conversation* tab and click **Label every
turn**.
> "For a full dialogue, each turn is labelled using only the turns before it, so this could
> run live inside a chatbot to respond empathetically or flag angry customers."

**5. Why it works and its limitations (about 45 s).** Open the *About the model* tab.
> "Why it works: RoBERTa was pre-trained on 160 GB of text, so it already understands negation,
> intensity and idioms. Fine-tuning teaches it which cues map to which emotion, and class
> weighting stops it from just predicting 'neutral', which is half the data. On the 2,610
> unseen test utterances it clearly beats the baseline; the table shows accuracy, weighted F1
> and macro F1."
>
> "Limitations: it's text only. The dataset was labelled while watching the video, so tone
> of voice is missing and sarcasm fools it." Go back to tab 1 and click the sarcasm example
> ("Oh great, another meeting...").
> "The rare emotions, fear and disgust, have only about 270 training examples each and are the
> weakest. It also gives one label per turn, so mixed feelings can't be expressed, and it was
> trained on a 90s sitcom, so slang or Hinglish may not transfer."

## Likely questions, with short answers

**Why did you pick this dataset?**
The task says "from dialogues", and MELD is the standard benchmark for emotion recognition in
conversation. It has real multi-turn dialogues, a label for every turn and an official
train/dev/test split, so my numbers are comparable with published work.

**Why RoBERTa and not BERT or an LLM?**
RoBERTa is BERT trained longer, on more data, with better settings, and it is consistently
stronger on classification. It fine-tunes on a CPU in about 40 minutes. A large LLM would be
far slower, could not be fine-tuned on my hardware, and a zero-shot LLM has no access to
MELD's labelling conventions.

**How does the model use context?**
The input is a pair: `<s> current utterance </s></s> previous two turns </s>`. Self-attention
lets every token of the current utterance look at the context tokens. The classifier reads
the `<s>` vector. Only past turns are used, never future ones.

**Why accuracy and two kinds of F1?**
Accuracy is misleading with imbalanced classes: always saying "neutral" already gives 48%.
Weighted F1 (averaged by class size) is the standard MELD metric. Macro F1 treats all seven
emotions equally, so it shows whether the rare emotions work.

**What is precision vs recall?**
For a class such as anger: precision is the share of "anger" predictions that were really
anger. Recall is the share of real anger turns that the model found. F1 is their harmonic mean.

**How did you handle class imbalance?**
The cross-entropy loss is weighted by 1/sqrt(class frequency). Fully inverse weights made the
model over-predict rare classes, and no weights made it ignore them. There is also an
ablation without weights in `results/`.

**How do you know you're not overfitting to the test set?**
All choices (baseline regularisation, best training epoch) were made on the dev set. The test
set was evaluated once at the end. I also repeated training with 3 random seeds and report the
mean and standard deviation.

**Why remove speaker names?**
With names, the model could learn shortcuts like "Joey is usually happy", which would not work
for real users.

**What were the hardest classes and why?**
Fear and disgust. They have very few examples and share cues with surprise and anger ("Oh my
God!"). Most errors in every class are confusions with neutral, because mild emotion often
looks neutral in text.

**Where did the pretrained weights come from?**
`roberta-base`. The cloud machine I trained on could not reach the Hugging Face Hub, so I
extracted the same RoBERTa-base encoder from spaCy's English transformer package (downloaded
from GitHub) and converted it. `scripts/verify_conversion.py` shows the converted model gives
the same outputs as the original implementation at every layer. With normal internet access,
`--base-model roberta-base` downloads it directly.

**How could it be improved?**
A larger model (roberta-large), more context turns, speaker-aware context, data augmentation
or focal loss for rare classes, and adding audio features (MELD includes them).
