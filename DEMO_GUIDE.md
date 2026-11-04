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

**3. Context (about 45 s). This is the most visual moment of the demo.** Click the three
**"Come on!"** examples one after the other:
- No context: a toss-up, roughly 42% joy vs 41% anger.
- After "What a jerk! I cannot believe he's using our divorce to sell mattresses.": **anger**
  (about 69%).
- After "We won! We actually won the tickets!": **joy** (about 66%).

The text under the prediction says what the same model would answer *without* the previous
turns, so the audience can see that the context changed the decision.

> "The message is identical; only the previous turns differ. The model reads the last two
> turns of the conversation. Removing the context from this model costs it about 1.5 F1
> points, so it really uses it. But here's an honest finding: when I trained the same model
> *without* context, three times with different random seeds, it was just as accurate on
> average. A single run had suggested context helps, and the repeats showed that was luck. So
> context helps on ambiguous replies like this one, but in this setup it isn't a reliable
> overall gain."

**4. Whole conversations (about 30 s).** Open the *Whole conversation* tab and click **Label every
turn**.
> "For a full dialogue, each turn is labelled using only the turns before it, so this could
> run live inside a chatbot to respond empathetically or flag angry customers."

Expect turn 3 ("I'm kind of terrified, actually.") to come out as *sadness* rather than
*fear*. Don't hide it: it's a live example of the weakest class and a good bridge to step 5.

**5. Why it works and its limitations (about 45 s).** Open the *About the model* tab.
> "Why it works: RoBERTa was pre-trained on 160 GB of text, so it already understands negation,
> intensity and idioms. Fine-tuning teaches it which cues map to which emotion, and class
> weighting stops it from just predicting 'neutral', which is half the data. On the 2,610
> unseen test utterances it clearly beats the baseline; the table shows accuracy, weighted F1
> and macro F1."
>
> "Limitations: it's text only. The dataset was labelled while watching the video, so tone
> of voice is missing and sarcasm fools it." Go back to tab 1 and click the sarcasm example
> ("Oh great, another meeting..."); the model says *joy*.
> "The rare emotions, fear and disgust, have only about 270 training examples each and are the
> weakest. Fear is mostly read as neutral or anger. The model also uses context mostly as a
> positive-or-negative signal: after sad news, 'Come on!' becomes anger rather than sadness.
> It gives one label per turn, so mixed feelings can't be expressed, and it was trained on a
> 90s sitcom, so slang or Hinglish may not transfer."

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
Fear and disgust. They have very few examples (50 and 68 in the test set) and share cues with
surprise and anger ("Oh my God!"). For five of the six emotions the most common error is
predicting neutral, because mild emotion often looks neutral in text.

**Does the model really use the context? Does context help?**
Two different questions, and two different answers:
- *Does it use it?* Yes. With the context removed, the shipped model's weighted-F1 drops
  from 0.633 to 0.618, and 289 test predictions change (110 fixed, 84 broken). It flips
  ambiguous replies such as "Come on!" or "I know!" between joy and anger.
- *Does training with context beat training without it?* Not in my experiments. Over three
  seeds the averages are practically identical (about 0.625 weighted-F1 each); context won on
  one of three seed pairs. A model trained without context learns to get the same accuracy
  from the utterance alone.
- *Why might that be?* MELD labels often depend on tone of voice, which no text contains. My
  context has no speaker identities, so the model can't tell a reply from a continuation.
  Published context gains on MELD use larger, speaker-aware models.
- *Why keep the context model, then?* It's equally accurate, and it handles cases where the
  wording alone is ambiguous.

**Why did you run three seeds?**
Fine-tuning results vary from run to run (here by about ±0.7 F1 points), which is as large as
the effect I wanted to measure. With one seed I would have reported a 1.3-point gain from
context that doesn't exist.

**Where did the pretrained weights come from?**
RoBERTa-base. The cloud machine I trained on could not reach the Hugging Face Hub, so I
extracted the RoBERTa-base encoder bundled in spaCy's English transformer package (downloaded
from GitHub) and converted it to the Hugging Face format. spaCy had trained it a bit further
for its own tagging and parsing tasks, so it's not bit-identical to the original roberta-base.
`scripts/verify_conversion.py` shows the converted model gives the same outputs as spaCy's
implementation at every layer. With normal internet access, `--base-model roberta-base`
downloads the original directly.

**How could it be improved?**
A larger model (roberta-large), more context turns, speaker-aware context, data augmentation
or focal loss for rare classes, and adding audio features (MELD includes them).
