"""Gradio demo: emotion recognition in dialogue.

    python app.py            # opens http://127.0.0.1:7860
    python app.py --share    # also prints a temporary public link

Tab 1 - classify one message (optionally with the previous turns as context) and
        compare the fine-tuned RoBERTa model with the TF-IDF baseline.
Tab 2 - paste a whole conversation and label every turn.
Tab 3 - what the model is, how well it does, and where it fails.
"""
from __future__ import annotations

import argparse
import json
import re

import gradio as gr

from src.emotion.config import LABELS, RESULTS_DIR
from src.emotion.predict import BaselineClassifier, EmotionClassifier

EMOJI = {"neutral": "😐", "joy": "😄", "surprise": "😮", "anger": "😠",
         "sadness": "😢", "disgust": "🤢", "fear": "😨"}
COLORS = {"neutral": "#9ca3af", "joy": "#facc15", "surprise": "#38bdf8", "anger": "#ef4444",
          "sadness": "#6366f1", "disgust": "#22c55e", "fear": "#a855f7"}
SPEAKER = re.compile(r"^\s*[A-Z][\w .'-]{0,20}:\s+")  # "Rachel: ..." -> "..."

model: EmotionClassifier | None = None  # loaded in __main__ (or by load_models)
baseline: BaselineClassifier | None = None


def load_models(model_dir=None):
    global model, baseline
    model = EmotionClassifier(model_dir) if model_dir else EmotionClassifier()
    baseline = BaselineClassifier()


def _lines(text: str) -> list[str]:
    return [SPEAKER.sub("", line).strip() for line in (text or "").splitlines() if line.strip()]


def classify(message: str, context: str):
    if not message or not message.strip():
        return {}, {}, "Type a message first."
    turns = _lines(context)
    probs = model.predict(message, context=turns)
    base = baseline.predict(message)
    top, conf = next(iter(probs.items()))
    btop = next(iter(base))
    used = (f"using the previous {min(len(turns), model.context_turns)} turn(s) as context"
            if turns else "without context")
    text = f"### {EMOJI[top]} **{top}** ({conf:.0%} confidence, {used})\n"
    if turns:  # show what the same model says when the context is removed
        no_ctx = model.predict(message)
        ntop, nconf = next(iter(no_ctx.items()))
        if ntop != top:
            text += (f"\nWithout the previous turns the same model says **{ntop}** ({nconf:.0%}), "
                     "so **the context changed the decision**.\n")
        else:
            text += f"\nWithout the previous turns: also **{ntop}** ({nconf:.0%}).\n"
    if btop != top:
        text += (f"\nThe TF-IDF baseline says **{btop}**: it only sees surface words, "
                 "while RoBERTa reads the whole sentence (and the context).")
    if conf < 0.5:
        text += "\n\n*Low confidence: the message is ambiguous, so look at the runner-up emotions too.*"
    return ({f"{EMOJI[k]} {k}": v for k, v in probs.items()},
            {f"{EMOJI[k]} {k}": v for k, v in base.items()}, text)


def classify_conversation(conversation: str):
    turns = _lines(conversation)
    if not turns:
        return [], []
    results = model.predict_conversation(turns)
    highlighted = []
    for r in results:
        highlighted.append((r["text"], r["emotion"]))
        highlighted.append(("\n", None))
    table = [[i + 1, r["text"], f"{EMOJI[r['emotion']]} {r['emotion']}", f"{r['confidence']:.0%}"]
             for i, r in enumerate(results)]
    return highlighted, table


def _runs(name: str) -> list[dict]:
    """Metrics of a configuration: the main run plus any extra seeds that exist."""
    paths = [RESULTS_DIR / f"{name}{suffix}_metrics.json" for suffix in ("", "_seed13", "_seed7")]
    return [json.loads(p.read_text()) for p in paths if p.exists()]


def about_markdown() -> str:
    rows = []
    for name, title in [("majority", "Always 'neutral'"), ("tfidf_logreg", "TF-IDF + Logistic Regression"),
                        ("roberta_utterance", "RoBERTa (utterance only)"),
                        ("roberta_context", "**RoBERTa + 2 previous turns (this demo)**")]:
        runs = _runs(name)
        if runs:
            avg = {k: sum(r[k] for r in runs) / len(runs) for k in ("accuracy", "weighted_f1", "macro_f1")}
            note = f" (mean of {len(runs)} runs)" if len(runs) > 1 else ""
            rows.append(f"| {title}{note} | {avg['accuracy']:.3f} | {avg['weighted_f1']:.3f} | {avg['macro_f1']:.3f} |")
    demo = _runs("roberta_context")[:1]
    demo_note = (f"\n\nThe model running in this demo is one of those runs (seed 42): accuracy "
                 f"{demo[0]['accuracy']:.3f}, weighted F1 {demo[0]['weighted_f1']:.3f}." if demo else "")
    table = ("| Model | Accuracy | Weighted F1 | Macro F1 |\n|---|---|---|---|\n" + "\n".join(rows) + demo_note) if rows else ""
    return f"""
## What it does
Predicts which of **7 emotions** ({", ".join(LABELS)}) a speaker expresses in a dialogue turn.

## How it works
* **Data:** MELD - 13,708 utterances from 1,433 dialogues of the TV series *Friends*, each labelled
  with one emotion (train 9,989 / dev 1,109 / test 2,610).
* **Model:** RoBERTa-base (a 125M-parameter transformer pre-trained on 160 GB of English text),
  fine-tuned to classify the target utterance **together with the two previous turns** of the
  conversation. Pre-training gives it general language understanding (negation, intensity,
  idioms); fine-tuning teaches it which cues signal which emotion; the context can change the
  reading of an ambiguous reply: "Come on!" alone is a toss-up between joy and anger, after a
  complaint it is anger, after good news it is joy.
* **Imbalance:** neutral is 47% of the data while fear/disgust are under 3% each, so the loss
  up-weights rare classes (1/sqrt(frequency)).

## Test-set results (2,610 unseen utterances)
{table}

Weighted F1 is the standard MELD metric; macro F1 weighs all seven emotions equally.

## Limitations
* Text only: MELD labels were made while *watching* the show, so tone of voice and facial
  expressions are missing; sarcasm is often misread.
* Rare emotions (fear, disgust) have few training examples and the lowest scores; fear is
  often read as anger or sadness.
* Context is used mostly as a positive/negative signal: after sad news a short reply tends to
  become *anger* rather than *sadness*, and strongly marked replies ("Seriously?", "Oh my God.")
  stay *surprise* whatever came before.
* Averaged over three training runs, this context model is **not more accurate** than the same
  model trained without context (weighted F1 about 0.625 for both). Context helps on some
  ambiguous replies and hurts on others.
* One label per turn: mixed feelings ("happy but nervous") are forced into one class.
* Trained on a 1990s US sitcom: slang, other dialects and other domains may transfer poorly.
"""


EXAMPLES = [
    ["I can't believe I failed the exam again. I'm so disappointed in myself.", ""],
    ["I got the job!! I'm so excited!", ""],
    ["Come on!", ""],
    ["Come on!", "Ross: What a jerk!\nRachel: I cannot believe he's using our divorce to sell mattresses."],
    ["Come on!", "Joey: We won! We actually won the tickets!\nChandler: This is going to be the best weekend ever!"],
    ["Ew, ew, ew! That's the grossest thing I've ever seen!", ""],
    ["I'm scared. What if something goes wrong?", ""],
    ["Oh great, another meeting. Exactly what I needed today.", ""],
]

CONVERSATION = """Monica: Guess what? I got the head chef job!
Rachel: Oh my God! That's amazing!
Monica: I start Monday. I'm kind of terrified, actually.
Rachel: You'll be great. But wait, does this mean you're moving out?
Monica: Well... yes.
Rachel: You're just telling me this now?!"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Emotion Recognition in Dialogue") as demo:
        gr.Markdown("# 🎭 Emotion Recognition in Dialogue\nFine-tuned RoBERTa on the MELD "
                    "dataset (*Friends* TV-show dialogues) - 7 emotions, context-aware.")
        with gr.Tab("Single message"):
            with gr.Row():
                with gr.Column():
                    msg = gr.Textbox(label="Message", lines=2,
                                     placeholder="e.g. I got the job!! I'm so excited!")
                    ctx = gr.Textbox(label="Previous turns (optional, one per line, oldest first)", lines=3,
                                     placeholder="Ross: I'm getting married next week.")
                    btn = gr.Button("Detect emotion", variant="primary")
                    gr.Examples(EXAMPLES, inputs=[msg, ctx], label="Try these")
                with gr.Column():
                    summary = gr.Markdown()
                    out_model = gr.Label(num_top_classes=7, label="RoBERTa + context (main model)")
                    out_base = gr.Label(num_top_classes=3, label="TF-IDF baseline (utterance only)")
            btn.click(classify, [msg, ctx], [out_model, out_base, summary])
            msg.submit(classify, [msg, ctx], [out_model, out_base, summary])
        with gr.Tab("Whole conversation"):
            conv = gr.Textbox(label="Conversation (one turn per line; 'Name:' prefixes are optional)",
                              lines=8, value=CONVERSATION)
            btn2 = gr.Button("Label every turn", variant="primary")
            hl = gr.HighlightedText(label="Emotion per turn", color_map=COLORS, show_legend=True)
            tbl = gr.Dataframe(headers=["#", "Turn", "Emotion", "Confidence"], interactive=False, wrap=True)
            btn2.click(classify_conversation, conv, [hl, tbl])
        with gr.Tab("About the model"):
            gr.Markdown(about_markdown())
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model-dir", default=None, help="fine-tuned model (default: models/roberta-meld-context)")
    args = ap.parse_args()
    load_models(args.model_dir)
    build().launch(share=args.share, server_port=args.port, theme=gr.themes.Soft())
