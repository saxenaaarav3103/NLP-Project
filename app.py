"""Gradio demo: emotion recognition in dialogue.

    python app.py            # opens http://127.0.0.1:7860
    python app.py --share    # also prints a temporary public link

Tab 1 - classify one message (optionally with the previous turns as context) and
        compare the fine-tuned RoBERTa model with the TF-IDF baseline.
Tab 2 - paste a whole conversation and label every turn.
Tab 3 - what the model is, how well it does, and where it fails.

The look (system font, light/dark colours, cards) lives in assets/app.css; the result
panels are plain HTML built below. Everything is local: no web fonts, CDNs or scripts.
"""
from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path

import gradio as gr

from src.emotion.config import LABELS, RESULTS_DIR
from src.emotion.predict import BaselineClassifier, EmotionClassifier

EMOJI = {"neutral": "😐", "joy": "😄", "surprise": "😮", "anger": "😠",
         "sadness": "😢", "disgust": "🤢", "fear": "😨"}
# One colour per emotion, used for bars, tags and the legend (light, dark).
COLORS = {"neutral": "#8E8E93", "joy": "#E6A700", "surprise": "#0A9AD6", "anger": "#FF3B30",
          "sadness": "#5856D6", "disgust": "#28A745", "fear": "#AF52DE"}
COLORS_DARK = {"neutral": "#98989D", "joy": "#FFD60A", "surprise": "#64D2FF", "anger": "#FF453A",
               "sadness": "#7D7AFF", "disgust": "#30D158", "fear": "#BF5AF2"}
LOW_CONFIDENCE = 0.5
SPEAKER = re.compile(r"^\s*[A-Z][\w .'-]{0,20}:\s+")  # "Rachel: ..." -> "..."
CSS_PATH = Path(__file__).resolve().parent / "assets" / "app.css"

model: EmotionClassifier | None = None  # loaded in __main__ (or by load_models)
baseline: BaselineClassifier | None = None


def load_models(model_dir=None):
    global model, baseline
    model = EmotionClassifier(model_dir) if model_dir else EmotionClassifier()
    baseline = BaselineClassifier()


def _lines(text: str) -> list[str]:
    return [SPEAKER.sub("", line).strip() for line in (text or "").splitlines() if line.strip()]


def _speaker_lines(text: str) -> list[tuple[str, str]]:
    """(speaker or "", turn text) per non-empty line; the text is exactly what _lines gives."""
    out = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = SPEAKER.match(line)
        name = m.group(0).strip().rstrip(":").strip() if m else ""
        out.append((name, SPEAKER.sub("", line).strip()))
    return [(n, t) for n, t in out if t]


# ---------------------------------------------------------------- HTML building blocks

def _pct(p: float) -> str:
    return f"{p:.0%}"


def _tag(emotion: str, conf: float | None = None, cls: str = "") -> str:
    """Pill with glyph + label (+ confidence): the label text is always shown, not just colour."""
    conf_html = f'<span class="er-tag-conf">{_pct(conf)}</span>' if conf is not None else ""
    return (f'<span class="er-tag emo-{emotion} {cls}"><span class="er-tag-glyph" aria-hidden="true">'
            f'{EMOJI[emotion]}</span><span class="er-tag-label">{emotion}</span>{conf_html}</span>')


def _bars(probs: dict[str, float], top: str) -> str:
    rows = []
    for k, v in probs.items():
        rows.append(
            f'<div class="er-bar-row emo-{k}{" is-top" if k == top else ""}">'
            f'<span class="er-bar-name"><span aria-hidden="true">{EMOJI[k]}</span> {k}</span>'
            f'<span class="er-bar-track" role="presentation"><span class="er-bar-fill" '
            f'style="width:{max(v * 100, 0.6):.1f}%"></span></span>'
            f'<span class="er-bar-val">{v:.1%}</span></div>')
    return '<div class="er-bars">' + "".join(rows) + "</div>"


def _empty(title: str, body: str, glyph: str = "💬") -> str:
    return (f'<div class="er-card er-empty"><div class="er-empty-glyph" aria-hidden="true">{glyph}</div>'
            f'<div class="er-empty-title">{title}</div><div class="er-empty-body">{body}</div></div>')


EMPTY_SINGLE = _empty("No prediction yet",
                      "Type a message, or pick one of the examples, then press <b>Detect emotion</b>.")
EMPTY_CONVERSATION = _empty("No labels yet",
                            "Paste a conversation, one turn per line, then press <b>Label every turn</b>.")


def legend_html() -> str:
    return ('<div class="er-legend" aria-label="Emotion colours">'
            + "".join(_tag(k) for k in LABELS) + "</div>")


# ---------------------------------------------------------------- predictions

def classify(message: str, context: str) -> str:
    if not message or not message.strip():
        return _empty("Type a message first",
                      "The message box is empty. Write something, or pick one of the examples.", "✍️")
    turns = _lines(context)
    probs = model.predict(message, context=turns)
    base = baseline.predict(message)
    top, conf = next(iter(probs.items()))
    btop, bconf = next(iter(base.items()))
    n_used = min(len(turns), model.context_turns)
    used = (f"using the previous {n_used} turn{'s' if n_used != 1 else ''} as context"
            if turns else "without context")

    # -- what was classified (user text is escaped: it is echoed into the page)
    ctx_html = ""
    if turns:
        shown = turns[-model.context_turns:]
        ctx_html = ('<div class="er-ctx">' + "".join(
            f'<div class="er-ctx-line">{escape(t)}</div>' for t in shown) + "</div>")
    quote = (f'<div class="er-input">{ctx_html}<div class="er-msg">{escape(message.strip())}</div></div>')

    # -- headline
    low = conf < LOW_CONFIDENCE
    hero = (f'<div class="er-hero emo-{top}">'
            f'<div class="er-hero-glyph" aria-hidden="true">{EMOJI[top]}</div>'
            f'<div class="er-hero-text"><div class="er-eyebrow">Predicted emotion</div>'
            f'<div class="er-hero-label">{top}</div>'
            f'<div class="er-hero-sub">RoBERTa, {used}</div></div>'
            f'<div class="er-hero-conf"><div class="er-hero-pct">{_pct(conf)}</div>'
            f'<div class="er-eyebrow">confidence</div></div></div>')
    note = ('<div class="er-note"><span aria-hidden="true">ⓘ</span> <span><b>Low confidence:</b> '
            'the message is ambiguous, so look at the runner-up emotions too.</span></div>') if low else ""

    # -- with vs without context: the key comparison
    if turns:
        no_ctx = model.predict(message)
        ntop, nconf = next(iter(no_ctx.items()))
        changed = ntop != top
        if changed:
            sentence = (f"Without the previous turns the same model says <b>{ntop}</b> ({_pct(nconf)}), "
                        "so <b>the context changed the decision</b>.")
        else:
            sentence = f"Without the previous turns: also <b>{ntop}</b> ({_pct(nconf)})."
        status = ('<span class="er-status is-changed">Context changed the decision</span>' if changed
                  else '<span class="er-status">Same decision</span>')
        compare = (f'<div class="er-section"><div class="er-section-head"><span class="er-section-title">'
                   f'Effect of the previous turns</span>{status}</div>'
                   f'<div class="er-compare{" is-changed" if changed else ""}">'
                   f'<div class="er-cmp"><div class="er-cmp-label">With context</div>{_tag(top, conf)}</div>'
                   f'<div class="er-cmp-sep" aria-hidden="true">{"≠" if changed else "="}</div>'
                   f'<div class="er-cmp"><div class="er-cmp-label">Without context</div>{_tag(ntop, nconf)}</div>'
                   f'</div><p class="er-cmp-text">{sentence}</p></div>')
    else:
        compare = ('<div class="er-section"><div class="er-section-head"><span class="er-section-title">'
                   'Effect of the previous turns</span><span class="er-status">No context given</span></div>'
                   '<p class="er-cmp-text">This is the message on its own. Add the turns that came before it '
                   'to see whether the context changes the reading.</p></div>')

    bars = (f'<div class="er-section"><div class="er-section-head"><span class="er-section-title">'
            f'All seven emotions</span><span class="er-section-meta">RoBERTa{" + context" if turns else ""}'
            f'</span></div>{_bars(probs, top)}</div>')

    if btop != top:
        btext = ("Disagrees: it only sees surface words, while RoBERTa reads the whole sentence "
                 "(and the context).")
    else:
        btext = "Agrees with RoBERTa."
    base_html = (f'<div class="er-section er-baseline"><div class="er-section-head"><span class="er-section-title">'
                 f'TF-IDF baseline</span><span class="er-section-meta">message only</span></div>'
                 f'<div class="er-base-row">{_tag(btop, bconf)}<span class="er-base-text">{btext}</span></div></div>')

    return f'<div class="er-card er-result">{quote}{hero}{note}{compare}{bars}{base_html}</div>'


def classify_conversation(conversation: str) -> str:
    lines = _speaker_lines(conversation)
    if not lines:
        return _empty("Nothing to label", "Paste a conversation first, one turn per line.", "✍️")
    results = model.predict_conversation([t for _, t in lines])
    items = []
    for i, ((speaker, _), r) in enumerate(zip(lines, results)):
        e, c = r["emotion"], r["confidence"]
        low = c < LOW_CONFIDENCE
        who = escape(speaker) if speaker else f"Turn {i + 1}"
        initial = escape(speaker[:1].upper()) if speaker else str(i + 1)
        items.append(
            f'<li class="er-turn emo-{e}"><div class="er-avatar" aria-hidden="true">{initial}</div>'
            f'<div class="er-turn-main"><div class="er-turn-meta"><span class="er-speaker">{who}</span>'
            f'{f"<span class=er-turn-idx>{i + 1}</span>" if speaker else ""}</div>'
            f'<div class="er-bubble">{escape(r["text"])}</div></div>'
            f'<div class="er-turn-tag">{_tag(e, c, "is-low" if low else "")}'
            f'{"<span class=er-low-text>low confidence</span>" if low else ""}</div></li>')
    n_low = sum(r["confidence"] < LOW_CONFIDENCE for r in results)
    foot = (f"{len(results)} turns. Each turn is labelled using only the turns before it "
            f"(up to {model.context_turns}).")
    if n_low:
        foot += f" Dashed tags: confidence below {LOW_CONFIDENCE:.0%}."
    return (f'<div class="er-card er-transcript">{legend_html()}<ol class="er-turns">{"".join(items)}</ol>'
            f'<div class="er-foot">{foot}</div></div>')


# ---------------------------------------------------------------- about tab

def _runs(name: str) -> list[dict]:
    """Metrics of a configuration: the main run plus any extra seeds that exist."""
    paths = [RESULTS_DIR / f"{name}{suffix}_metrics.json" for suffix in ("", "_seed13", "_seed7")]
    return [json.loads(p.read_text()) for p in paths if p.exists()]


def about_sections() -> list[tuple[str, str]]:
    """(title, markdown body) for each section of the About tab."""
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
    table = ("| Model | Accuracy | Weighted F1 | Macro F1 |\n|:---|---:|---:|---:|\n" + "\n".join(rows) + demo_note) if rows else ""
    return [
        ("What it does", f"""
Predicts which of **7 emotions** ({", ".join(LABELS)}) a speaker expresses in a dialogue turn.
"""),
        ("How it works", """
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
"""),
        ("Test-set results (2,610 unseen utterances)", f"""
{table}

Weighted F1 is the standard MELD metric; macro F1 weighs all seven emotions equally.
"""),
        ("Limitations", """
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
"""),
    ]


def about_markdown() -> str:
    return "".join(f"\n## {title}\n{body}" for title, body in about_sections())


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
EXAMPLE_LABELS = [
    "I failed the exam again…",
    "I got the job!!",
    "“Come on!” (no context)",
    "“Come on!” after a complaint",
    "“Come on!” after good news",
    "Ew, ew, ew!…",
    "I'm scared…",
    "Oh great, another meeting. (sarcasm)",
]

CONVERSATION = """Monica: Guess what? I got the head chef job!
Rachel: Oh my God! That's amazing!
Monica: I start Monday. I'm kind of terrified, actually.
Rachel: You'll be great. But wait, does this mean you're moving out?
Monica: Well... yes.
Rachel: You're just telling me this now?!"""

HEADER = """
<header class="er-header">
  <div class="er-kicker">NLP project · MELD</div>
  <h1>Emotion Recognition in Dialogue</h1>
  <p>A fine-tuned RoBERTa model reads a message together with the two turns before it, and
  names one of seven emotions. Trained on <i>Friends</i> TV-show dialogues; shown next to a
  TF-IDF baseline.</p>
</header>
"""


def theme() -> gr.themes.Base:
    font = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Helvetica Neue", "Arial", "sans-serif"]
    mono = ["ui-monospace", "SF Mono", "Menlo", "monospace"]
    return gr.themes.Base(primary_hue="blue", neutral_hue="gray", font=font, font_mono=mono).set(
        body_background_fill="#F5F5F7", body_background_fill_dark="#000000",
        body_text_color="#1D1D1F", body_text_color_dark="#F5F5F7",
        body_text_color_subdued="#6E6E73", body_text_color_subdued_dark="#A1A1A6",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#1C1C1E",
        background_fill_secondary="#F5F5F7", background_fill_secondary_dark="#2C2C2E",
        block_background_fill="#FFFFFF", block_background_fill_dark="#1C1C1E",
        block_border_color="rgba(0,0,0,0.08)", block_border_color_dark="rgba(255,255,255,0.10)",
        border_color_primary="rgba(0,0,0,0.10)", border_color_primary_dark="rgba(255,255,255,0.12)",
        border_color_accent="#0071E3", border_color_accent_dark="#2997FF",
        color_accent="#0071E3", color_accent_soft="rgba(0,113,227,0.10)",
        color_accent_soft_dark="rgba(41,151,255,0.16)",
        block_label_text_color="#6E6E73", block_label_text_color_dark="#A1A1A6",
        block_title_text_color="#6E6E73", block_title_text_color_dark="#A1A1A6",
        input_background_fill="#FFFFFF", input_background_fill_dark="#2C2C2E",
        input_border_color="rgba(0,0,0,0.12)", input_border_color_dark="rgba(255,255,255,0.12)",
        input_border_color_focus="#0071E3", input_border_color_focus_dark="#2997FF",
        input_placeholder_color="#8E8E93", input_placeholder_color_dark="#8E8E93",
        button_primary_background_fill="#0071E3", button_primary_background_fill_dark="#0071E3",
        button_primary_background_fill_hover="#0077ED", button_primary_background_fill_hover_dark="#0A84FF",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
        button_secondary_background_fill="#E8E8ED", button_secondary_background_fill_dark="#2C2C2E",
        button_secondary_text_color="#1D1D1F", button_secondary_text_color_dark="#F5F5F7",
        loader_color="#0071E3", loader_color_dark="#2997FF",
        link_text_color="#0066CC", link_text_color_dark="#2997FF",
        table_border_color="rgba(0,0,0,0.08)", table_border_color_dark="rgba(255,255,255,0.10)",
        table_even_background_fill="#FFFFFF", table_even_background_fill_dark="#1C1C1E",
        table_odd_background_fill="#FAFAFC", table_odd_background_fill_dark="#232325",
        shadow_drop="0 1px 3px rgba(0,0,0,0.06)", shadow_drop_lg="0 4px 16px rgba(0,0,0,0.06)",
    )


def css() -> str:
    """Stylesheet: assets/app.css plus the per-emotion colour variables generated from COLORS."""
    light = "".join(f"--emo-{k}:{v};" for k, v in COLORS.items())
    dark = "".join(f"--emo-{k}:{v};" for k, v in COLORS_DARK.items())
    per = "".join(f".emo-{k}{{--c:var(--emo-{k});}}" for k in COLORS)
    return f":root{{{light}}}\n.dark{{{dark}}}\n{per}\n" + CSS_PATH.read_text(encoding="utf-8")


def build() -> gr.Blocks:
    with gr.Blocks(title="Emotion Recognition in Dialogue") as demo:
        gr.HTML(HEADER)
        with gr.Tabs(elem_classes="er-tabs"):
            with gr.Tab("Single message"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=380, elem_classes="er-panel"):
                        msg = gr.Textbox(label="Message", lines=2, autofocus=True,
                                         placeholder="e.g. I got the job!! I'm so excited!")
                        ctx = gr.Textbox(label="Previous turns (optional, one per line, oldest first)", lines=3,
                                         placeholder="Ross: I'm getting married next week.")
                        btn = gr.Button("Detect emotion", variant="primary", elem_classes="er-btn")
                        out = gr.HTML(EMPTY_SINGLE, elem_classes="er-out", render=False)
                        gr.Examples(EXAMPLES, inputs=[msg, ctx], outputs=out, fn=classify,
                                    run_on_click=True, cache_examples=False, label="Try these",
                                    example_labels=EXAMPLE_LABELS, examples_per_page=len(EXAMPLES))
                    with gr.Column(scale=6, min_width=380):
                        out.render()
                btn.click(classify, [msg, ctx], out, show_progress="hidden")
                msg.submit(classify, [msg, ctx], out, show_progress="hidden")
            with gr.Tab("Whole conversation"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=340, elem_classes="er-panel"):
                        conv = gr.Textbox(label="Conversation (one turn per line; 'Name:' prefixes are optional)",
                                          lines=9, value=CONVERSATION)
                        btn2 = gr.Button("Label every turn", variant="primary", elem_classes="er-btn")
                    with gr.Column(scale=7, min_width=420):
                        transcript = gr.HTML(EMPTY_CONVERSATION, elem_classes="er-out")
                btn2.click(classify_conversation, conv, transcript, show_progress="hidden")
            with gr.Tab("About the model"):
                (t1, b1), (t2, b2), (t3, b3), (t4, b4) = about_sections()
                with gr.Row(equal_height=False, elem_classes="er-about"):
                    with gr.Column(scale=1, min_width=360, elem_classes="er-about-col"):
                        with gr.Column(elem_classes="er-about-card"):
                            gr.Markdown(f"### {t1}\n{b1}")
                        with gr.Column(elem_classes="er-about-card"):
                            gr.Markdown(f"### {t2}\n{b2}")
                    with gr.Column(scale=1, min_width=360, elem_classes="er-about-col"):
                        with gr.Column(elem_classes="er-about-card er-results"):
                            gr.Markdown(f"### {t3}\n{b3}")
                with gr.Column(elem_classes="er-about-card er-limits"):
                    gr.Markdown(f"### {t4}\n{b4}")
    return demo


def launch_kwargs() -> dict:
    return dict(theme=theme(), css=css(), footer_links=[])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model-dir", default=None, help="fine-tuned model (default: models/roberta-meld-context)")
    args = ap.parse_args()
    load_models(args.model_dir)
    build().launch(share=args.share, server_port=args.port, **launch_kwargs())
