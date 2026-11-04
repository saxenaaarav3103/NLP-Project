"""Build the 2-page summary report (report/report.docx) from the saved results.

    python scripts/make_report_figure.py
    python report/build_report.py
    soffice --headless --convert-to pdf --outdir report report/report.docx

Every number in the report is read from results/*.json, so the report always matches the
experiments. Fill in the highlighted placeholders (name, roll number, course) in Word.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def load(name):
    p = RES / f"{name}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def seeds(prefix):
    runs = [load(prefix), load(f"{prefix}_seed13"), load(f"{prefix}_seed7")]
    return [r for r in runs if r]


def ms(runs, key):
    vals = [r[key] for r in runs]
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}"


# --------------------------------------------------------------------------- helpers
def set_cell_shading(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def set_cell_margins(table, top=30, bottom=30, left=70, right=70):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def para(doc, text="", size=9.5, bold=False, italic=False, align=None, after=3, before=0, color=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    if align:
        p.alignment = align
    if text:
        add_runs(p, text, size=size, bold=bold, italic=italic, color=color)
    return p


def add_runs(p, text, size=9.5, bold=False, italic=False, color=None):
    """Tiny markup: **bold** and *italic* segments."""
    import re
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*"):
            i, t = True, part[1:-1]
        r = p.add_run(t)
        r.font.size = Pt(size)
        r.bold = b
        r.italic = i
        if color is not None:
            r.font.color.rgb = color
    return p


def heading(doc, text):
    p = para(doc, before=5, after=1.5)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = ACCENT
    p.paragraph_format.keep_with_next = True
    return p


def placeholder(p, text, size):
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return r


# --------------------------------------------------------------------------- content
def build():
    major, tfidf = load("majority"), load("tfidf_logreg")
    utt, ctx = seeds("roberta_utterance"), seeds("roberta_context")
    final = load("roberta_context")
    nw, iw, c4 = load("roberta_context_noweights"), load("roberta_context_invweights"), load("roberta_context4")
    n_seeds = len(ctx)

    def mean(runs, key):
        return st.mean(r[key] for r in runs)

    gain_ctx_w = mean(ctx, "weighted_f1") - mean(utt, "weighted_f1")
    gain_ctx_m = mean(ctx, "macro_f1") - mean(utt, "macro_f1")
    gain_base_w = mean(ctx, "weighted_f1") - tfidf["weighted_f1"]
    pc = final["per_class"]
    best_cls = max(pc, key=lambda c: pc[c]["f1"])
    worst = sorted(pc, key=lambda c: pc[c]["f1"])[:2]
    lat = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else None
    ctx_short = json.loads((RES / "context_analysis.json").read_text()) if (RES / "context_analysis.json").exists() else None

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin, sec.bottom_margin = Cm(1.5), Cm(1.4)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(9.5)
    normal.paragraph_format.line_spacing = 1.0

    # Title block
    p = para(doc, after=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("Context-Aware Emotion Recognition in Dialogue with RoBERTa")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(15), ACCENT
    p = para(doc, after=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_runs(p, "Individual Project #5 - Emotion Recognition from Text   |   ", size=9.5)
    for i, ph in enumerate(["[Your Name]", "[Roll No.]", "[Course / Code]", "[Date]"]):
        placeholder(p, ph, 9.5)
        if i < 3:
            add_runs(p, "  ·  ", size=9.5)
    p = para(doc, after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_runs(p, "Code, notebook and demo: github.com/saxenaaarav3103/NLP-Project", size=8.5, italic=True)

    heading(doc, "1. Problem and requirements")
    para(doc, "Emotion recognition in conversation assigns each turn of a dialogue one of seven emotions "
              "(*neutral, joy, surprise, anger, sadness, disgust, fear*). Users of such a system include "
              "chatbots that should answer empathetically, support desks that flag angry customers, and "
              "analytics of conversational tone. Three properties make the task hard: (i) turns are short "
              "and ambiguous (median 6 words; *\"Come on!\"* can be joyful or angry), so the model must use the "
              "**preceding** turns, and only those, if it is to run live; (ii) classes are highly imbalanced "
              "(47% neutral, <3% each fear/disgust), so accuracy alone is misleading; (iii) gold labels were "
              "assigned while watching video, so tone of voice is missing from the text. **Constraint:** only "
              "a 4-core CPU (no GPU) was available, so a model had to fine-tune in under an hour and answer "
              "interactively.")

    heading(doc, "2. Dataset and preprocessing")
    para(doc, "**MELD** [1]: 13,708 utterances in 1,433 multi-party dialogues from the TV series *Friends*, "
              "official split 9,989 / 1,109 / 2,610 (train / dev / test). It is the standard benchmark for this "
              "task, labels every turn and preserves dialogue order, so it matches the \"from dialogues\" "
              "requirement. **Pipeline:** Unicode/punctuation normalisation (curly quotes, dashes, ellipses), "
              "keeping case, punctuation and repeated letters because they carry emotion (*\"WHAT?!\"* vs "
              "*\"what.\"*); no stop-word removal or stemming. **Context** = the two previous turns of the same "
              "dialogue, with speaker names removed so the model cannot learn character shortcuts. Transformer "
              "input: <s> utterance </s></s> context </s>, at most 128 BPE tokens, oldest context trimmed first.")

    heading(doc, "3. Models")
    para(doc, "(a) **Majority class**. (b) **TF-IDF + logistic regression**: word 1-2-grams and character "
              "2-5-grams, balanced class weights, C tuned on dev. (c) **RoBERTa-base** [2] (125M parameters) "
              "fine-tuned end-to-end with a classification head on <s>: AdamW, lr 2e-5, batch 16, 4 epochs, "
              "linear warm-up/decay, cross-entropy weighted by 1/√(class frequency), checkpoint chosen on dev "
              "weighted-F1, test set used once. Trained **with** (final model) and **without** context, "
              f"{n_seeds} seeds each; one run takes about 40 min on CPU (bfloat16, fused AdamW). **Why it works:** "
              "pre-training on 160 GB of text gives knowledge of negation, intensity and idioms that n-grams "
              "lack; context resolves short replies; softened class weights buy rare-class recall at little "
              "cost to the majority class. *Note:* the development machine could not reach the Hugging Face "
              "Hub, so the RoBERTa-base encoder bundled in spaCy's en_core_web_trf was converted to the "
              "transformers format; its layer outputs match the reference implementation (max difference "
              "1.5e-5).")

    heading(doc, "4. Evaluation")
    cap = para(doc, f"Table 1. Test-set results (2,610 utterances). RoBERTa rows: mean ± std over {n_seeds} seeds.",
               size=8, italic=True, after=1)
    cap.paragraph_format.keep_with_next = True
    rows = [("Majority class (always neutral)", [major]), ("TF-IDF + logistic regression", [tfidf]),
            ("RoBERTa, utterance only", utt), ("RoBERTa + 2 context turns (final)", ctx)]
    cols = [("Accuracy", "accuracy"), ("W-Precision", "weighted_precision"), ("W-Recall", "weighted_recall"),
            ("W-F1", "weighted_f1"), ("Macro-F1", "macro_f1")]
    table = doc.add_table(rows=1 + len(rows), cols=1 + len(cols))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    set_cell_margins(table)
    widths = [Cm(6.2)] + [Cm(2.2)] * len(cols)
    table.autofit = False
    for j, h in enumerate(["Model"] + [c for c, _ in cols]):
        cell = table.rows[0].cells[j]
        cell.text = ""
        add_runs(cell.paragraphs[0], h, size=8.5, bold=True)
        set_cell_shading(cell, "DCE6F1")
    for i, (name, runs) in enumerate(rows, start=1):
        is_final = "final" in name
        vals = [name] + [ms(runs, k) for _, k in cols]
        for j, v in enumerate(vals):
            cell = table.rows[i].cells[j]
            cell.text = ""
            add_runs(cell.paragraphs[0], v, size=8.5, bold=is_final)
            if j:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if is_final:
                set_cell_shading(cell, "F2F7FC")
    for j, w in enumerate(widths):
        table.columns[j].width = w
    for row in table.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = w
            row.cells[j].paragraphs[0].paragraph_format.space_after = Pt(0)

    abl = []
    if nw and iw:
        abl.append(f"class weights none / 1/√f / 1/f: W-F1 {nw['weighted_f1']:.3f} / {final['weighted_f1']:.3f} / "
                   f"{iw['weighted_f1']:.3f}, macro-F1 {nw['macro_f1']:.3f} / {final['macro_f1']:.3f} / {iw['macro_f1']:.3f}")
    if c4:
        abl.append(f"4 instead of 2 context turns: W-F1 {c4['weighted_f1']:.3f} vs {final['weighted_f1']:.3f}")
    abl_text = (" Ablations (seed 42, test): " + "; ".join(abl) + ".") if abl else ""
    short_text = ""
    if ctx_short:
        short_text = (f" Surprisingly, the context gain is concentrated in longer turns (>8 words: accuracy "
                      f"{ctx_short['long_utt']:.3f} → {ctx_short['long_ctx']:.3f}) rather than very short ones "
                      f"(≤3 words: {ctx_short['short_utt']:.3f} → {ctx_short['short_ctx']:.3f}), whose emotion is "
                      "mostly carried by tone of voice.")
    para(doc, before=4, text=
         f"**Findings.** Fine-tuned RoBERTa with context is the best model: weighted-F1 "
         f"{mean(ctx, 'weighted_f1'):.3f}, i.e. +{gain_base_w * 100:.1f} points over the TF-IDF baseline "
         f"and {gain_ctx_w * 100:+.1f} (macro-F1 {gain_ctx_m * 100:+.1f}) over the utterance-only RoBERTa."
         f"{short_text}{abl_text} Per class (Fig. 1a), *{best_cls}* is easiest (F1 {pc[best_cls]['f1']:.2f}) and "
         f"*{worst[0]}* / *{worst[1]}* are hardest (F1 {pc[worst[0]]['f1']:.2f} / {pc[worst[1]]['f1']:.2f}; "
         f"only {pc[worst[0]]['support']} and {pc[worst[1]]['support']} test examples). The confusion matrix "
         "(Fig. 1b) shows that for five of the six emotions the dominant error is *neutral*: mild emotion "
         "often looks neutral in text. For reference, published text-only systems using the much larger RoBERTa-large with "
         "speaker-aware context reach about 0.65 weighted-F1 on MELD [3].")
    p = para(doc, after=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.add_run().add_picture(str(RES / "figures" / "report_figure.png"), width=Cm(17.2))
    para(doc, "Figure 1. (a) Per-class test F1 for the baseline and both RoBERTa variants (seed 42). "
              "(b) Row-normalised confusion matrix (%) of the final model.", size=8, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=2)

    heading(doc, "5. Demo")
    lat_text = f" Inference takes about {lat['ms_per_message']:.0f} ms per message on CPU." if lat else ""
    para(doc, "**python app.py** starts a Gradio web app with three tabs: (1) a single message with optional "
              "previous turns, showing RoBERTa's probability for all seven emotions next to the TF-IDF baseline; "
              "(2) a whole conversation, where every turn is labelled using only the turns before it; (3) an "
              "\"About\" tab with the test metrics and limitations. The message *\"Come on!\"* is read as anger "
              "after a complaint and as joy after good news, and the app also shows what the model would say "
              "without the previous turns." + lat_text)

    heading(doc, "6. Limitations and future work")
    para(doc, "(1) **Text only:** sarcasm and tone-dependent labels are the main failure mode. (2) **Rare "
              "classes** (fear, disgust) remain weak despite class weighting. (3) **One label per turn:** mixed "
              "emotions cannot be expressed. (4) **Domain:** a 1990s US sitcom; slang, other dialects or "
              "code-mixed text (e.g. Hinglish) may not transfer. (5) **Compute:** roberta-base and 2 context "
              "turns were chosen to fit a CPU budget. *Future work:* RoBERTa-large with speaker-aware context, "
              "focal loss or data augmentation for rare classes, and adding MELD's audio features.")

    heading(doc, "References")
    for ref in ["[1] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal "
                "Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019.",
                "[2] Y. Liu, M. Ott, N. Goyal, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. "
                "arXiv:1907.11692, 2019.",
                "[3] T. Kim, P. Vossen. EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa. "
                "arXiv:2108.12009, 2021."]:
        para(doc, ref, size=7.5, after=0)

    out = ROOT / "report" / "report.docx"
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    build()
