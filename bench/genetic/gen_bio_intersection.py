#!/usr/bin/env python3
"""
bio_intersection: a synthetic tool-grounded set-intersection benchmark.

Inspired by the eval datasets catalogued in
https://github.com/tucca-cellag/caail/blob/main/Datasets/Benchmarks.md
(LAB-Bench / BixBench style: multi-step, tool-mediated bioinformatics reasoning),
but deliberately *synthetic and single-answer* so that ground truth is pinned by
construction (the CompBioBench trick) instead of by expert annotation.

Task family
-----------
    "Given a list of genetic sequences S_i that respect a pattern X
     {result from tool db_lookup(X)} and are confirmed or not to exhibit
     feature Y {result from tool doc_lookup(S_i, Y)}, report the genetic
     sequences that both respect the pattern X and have feature Y."

The two tool results are *inlined* into the prompt, so no live tool runtime is
needed to evaluate a model. What the benchmark actually measures is whether a
model can (a) read a structured tool dump, (b) read N free-text curator notes
whose polarity is expressed in natural language rather than yes/no, and
(c) return the intersection of the two evidence sources.

Deliberate difficulty knobs (all recorded per row):
  * doc_lookup covers a *wider* candidate set than db_lookup returns, so a model
    that merely filters the doc notes overshoots -> a real intersection is needed.
  * some db_lookup hits have no doc note at all -> "not confirmed".
  * doc notes are natural-language paragraphs, never "yes"/"no". Adversarial
    classes include hedged positives ("initially scored negative ... ultimately
    confirmed"), negatives dense with positive keywords, notes that confirm a
    *different* feature, and inconclusive/no-data notes.

Output columns (four, no JSON)
------------------------------
  prompt
      The full prompt, with both tool results already inlined.

  tool_calls
      (db_lookup[210:451],doc_lookup[452:538],doc_lookup[539:625],...)
      One entry per inlined tool result, in prompt order. [start:stop) are
      *token* indices into the prompt under --tokenizer, and the span covers the
      whole result block including its ">>> tool(...)" header line.

  correct_answer
      "S_2, S_5, S_9"  (increasing index order) or "NONE".

  explanation_per_sequence
      (S_1: in db_lookup and directly confirmed by doc_lookup; S_2: not in
       db_lookup and indirectly rejected by doc_lookup; ...)
      Every S_i visible to the model, in label order. The db half is
      "in"/"not in db_lookup"; the doc half is one of
        directly confirmed | indirectly confirmed | directly rejected |
        indirectly rejected | left undetermined | not covered
      "indirectly" marks the adversarial notes -- the ones whose surface wording
      points the opposite way from their actual verdict.
      A sequence is in `correct_answer` iff it is "in db_lookup" and
      "directly/indirectly confirmed".

Usage
-----
    python gen_bio_intersection.py                        # 10k rows, regex tokenizer
    python gen_bio_intersection.py --preview              # print one example, write nothing
    python gen_bio_intersection.py --tokenizer google/gemma-3-4b-it
    python gen_bio_intersection.py -n 500 --out /tmp/small.csv

Grading (suggested): parse the last /ANSWER:\\s*(.*)/ line of the model output,
split on commas, compare as a set against `correct_answer`.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Domain vocabulary
# --------------------------------------------------------------------------- #

IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}

# (human-readable name, IUPAC motif)
MOTIFS: list[tuple[str, str]] = [
    ("TATA box", "TATAWAWR"),
    ("CAAT box", "GGCCAATCT"),
    ("GC box (Sp1)", "GGGGCGGGG"),
    ("canonical E-box", "CANNTG"),
    ("MyoD-type E-box", "CAGCTG"),
    ("Kozak consensus", "GCCRCCATGG"),
    ("cAMP response element", "TGACGTCA"),
    ("AP-1 / TRE site", "TGASTCA"),
    ("NF-kB response element", "GGGRNNYYCC"),
    ("MEF2 binding site", "YTAWWWWTAR"),
    ("CArG box (SRF)", "CCWWWWWWGG"),
    ("GATA factor site", "WGATAR"),
    ("heat shock element", "NGAANNTTCNNGAAN"),
    ("polyadenylation signal", "AATAAA"),
    ("Shine-Dalgarno sequence", "AGGAGG"),
    ("Pribnow box", "TATAAT"),
    ("T7 promoter", "TAATACGACTCACTATAG"),
    ("Oct4/Sox2 composite site", "CATTGTNATGCAAAT"),
    ("glucocorticoid response element", "GGTACANNNTGTTCT"),
    ("insulin response element", "CAAAACAAA"),
    ("SV40 core enhancer", "GTGGWWWG"),
    ("iron response element stem", "CAGWGH"),
]

# Cellular-agriculture / cell-line flavoured phenotypes.
@dataclass(frozen=True)
class Feature:
    fid: str
    name: str        # goes in the prompt as Y
    assay: str
    phenotype: str   # a concrete measurable restatement


FEATURES: list[Feature] = [
    Feature("thermostab65", "thermostability at 65 °C",
            "differential scanning fluorimetry",
            "retention of >80% activity after 30 min at 65 °C"),
    Feature("myoblast_prom", "promoter activity in bovine myoblasts",
            "dual-luciferase reporter assay",
            ">4-fold reporter induction over the empty-vector control"),
    Feature("serumfree_growth", "serum-free proliferation support",
            "48 h EdU incorporation in B8 medium",
            "doubling time under 26 h without FBS"),
    Feature("cas9_offtarget", "measurable Cas9 off-target cleavage",
            "GUIDE-seq",
            ">0.5% indel frequency at a non-cognate locus"),
    Feature("myogenic_diff", "myogenic differentiation competence",
            "MyHC immunostaining after 5 days in differentiation medium",
            "a fusion index above 35%"),
    Feature("lipid_accum", "intramuscular lipid accumulation",
            "Oil Red O quantification",
            "an absorbance ratio above 2.1 versus undifferentiated controls"),
    Feature("scaffold_adhesion", "cell adhesion to textured soy-protein scaffolds",
            "a 2 h static adhesion assay",
            ">70% seeded-cell retention after PBS washing"),
    Feature("heme_binding", "leghemoglobin-like heme binding",
            "Soret-band absorbance spectroscopy",
            "a distinct 412 nm Soret peak"),
    Feature("umami_release", "umami peptide release on simulated digestion",
            "INFOGEST digestion followed by LC-MS/MS",
            "a free-glutamate increase above 180 mg/kg"),
    Feature("immortalization", "spontaneous immortalization potential",
            "extended serial passaging past P40",
            "no senescence-associated beta-galactosidase plateau"),
    Feature("hypoxia_tolerance", "hypoxia tolerance at 1% O2",
            "a 72 h hypoxic-chamber viability assay",
            ">85% viability relative to normoxia"),
    Feature("cho_secretion", "high-titre secretion in CHO-K1",
            "fed-batch shake-flask titre measurement",
            "a titre above 1.4 g/L on day 10"),
    Feature("bhk_transfect", "efficient lipofection in BHK-21",
            "flow cytometry of a GFP fusion",
            ">60% GFP-positive cells at 24 h"),
    Feature("codon_stability", "elevated mRNA stability from codon optimisation",
            "actinomycin-D chase followed by RT-qPCR",
            "a transcript half-life above 9 h"),
    Feature("crispr_ki", "efficient HDR knock-in",
            "ddPCR of the integration junction",
            ">12% biallelic knock-in"),
    Feature("antifungal", "antifungal activity against Candida albicans",
            "a broth microdilution MIC assay",
            "an MIC at or below 16 µg/mL"),
]

SOURCES = [
    "CellAgDB", "BovReg curation team", "the FAANG annotation group",
    "the AgBioSeq consortium", "an internal Tucca screen", "the MyoAtlas portal",
    "the FarmGTEx re-analysis", "the Cultivated Tissue Repository",
]

# --------------------------------------------------------------------------- #
# doc_lookup note templates.
#
# Every note is a natural-language curator paragraph -- never a bare yes/no.
# CONFIRMED classes  -> the sequence has feature Y.
# NOT_* classes      -> it does not (or is not established, which counts the same).
# --------------------------------------------------------------------------- #

VERDICT_CONFIRMED = {"confirmed_plain", "confirmed_hedged"}

# How each note class is described in the per-sequence explanation column.
# "directly"   -> the note states the outcome plainly.
# "indirectly" -> the outcome has to be inferred past misleading surface wording.
VERDICT_PHRASE = {
    "confirmed_plain":               "directly confirmed by doc_lookup",
    "confirmed_hedged":              "indirectly confirmed by doc_lookup",
    "not_confirmed_plain":           "directly rejected by doc_lookup",
    "not_confirmed_other_feature":   "indirectly rejected by doc_lookup",
    "not_confirmed_negated_positive": "indirectly rejected by doc_lookup",
    "not_confirmed_no_data":         "left undetermined by doc_lookup",
    None:                            "not covered by doc_lookup",
}

NOTE_TEMPLATES: dict[str, list[str]] = {
    # ---- clear positives -------------------------------------------------- #
    "confirmed_plain": [
        "Record curated by {source}. {n} independent runs of {assay} were logged for "
        "this construct and all {n} met the acceptance threshold, namely {phenotype}. "
        "The entry is therefore annotated as a validated carrier of {feature}.",

        "{source} deposited {n} replicate measurements for this accession. "
        "The construct reaches {phenotype}, comfortably clearing the panel cut-off, "
        "and the reviewers signed the record off as positive for {feature} in {year}.",

        "Phenotyping is complete for this entry: {assay} places it in the upper "
        "quartile of the {year} panel, with {phenotype}. Curators list {feature} as "
        "an established property of the sequence.",

        "This accession is one of the reference positives of the panel. Across {n} "
        "replicate runs of {assay} it consistently showed {phenotype}, and {source} cites it "
        "as an exemplar when describing {feature}.",
    ],

    # ---- positives that *look* negative on a keyword skim ------------------ #
    "confirmed_hedged": [
        "The first-pass screen flagged this construct as a likely non-carrier, but that "
        "call was withdrawn: {source} repeated the {assay} under corrected buffer "
        "conditions and observed {phenotype}. The retraction notice makes clear the "
        "sequence does in fact display {feature}.",

        "Do not be misled by the archived 'FAIL' tag on this accession, which refers to a "
        "sample-tracking error in the {year} submission. Once the mix-up was resolved, "
        "{n} clean replicate runs of {assay} gave {phenotype}, so {feature} is present.",

        "There is no evidence of any defect here. The construct was re-tested by {source} "
        "after the {year} audit and reached {phenotype}; {feature} is recorded as present, "
        "notwithstanding the earlier inconclusive note that remains in the change log.",

        "Although the abstract of the original report never uses the word 'confirmed', the "
        "supplementary table is unambiguous: {assay} yields {phenotype} for this accession "
        "in {n} of {n} runs, which is what {source} means by carrying {feature}.",

        "Marked 'pending' for two years, this entry was finally cleared. The blocking issue "
        "was a missing consent form, not the biology -- {assay} had already shown "
        "{phenotype} -- and the sequence is now listed among those exhibiting {feature}.",
    ],

    # ---- clear negatives --------------------------------------------------- #
    "not_confirmed_plain": [
        "Record curated by {source}. {n} runs of {assay} were performed and none approached "
        "{phenotype}; the construct sits near the assay floor. The entry is annotated as a "
        "non-carrier with respect to {feature}.",

        "{source} tested this accession in the {year} round and reports a negative outcome: "
        "{assay} gave values well below the {phenotype} threshold in every replicate, so "
        "{feature} is not attributed to the sequence.",

        "Phenotyping is complete and negative. The construct fails to reach {phenotype} even "
        "after the protocol was relaxed, and reviewers explicitly rejected the claim that it "
        "shows {feature}.",

        "This is one of the panel's reference negatives. Repeated {assay} measurements place "
        "it below the detection limit, and {source} uses it as the baseline control when "
        "quantifying {feature} in other constructs.",
    ],

    # ---- negatives that confirm a *different* feature ---------------------- #
    "not_confirmed_other_feature": [
        "The strongly worded confirmation in this record concerns {other_feature}, not the "
        "property under review. For {feature} the {assay} result was negative -- {phenotype} "
        "was never reached -- so only the former is established.",

        "{source} validated this accession for {other_feature} in {year} and the entry "
        "carries a green flag for that trait. The same submission notes that {feature} could "
        "not be demonstrated: {assay} stayed below {phenotype} throughout.",

        "Two traits are tracked on this record. {other_feature} is confirmed with {n} "
        "replicates; {feature} is not, the construct having missed {phenotype} in every "
        "{assay} run reported so far.",

        "Positive, but for the wrong trait: the {year} certificate attached here attests to "
        "{other_feature}. With respect to {feature} the curators recorded a clear negative "
        "after {assay}.",
    ],

    # ---- no data / inconclusive (counts as not confirmed) ------------------ #
    "not_confirmed_no_data": [
        "No phenotyping data are attached to this accession. {source} imported the sequence "
        "from a genome annotation in {year} and it has never been put through {assay}, so its "
        "status for {feature} is simply unknown.",

        "The single {assay} run on file was voided when the plate reader failed calibration. "
        "Nothing can be said about whether the construct reaches {phenotype}; {feature} "
        "remains undetermined pending a repeat.",

        "This entry is a computational prediction only. A model scores it as a plausible "
        "candidate for {feature}, but no wet-lab {assay} has been carried out and {source} "
        "keeps the trait field empty.",

        "Results here are contradictory and the record is frozen: two replicate runs of "
        "{assay} cleared {phenotype} and three did not. {source} has withheld any call on {feature} "
        "until the discrepancy is resolved.",

        "Access to the underlying dataset is restricted. The public record for this accession "
        "lists an experiment relating to {feature}, but no outcome is exposed, so the trait "
        "cannot be treated as established.",
    ],

    # ---- negatives phrased with heavy positive vocabulary ------------------ #
    "not_confirmed_negated_positive": [
        "It would be wrong to read this record as a confirmation. The words 'validated', "
        "'confirmed' and 'positive' all appear, but they qualify the taxonomic assignment and "
        "the sequencing depth. On {feature} itself the {assay} verdict was negative.",

        "A frequently miscited entry. The high-confidence, fully confirmed, expert-reviewed "
        "annotation attached here is about the reading frame. Regarding {feature}, {source} "
        "states plainly that {phenotype} was not achieved.",

        "Yes, an earlier version of this record claimed {feature}. That claim was retracted in "
        "{year} after {source} failed to reproduce it in {n} attempts, and the current release "
        "carries the property as absent.",

        "The construct is confirmed to be a close homologue of a known carrier, which is why it "
        "keeps surfacing in searches for {feature}. Homology is not phenotype: its own result "
        "from {assay} is negative for {phenotype}.",

        "Successful, well-controlled, and negative. The {year} campaign of {assay} ran exactly "
        "as designed and demonstrated that this accession does not reach {phenotype}, ruling out "
        "{feature}.",
    ],
}

DIFFICULTY_PROFILE = {
    # name:    (n_db_hits,  n_decoys, n_missing_notes, p_adversarial_phrasing)
    "easy":   ((4, 6),  (0, 1), (0, 0), 0.15),
    "medium": ((6, 10), (1, 3), (0, 1), 0.45),
    "hard":   ((9, 14), (3, 6), (1, 3), 0.75),
}
DIFFICULTY_WEIGHTS = [("easy", 0.30), ("medium", 0.40), ("hard", 0.30)]

ADVERSARIAL_POS = ["confirmed_hedged"]
PLAIN_POS = ["confirmed_plain"]
ADVERSARIAL_NEG = ["not_confirmed_other_feature", "not_confirmed_negated_positive"]
PLAIN_NEG = ["not_confirmed_plain", "not_confirmed_no_data"]


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #

def iupac_to_regex(motif: str) -> str:
    out = []
    for ch in motif:
        alt = IUPAC[ch]
        out.append(alt if len(alt) == 1 else f"[{alt}]")
    return "".join(out)


def resolve_motif(motif: str, rng: random.Random) -> str:
    return "".join(rng.choice(IUPAC[ch]) for ch in motif)


def random_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def make_matching(motif: str, length: int, rng: random.Random) -> str:
    inst = resolve_motif(motif, rng)
    pad = length - len(inst)
    left = rng.randint(0, max(pad, 0))
    return random_dna(left, rng) + inst + random_dna(max(pad - left, 0), rng)


def make_non_matching(pat: re.Pattern[str], length: int, rng: random.Random) -> str:
    for _ in range(400):
        s = random_dna(length, rng)
        if not pat.search(s):
            return s
    # Fall back to a homopolymer-ish string; every motif in MOTIFS has >=2
    # distinct fixed bases, so this can never match.
    return "A" * length


def accession(rng: random.Random) -> str:
    prefix = rng.choice(["BGX", "CAG", "MYB", "SCF", "HEM", "TCA", "BOS", "GLG"])
    return f"{prefix}-{rng.randint(10000, 99999)}"


# --------------------------------------------------------------------------- #
# Prompt builder with span tracking
# --------------------------------------------------------------------------- #

class SpanWriter:
    """Accumulates prompt text while recording character spans of marked regions."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0

    def write(self, text: str) -> None:
        self._parts.append(text)
        self._len += len(text)

    def mark(self) -> int:
        return self._len

    @property
    def text(self) -> str:
        return "".join(self._parts)


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\w+|[^\w\s]")


class RegexTokenizer:
    """Dependency-free deterministic tokenizer: words and single punctuation."""

    name = "regex"

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _WORD_RE.finditer(text)]


class HFTokenizer:
    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.name = name
        self._tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if not getattr(self._tok, "is_fast", False):
            raise SystemExit(
                f"tokenizer {name!r} has no fast implementation, so character->token "
                f"offsets are unavailable. Use --tokenizer regex."
            )

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(a, b) for a, b in enc["offset_mapping"] if b > a]


def format_tool_calls(names: list[str], spans: list[tuple[int, int]]) -> str:
    """(db_lookup[210:451],doc_lookup[452:538],...) -- token ranges, in prompt order."""
    return "(" + ",".join(f"{n}[{a}:{b}]" for n, (a, b) in zip(names, spans)) + ")"


def char_spans_to_token_spans(
    offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    out: list[tuple[int, int]] = []
    for cs, ce in spans:
        i = bisect.bisect_right(ends, cs)           # first token ending after cs
        j = bisect.bisect_left(starts, ce) - 1      # last token starting before ce
        if j < i:
            j = i
        out.append((min(i, len(offsets)), min(j + 1, len(offsets))))
    return out


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #

@dataclass
class Seq:
    label: str
    acc: str
    nt: str
    in_db: bool
    verdict: str | None = None  # None -> no doc_lookup note at all


@dataclass
class Example:
    prompt: str
    tool_names: list[str]              # one per inlined tool result, in prompt order
    char_spans: list[tuple[int, int]]  # aligned 1:1 with tool_names
    answer_labels: list[str]
    answer_seqs: list[str]
    explanation: str
    meta: dict = field(default_factory=dict)


def weighted_choice(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def pick_verdict(rng: random.Random, positive: bool, p_adv: float) -> str:
    pool = (ADVERSARIAL_POS if positive else ADVERSARIAL_NEG) if rng.random() < p_adv \
        else (PLAIN_POS if positive else PLAIN_NEG)
    return rng.choice(pool)


_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?] ))([a-z])")


def render_note(
    verdict: str, feature: Feature, other: Feature, rng: random.Random
) -> str:
    tpl = rng.choice(NOTE_TEMPLATES[verdict])
    text = tpl.format(
        feature=feature.name,
        assay=feature.assay,
        phenotype=feature.phenotype,
        other_feature=other.name,
        source=rng.choice(SOURCES),
        n=rng.randint(3, 8),
        year=rng.randint(2016, 2025),
    )
    text = " ".join(text.split())
    # Substituted source / feature names can land at a sentence start in
    # lowercase ("the MyoAtlas portal deposited ...") -> fix them up.
    return _SENTENCE_START.sub(lambda m: m.group(1).upper(), text)


def build_example(rng: random.Random) -> Example:
    difficulty = weighted_choice(rng, DIFFICULTY_WEIGHTS)
    (db_lo, db_hi), (dc_lo, dc_hi), (ms_lo, ms_hi), p_adv = DIFFICULTY_PROFILE[difficulty]

    n_db = rng.randint(db_lo, db_hi)
    n_decoy = rng.randint(dc_lo, dc_hi)
    n_missing = min(rng.randint(ms_lo, ms_hi), n_db - 1)

    motif_name, motif = rng.choice(MOTIFS)
    pat = re.compile(iupac_to_regex(motif))
    feature = rng.choice(FEATURES)
    other = rng.choice([f for f in FEATURES if f.fid != feature.fid])

    # --- build the sequence universe ------------------------------------- #
    seqs: list[Seq] = []
    for _ in range(n_db):
        seqs.append(Seq("", "", make_matching(motif, rng.randint(36, 66), rng), True))
    for _ in range(n_decoy):
        seqs.append(Seq("", "", make_non_matching(pat, rng.randint(36, 66), rng), False))
    rng.shuffle(seqs)
    for i, s in enumerate(seqs, start=1):
        s.label = f"S_{i}"
        s.acc = accession(rng)

    # --- assign verdicts --------------------------------------------------- #
    p_pos = rng.uniform(0.15, 0.85)
    forced = rng.random()
    for s in seqs:
        positive = rng.random() < p_pos
        if forced < 0.08 and s.in_db:
            positive = False              # ~8% of rows: empty answer
        elif forced > 0.95 and s.in_db:
            positive = True               # ~5% of rows: everything qualifies
        s.verdict = pick_verdict(rng, positive, p_adv)

    # some db hits get no doc note at all -> not confirmed
    db_seqs = [s for s in seqs if s.in_db]
    for s in rng.sample(db_seqs, k=min(n_missing, len(db_seqs))):
        s.verdict = None

    documented = [s for s in seqs if s.verdict is not None]
    rng.shuffle(documented)  # doc_lookup call order != db listing order

    answer = [s for s in seqs if s.in_db and s.verdict in VERDICT_CONFIRMED]

    # --- render the prompt ------------------------------------------------- #
    pattern_desc = f'contains the {motif_name} motif "{motif}" (IUPAC)'
    w = SpanWriter()
    tool_names: list[str] = []
    spans: list[tuple[int, int]] = []

    w.write(
        "You are reading the transcript of two database tools that have already been run.\n"
        "\n"
        "TASK\n"
        "Given a list of genetic sequences S_i that respect a pattern X (the result of\n"
        "db_lookup(X)) and are confirmed or not to exhibit a feature Y (the result of\n"
        "doc_lookup(S_i, Y)), report the genetic sequences that both respect the pattern X\n"
        "and have the feature Y.\n"
        "\n"
        f"X = the sequence {pattern_desc}\n"
        f"Y = {feature.name}\n"
        "\n"
        "RULES\n"
        "1. A sequence qualifies only if it appears in the db_lookup result AND its\n"
        "   doc_lookup note establishes that it has feature Y.\n"
        "2. doc_lookup was run over a wider candidate set than db_lookup returned. Notes for\n"
        "   sequences absent from the db_lookup result must be ignored.\n"
        "3. Some sequences in the db_lookup result have no doc_lookup note. They do not\n"
        "   qualify.\n"
        "4. Notes are free text. Read them for meaning, not for keywords: a note may be\n"
        "   negative while using positive vocabulary, may confirm a different feature, or\n"
        "   may leave the status undetermined. Undetermined counts as not qualifying.\n"
        "\n"
        "TOOL OUTPUTS\n"
        "\n"
    )

    # db_lookup block
    start = w.mark()
    w.write(f'>>> db_lookup(pattern="{motif}")\n')
    w.write(f"    {len(db_seqs)} records match the pattern.\n")
    w.write("    label   accession   sequence\n")
    for s in seqs:
        if s.in_db:
            w.write(f"    {s.label:<7} {s.acc:<11} {s.nt}\n")
    spans.append((start, w.mark()))
    tool_names.append("db_lookup")

    w.write("\n")

    # doc_lookup blocks
    for s in documented:
        start = w.mark()
        w.write(f'>>> doc_lookup(sequence="{s.label}", feature="{feature.name}")\n')
        w.write(f"    {s.label} / {s.acc}\n")
        w.write(textwrap.fill(render_note(s.verdict, feature, other, rng),
                              width=92, initial_indent="    ", subsequent_indent="    "))
        w.write("\n")
        spans.append((start, w.mark()))
        tool_names.append("doc_lookup")
        w.write("\n")

    w.write(
        "OUTPUT FORMAT\n"
        "Reply with exactly one line and nothing else:\n"
        "ANSWER: <labels, comma separated, in increasing index order>\n"
        "for example  ANSWER: S_2, S_5\n"
        "If no sequence qualifies, reply exactly:  ANSWER: NONE\n"
    )

    answer.sort(key=lambda s: int(s.label.split("_")[1]))

    # Per-sequence derivation, covering every S_i the model can see, in label order.
    explanation = "(" + "; ".join(
        f"{s.label}: {'in' if s.in_db else 'not in'} db_lookup and "
        f"{VERDICT_PHRASE[s.verdict]}"
        for s in sorted(seqs, key=lambda s: int(s.label.split("_")[1]))
    ) + ")"

    return Example(
        prompt=w.text,
        tool_names=tool_names,
        char_spans=spans,
        answer_labels=[s.label for s in answer],
        answer_seqs=[s.nt for s in answer],
        explanation=explanation,
        meta={
            "pattern": pattern_desc,
            "pattern_name": motif_name,
            "pattern_motif": motif,
            "pattern_regex": iupac_to_regex(motif),
            "feature": feature.name,
            "feature_id": feature.fid,
            "n_db_hits": len(db_seqs),
            "n_doc_calls": len(documented),
            "n_decoys": n_decoy,
            "n_missing_notes": len([s for s in db_seqs if s.verdict is None]),
            "n_confirmed_in_db": len(answer),
            "difficulty": difficulty,
            "doc_verdicts": {s.label: s.verdict for s in documented},
        },
    )


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

def self_check(ex: Example) -> None:
    """Assert the ground truth really is derivable from the rendered prompt."""
    pat = re.compile(ex.meta["pattern_regex"])
    for nt in ex.answer_seqs:
        assert pat.search(nt), "answer sequence does not match the pattern"
    assert len(ex.tool_names) == len(ex.char_spans)
    for (a, b), name in zip(ex.char_spans, ex.tool_names):
        block = ex.prompt[a:b]
        assert block.startswith(f">>> {name}("), f"span misaligned: {block[:40]!r}"
    # every answer label must be listed in the db_lookup block
    db_block = ex.prompt[ex.char_spans[0][0]:ex.char_spans[0][1]]
    for lab in ex.answer_labels:
        assert re.search(rf"^\s+{lab}\s", db_block, re.M), f"{lab} missing from db block"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

COLUMNS = ["prompt", "tool_calls", "correct_answer", "explanation_per_sequence"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=10_000, help="number of rows (default 10000)")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("data/bio_intersection.csv"))
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--tokenizer", default="regex",
                    help="'regex' (default, no deps) or a HuggingFace id, "
                         "e.g. google/gemma-3-4b-it")
    ap.add_argument("--preview", action="store_true", help="print one example and exit")
    ap.add_argument("--no-check", action="store_true", help="skip per-row self-checks")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    if args.preview:
        ex = build_example(rng)
        self_check(ex)
        print(ex.prompt)
        print("-" * 92)
        tok = RegexTokenizer() if args.tokenizer == "regex" else HFTokenizer(args.tokenizer)
        spans = char_spans_to_token_spans(tok.offsets(ex.prompt), ex.char_spans)
        print("tool_calls  :", format_tool_calls(ex.tool_names, spans))
        print()
        print("correct_answer:", ", ".join(ex.answer_labels) or "NONE")
        print()
        print("explanation_per_sequence:")
        print(textwrap.fill(ex.explanation, width=92,
                            initial_indent="  ", subsequent_indent="  "))
        return 0

    tok = RegexTokenizer() if args.tokenizer == "regex" else HFTokenizer(args.tokenizer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "difficulty": {},
        "verdict": {},
        "answer_len": {},
        "empty": 0,
        "tokens": 0,
        "db_hits": 0,
        "doc_calls": 0,
    }

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(COLUMNS)

        for i in range(args.num):
            ex = build_example(rng)
            if not args.no_check:
                self_check(ex)

            offs = tok.offsets(ex.prompt)
            tspans = char_spans_to_token_spans(offs, ex.char_spans)

            writer.writerow([
                ex.prompt,
                format_tool_calls(ex.tool_names, tspans),
                ", ".join(ex.answer_labels) if ex.answer_labels else "NONE",
                ex.explanation,
            ])

            d = stats["difficulty"]
            d[ex.meta["difficulty"]] = d.get(ex.meta["difficulty"], 0) + 1
            for v in ex.meta["doc_verdicts"].values():
                stats["verdict"][v] = stats["verdict"].get(v, 0) + 1
            k = len(ex.answer_labels)
            stats["answer_len"][k] = stats["answer_len"].get(k, 0) + 1
            stats["empty"] += int(k == 0)
            stats["tokens"] += len(offs)
            stats["db_hits"] += ex.meta["n_db_hits"]
            stats["doc_calls"] += ex.meta["n_doc_calls"]

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.num}", file=sys.stderr, flush=True)

    n = args.num
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {n} rows -> {args.out}  ({size_mb:.1f} MB)")
    print(f"tokenizer            : {tok.name}")
    print(f"mean prompt tokens   : {stats['tokens'] / n:.0f}")
    print(f"mean db hits / row   : {stats['db_hits'] / n:.1f}")
    print(f"mean doc calls / row : {stats['doc_calls'] / n:.1f}")
    print(f"empty-answer rows    : {stats['empty']} ({100 * stats['empty'] / n:.1f}%)")
    print("difficulty           : " + "  ".join(
        f"{k}={v}" for k, v in sorted(stats["difficulty"].items())))
    print("answer size          : " + "  ".join(
        f"{k}:{v}" for k, v in sorted(stats["answer_len"].items())))
    print("doc note classes     :")
    tot = sum(stats["verdict"].values())
    for k, v in sorted(stats["verdict"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<32} {v:>7}  ({100 * v / tot:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
