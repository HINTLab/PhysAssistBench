# PhysAssistBench

Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistance ([arXiv](https://arxiv.org/abs/2606.18613))

> 🎉 Accepted to **Findings of EMNLP 2026**.

## Overview

PhysAssistBench evaluates LLMs as physician assistants in multi-turn clinical
sessions built from real MIMIC-IV admissions: the model must resolve implicit
physician queries, call FHIR R4 EHR tools against real patient data, interview a
simulated patient, and integrate the evidence into physician-facing answers.

- **324 sessions × 4 turns**, fully bilingual (English + Chinese)
- **4 clinical scenarios**: diagnostic workup, medication safety, treatment response, discharge planning
- **4 task types**: Information Lookup / Data Gathering / Clinical Reasoning / Write-Update
- **3 implicitness types**: Nominal Anaphora / Predicate Ellipsis / Abstract Event Anaphora
- **18-tool set**: FHIR R4 EHR APIs + patient-interview tools, aligned EN/ZH schemas
- **Rubric-based LLM-judge scoring**: mean Rubric Score (mRS), Pass@Turn, Pass@Session

The benchmark is one JSONL file per clinical scenario. Because every session is
derived from MIMIC-IV, the files are **not stored in this repository**: obtain
them through PhysioNet (credentialed access, see [License and data use](#license-and-data-use))
and place them in `physassistbench/data/`:

```
physassistbench/data/diagnostic_workup.jsonl
physassistbench/data/med_safety.jsonl
physassistbench/data/treatment_response.jsonl
physassistbench/data/discharge_planning.jsonl
```

Raw MIMIC-IV inputs — used to generate sessions and to execute tool calls
against — live under `raw_data/` (see [Data](#data-mimic-iv)); they are distinct
from the `physassistbench/data/` evaluation benchmark above.

## Environment setup

The code needs **Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/). One
command installs everything:

```bash
uv sync
```

Create a `.env` at the repo root with the LLM credentials:

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
# Optional: GATEWAY_OPENAI_API_KEY=... # enables the GPT-5.4-mini rubric judge used in the paper
                                       # (via an Azure OpenAI gateway); without it the judge falls back to DeepSeek
```

### Data (MIMIC-IV)

MIMIC-IV is credentialed data. The four source modules come from three separate
PhysioNet datasets — download each (a credentialed PhysioNet account and its
data-use agreement are required for all):

| Module | Download | Path used below |
|--------|----------|-----------------|
| `hosp`, `icu` | [MIMIC-IV](https://physionet.org/content/mimiciv/) | `mimiciv/3.1/hosp`, `mimiciv/3.1/icu` |
| `ed` | [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) | `mimiciv/ed-2.2/ed` |
| `note` | [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) | `mimic-iv-note/2.2/note` |

PhysAssistBench reads patient data as one directory per subject:

- `raw_data/MIMIC-IV-split-by-patient/split_data_each_patient/<subject_id>/<module>_<table>.csv.csv.gz`
  — per-patient MIMIC-IV tables (override location with `MIMIC_PATIENT_ROOT`)
- `raw_data/mimiciv/hosp/d_labitems.csv.gz`, `raw_data/mimiciv/hosp/d_icd_diagnoses.csv.gz`,
  `raw_data/mimiciv/icu/d_items.csv.gz` — reference tables (override with `MIMIC_REF_ROOT`)

**Splitting the raw download into the per-subject layout.** After downloading the
raw MIMIC-IV CSVs from PhysioNet, use `scripts/split_mimic_by_subject.py` to
produce the `split_data_each_patient/` structure:

```bash
uv run python scripts/split_mimic_by_subject.py \
    --out_root raw_data/MIMIC-IV-split-by-patient/split_data_each_patient \
    --source hosp /path/to/mimiciv/3.1/hosp \
    --source icu  /path/to/mimiciv/3.1/icu \
    --source ed   /path/to/mimiciv/ed-2.2/ed \
    --source note /path/to/mimic-iv-note/2.2/note

# Build only a small sample (fast) by restricting subjects:
uv run python scripts/split_mimic_by_subject.py ... --subjects 10000032 10000048
```

Each source table with a `subject_id` column is split into
`<module>_<table>.csv.csv.gz` per subject; dimension tables (d_labitems,
d_icd_diagnoses, d_items) have no `subject_id` and are skipped — copy those into
`MIMIC_REF_ROOT` yourself. Splitting the full dataset is I/O-heavy (labevents and
chartevents dominate); use `--subjects` to build a working sample quickly.

## Quick smoke test

Once dependencies are installed (`uv sync`) and `.env` has `DEEPSEEK_API_KEY`,
run the end-to-end smoke test — it generates one clinical session and then
evaluates a doctor-agent LLM on it, all via DeepSeek V4 Flash:

```bash
scripts/smoke_test.sh            # generate 1 med_safety session, then evaluate it
scripts/smoke_test.sh example    # skip generation; evaluate the example session
```

`example` mode evaluates `physassistbench/examples/med_safety_example.jsonl`, a
ready-made session, so it runs without generating new data (only
`DEEPSEEK_API_KEY` is needed). The example session is MIMIC-IV-derived and is
distributed with the benchmark files via PhysioNet, not in this repository —
copy it to that path first. Results are written to
`physassistbench/results_smoke/` (`eval_report.txt`, `metrics_summary.json`).

## Code Structure

**`physassistbench/data/` is the benchmark** — everything else is the generation
pipeline that built it or the evaluation harness that scores models on it.

```
physassistbench/
├── data/                the benchmark: 324 bilingual sessions, one JSONL per scenario (from PhysioNet; not in git)
├── pipeline/            session generation: planner, user/patient agents, checkers, rubric generator
├── tools/               the 18-tool set: FHIR R4 EHR APIs + patient-interview tools (EN/ZH schemas)
├── phm/                 patient history model: persona building and the simulated-patient runtime
├── eval/                scoring: rubric LLM judge, rule checker, metrics (mRS, Pass@Turn, Pass@Session)
├── run_eval.py          evaluate a doctor-agent model on the benchmark
├── generate_all.py      generate new sessions from MIMIC-IV
├── model_configs.yaml   model registry (model id, API base, key env var, thinking mode, budgets)
└── examples/            example session for the smoke test (from PhysioNet; not in git)

scripts/
├── smoke_test.sh              one-command generate-and-evaluate check
└── split_mimic_by_subject.py  split raw MIMIC-IV into the per-subject layout

raw_data/              credentialed MIMIC-IV inputs (per-subject tables + reference tables; not in git)
```

Evaluating your own model needs only `model_configs.yaml` (add an entry) and
`run_eval.py`; see below.

## How to run

| Step | Entry point | Notes |
|---|---|---|
| Pre-filter eligible patients | `physassistbench/prefilter_patients.py` | writes `physassistbench/qualified_patients.json` (local only; not in git) |
| Generate benchmark sessions | `physassistbench/generate_all.py` | `--scenarios med_safety --n 1` → `physassistbench/data_<timestamp>/<scenario>.jsonl` |
| Evaluate a model | `physassistbench/run_eval.py` | see flags below |
| Aggregate results | `physassistbench/fill_main_table.py`, `physassistbench/results_to_excel.py` | mRS / Pass@Turn / Pass@Session table; per-turn Excel report |
| Unit tests | `uv run pytest physassistbench/tests` | |

Models are declared in `physassistbench/model_configs.yaml` (model id, API base,
API-key env var, thinking mode, token budgets). A typical evaluation run:

```bash
uv run python physassistbench/run_eval.py \
    --model deepseek-v4-flash-thinking \
    --language en \
    --skip_generate \
    --data_dir physassistbench/data
```

### Paper evaluation protocol

The harness implements the protocol of the paper (§3.4, §5.1, Appendix D):

- every model runs with reasoning ("thinking") enabled — `reasoning_effort=high`
  for the GPT-5 series — at temperature 0.2 (not sent where the provider rejects
  sampling parameters together with reasoning); the 14 models of Table 2 are the
  entries listed at the top of `model_configs.yaml`;
- all 18 tools are exposed at every turn, with at most 16 tool calls per turn;
- patient-interview tools return the session's pre-generated reply by
  deterministic replay — no patient LLM is called during evaluation;
- `prepare_to_answer` is recorded per turn (`prepare_to_answer_called`,
  `plan_complete`); omitting it counts as an incomplete plan;
- rubric items are scored by a fixed GPT-5.4-mini judge (`gpt-5.4-mini-judge`
  in `model_configs.yaml`) that sees the model's answer, the EHR ground truth
  and the rubric items;
- mRS, Pass@Turn and Pass@Session are computed over all 4 × |D| turns
  (`fill_main_table.py`).

Useful flags: `--language zh`, `--use_explicit` (explicit-paraphrase ablation),
`--judge_only` (re-judge existing outputs), `--resume`, `--output_dir`.

Results land in `physassistbench/results_<lang>_<model>_<timestamp>/`
(`turn_results_final.json`, `metrics_summary.json`).

## Results

The paper's main results — mRS, Pass@Turn, and Pass@Session across models,
languages, scenarios, and implicitness types — are in the
[paper](https://arxiv.org/abs/2606.18613), together with the
explicit-vs-implicit ablation.

## Citation

```bibtex
@inproceedings{du2026physassistbench,
  title     = {Are {LLMs} Ready to Assist Physicians? {PhysAssistBench} for Interactive Doctor-Patient-{EHR} Assistance},
  author    = {Du, Tianming and Yu, Peijie and Shang, Sihan and Shi, Danli and Nguyen, My Linh and Gao, Shengbo and Li, Guangyuan and Yu, Yinghong and Jiang, Yan and Zhao, Qianlong and Bozorgtabar, Behzad and Ji, Shaoxiong and Pan, Jiazhen and Rueckert, Daniel and Yang, Jiancheng},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License and data use

The code is released under Apache 2.0. Benchmark data derived from MIMIC-IV is
distributed via PhysioNet under its credentialed data-use agreement and must not
be redistributed here.
