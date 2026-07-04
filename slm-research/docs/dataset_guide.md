# Dataset Guide

## The mixture

Training uses a proportionally-weighted interleave of 8 Hugging Face datasets, configured in
[`configs/data/mixture.yaml`](../configs/data/mixture.yaml):

| Source | HF path | Sample count | Streaming | Val split |
|---|---|---|---|---|
| `wikitext` | `Salesforce/wikitext` (`wikitext-103-raw-v1`) | all | no | own (`validation`) |
| `openwebtext` | `Skylion007/openwebtext` | 100,000 | yes | carved (2%) |
| `bookcorpusopen` | `lucadiliello/bookcorpusopen` | 80,000 | yes | carved (2%) |
| `tinystories` | `roneneldan/TinyStories` | 80,000 | no | own (`validation`) |
| `ag_news` | `fancyzhx/ag_news` | all | no | own (`test`) |
| `cnn_dailymail` | `abisee/cnn_dailymail` (`3.0.0`) | 30,000 | no | own (`validation`) |
| `xsum` | `EdinburghNLP/xsum` | 30,000 | no | own (`validation`) |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | 100,000 | yes | carved (2%) |

"Own" val split means the dataset ships a pre-existing validation/test split and that's used
directly; "carved" means `val_split_fraction` (`configs/evaluation/default.yaml`, default `0.02`)
is sliced off the training data instead (see `data/loaders.py::load_source_splits`).

`weight: null` for every source means weights are derived proportionally from `sample_count`
rather than set explicitly — see `data/mixture.py`.

> **Note:** the mixture was originally 10 sources; two were dropped on 2026-07-04 because they no
> longer load with current `datasets` versions, discovered when they failed live during a training
> run — not a config/code bug on this repo's side, both are broken upstream:
> - `daily_dialog` (`li2017dailydialog/daily_dialog`) only ships a legacy loading script.
>   `datasets` >= 4.x refuses to execute loading scripts at all now
>   (`RuntimeError: Dataset scripts are no longer supported`), and HF's datasets-server reports
>   `is-valid=false` for it — no Parquet auto-conversion exists to fall back to.
> - `eli5` (`eli5`) was removed from the Hub entirely — `huggingface.co/datasets/eli5` now
>   resolves to an unrelated user's profile page, not the dataset (likely related to Reddit's API
>   policy changes, which the underlying data depended on).
>
> `configs/data/{daily_dialog,eli5}.yaml`, `data/adapters/{daily_dialog,eli5}.py`, and their
> `registry.py` entries are left in place (harmless, unused) rather than deleted, in case a working
> replacement path appears later. `MixtureConfig.sources` now validates for exactly 8 entries
> (`utils/config_schema.py`), and `_VALID_DATASET_NAMES` no longer includes either name.
>
> Separately: `fineweb_edu` itself was substituted for the original architecture spec's
> `yelp_review_full` (`docs/architecture.md` Section 1) during implementation, as a larger,
> higher-quality web-text source. `configs/data/yelp_review_full.yaml` is dead config from that
> earlier swap — nothing reads it, no adapter or registry entry exists for it.

## Where dataset identity actually lives

`configs/data/{name}.yaml` (one file per source, e.g. `ag_news.yaml`, `wikitext.yaml`) is a
holdover from the original architecture sketch and is **not read by any code path**. The real
source of truth is `src/slm_research/data/registry.py`: a `REGISTRY` dict mapping each short name
(as used in `mixture.yaml`) to its HF repo path, split names, and adapter class. Those
`configs/data/*.yaml` files still show `hf_path: null` / `text_field: null` "TODO(phase 4)"
placeholders — that's expected; they're vestigial, not a partially-finished config.

Adding a new dataset to the mixture means:
1. Add an adapter in `src/slm_research/data/adapters/` (subclass of `DatasetAdapter`, converts
   raw rows to `{"text": str}` — see any existing adapter, e.g. `adapters/ag_news.py`).
2. Register it in `registry.py`'s `REGISTRY` dict.
3. Add a `sources:` entry to `configs/data/mixture.yaml`.

## Pipeline (per source, then across sources)

```
load_source            data/loaders.py         load_dataset (streaming or full) + adapter → {"text": str}
  → preprocess_dataset  data/preprocessing.py   cleaning, dedup, EOS insertion
  → tokenize_dataset    data/tokenization.py    Qwen3 tokenizer, no truncation/padding yet
  → pack                data/packing.py         constant-length packing to sequence_length (2048)
                                                 — WITHIN this dataset, before mixing
→ build_mixture          data/mixture.py        weighted interleave ACROSS all packed sources
→ CausalLMCollator       data/collators.py       attention masks + labels per batch
→ DataLoader
```

`data/datamodule.py::DataModule` (via `build_data_module`) wires this whole pipeline end to end
and is the only entry point `scripts/train.py` and `scripts/evaluate.py` call — neither script
touches `loaders.py`/`packing.py`/etc. directly.

**Why pack before mixing, not after:** packing within each dataset before interleaving means
`val/ppl_by_length_bucket` (see the [Training Guide](training_guide.md#evaluation)) reflects each
source's genuine sequence-length/content distribution, rather than an averaged artifact of the
mixture.

## Streaming vs. full load

`openwebtext`, `bookcorpusopen`, and `fineweb_edu` are large enough to require streaming
(`IterableDataset` — shuffle-buffered rather than fully shuffled, `take`/`skip` instead of
`train_test_split` for carving a val split). Every other source loads in full and shuffles
normally. This is set per-source in `mixture.yaml`'s `streaming:` field.

Mixing the two types across sources needs two things `interleave_datasets` doesn't handle for
free, both fixed in `data/packing.py` and `data/mixture.py`:

- **Type alignment.** `datasets.interleave_datasets` requires every input to be the same type
  (all `Dataset` or all `IterableDataset`) and raises otherwise — but the real mixture legitimately
  combines both. `build_mixture` now converts any `Dataset` to `IterableDataset` (via
  `.to_iterable_dataset()`) whenever the list is mixed, and leaves an all-one-type list untouched.
- **Schema alignment.** Arrow infers the narrowest dtype that fits the *actual values present*
  (e.g. `int32`/`int8` for a small-vocab test fixture) unless a schema is given explicitly, so the
  non-streaming packer (`.map()`-based) and the streaming packer
  (`IterableDataset.from_generator`-based) could silently disagree on `input_ids`/`attention_mask`
  dtype and still fail the same alignment check. Both packers now pass an explicit
  `int64`/`int64` `Features` schema so they always agree, regardless of the token values involved.

See `tests/test_data/test_mixture.py` and `test_packing.py` for the regression coverage.

## Standalone data scripts

`scripts/download_data.py` warms the local HF cache for every configured source (materializes
full sources; for streaming sources, iterates through the mixture-configured `sample_count` bound
once, which is the same slice training reads). `scripts/preprocess.py` runs
load → preprocess → tokenize → pack per source (reusing the exact same `data/*` functions
`DataModule` uses) and writes each source's packed train/val `Dataset` to disk via
`save_to_disk`, under `<run.output_dir>/preprocessed/<source_name>/{train,val}/`. Streaming
results are materialized (bounded by `sample_count`) since `save_to_disk` isn't defined for a
lazy `IterableDataset`.

Neither script is required for `train`/`evaluate`/`benchmark` to work — `DataModule` still builds
the pipeline on demand, in-process, every run, and doesn't read `preprocess.py`'s cache. Run
`preprocess.py` to inspect pipeline output directly (token counts, packed sequence content) or to
pre-warm a cache on disk before a run; wiring `DataModule` to optionally load from that cache as a
fast path is a reasonable next step but isn't done.
