# Guide: Build Corpus

Corpus la lop bien du lieu tho thanh don vi hoc duoc.

## Input

- Raw JSONL: `data_store/raw/linear_equation/pretrain_sft.jsonl`
- Tokenizer: `data_store/tokenizers/byte_bpe_512/tokenizer.json`

Moi record JSONL nen giu field ro rang: `prompt`, `target`, va metadata neu
can. Logic rieng cua bai toan nam trong `mintdim_lab.problems`, khong nam trong
corpus pipeline va khong nam trong generic evaluator.

## Build

```powershell
python src/mintdim_lab/cli/main.py build-units recipes\corpus\linear_equation_unit96
```

Neu truyen thu muc corpus, CLI doc file chuan `unit_build.yaml` trong thu muc
do. Neu truyen file YAML ro rang, CLI dung dung file do.

Unit-build config:

```text
recipes/corpus/linear_equation_unit96/unit_build.yaml
```

Output:

```text
data_store/packed/linear_equation_unit96/
```

## Train Tokenizer

```powershell
python src/mintdim_lab/cli/main.py train-tokenizer `
  --input data_store\raw\linear_equation\pretrain_sft.jsonl `
  --output-dir data_store\tokenizers\byte_bpe_512
```

## Read

Training doc tu unit-read config:

```text
recipes/corpus/linear_equation_unit96/unit_read.yaml
```

Unit-read config:

```text
recipes/corpus/linear_equation_unit96/unit_read.yaml
```

## Boundary

- `mintdim_lab.problems`: sinh/render/grade bai toan khi xay task rieng.
- `mintdim_lab.corpus`: formatting, unit-build, unit-read, batch.
- `mintdim_lab.trainer`: dung batch da doc, khong tu render bai toan.
