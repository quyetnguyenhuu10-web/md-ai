# Guide: Problems

`mintdim_lab.problems` la khu vuc task/domain rieng.

No khong phai la dependency cua evaluator va khong phai plugin ngam cho
`python src/mintdim_lab/cli/main.py eval`.

## Khi khong can `problems`

Khong dung `problems` neu benchmark co the mo ta bang:

- file JSONL,
- cac input field,
- mot target field,
- prompt template,
- scorer generic nhu exact hoac normalized exact.

Truong hop nay dung thang `recipes/evaluation/*.yaml` va evaluator la du.

## Khi nen dung `problems`

Dung `problems` khi ban muon xay mot task rieng, vi task can logic domain ma
eval generic khong nen biet:

- sinh du lieu,
- convert data tho sang JSONL training/eval,
- render nhieu format rieng cua task,
- parse output rieng cua domain,
- grade bang luat rieng,
- tao protocol/runner rieng khi khong muon dung eval san co.

Vi du `mintdim_lab.problems.linear_equation` co the biet phuong trinh la gi,
cach rut gon phan so, cach sinh target pretrain/SFT. Generic evaluator khong
duoc biet cac chi tiet do.

## Boundary

Huong phu thuoc dung:

```text
problems -> data_store JSONL / corpus input
evaluator -> data_store JSONL + recipe evaluation
```

Huong phu thuoc sai:

```text
evaluator -> problems
problems -> evaluator
```

Neu can danh gia mot task co rule dac biet, co hai cach:

1. Chuyen task ve JSONL + template + scorer generic roi dung evaluator.
2. Viet runner rieng trong `problems` hoac mot boundary moi, khong nhung logic
   domain vao evaluator.
