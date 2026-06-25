from __future__ import annotations

import json

from mintdim_lab.problems.linear_equation import (
    convert_jsonl,
    equation_from_record,
    grade_answer,
    parse_equation,
    pretrain_prompt,
    pretrain_target,
    reduced_solution,
    sft_prompt,
    sft_target,
)


def test_linear_equation_pretrain_and_sft_rendering_contract():
    eq = parse_equation("2x - 2 = 3")

    assert eq.compact == "2x-2=3"
    assert reduced_solution(eq) == "5/2"
    assert pretrain_prompt(sample_id=1) == "<eq_000001>\n"
    assert pretrain_target(eq) == "\n".join(
        [
            "eq: 2x-2=3",
            "step: 2x=3+2",
            "step: 2x=5",
            "step: x=5/2",
            "ans: x=5/2",
        ]
    )
    assert sft_prompt(eq) == "Giải phương trình: 2x-2=3?"
    assert sft_target(eq) == "\n".join(
        [
            "Chuyển -2 sang vế phải và đổi dấu, ta được 2x=3+2.",
            "Tính vế phải, ta được 2x=5.",
            "Chia hai vế cho 2, ta được x=5/2.",
            "Vậy nghiệm là x=5/2.",
        ]
    )
    assert grade_answer(eq, "Vậy nghiệm là x=5/2.")


def test_linear_equation_convert_jsonl_uses_pretrain_subset_for_sft(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "converted.jsonl"
    records = [
        {"target": "eq: 2x-2=3"},
        {"target": "eq: 2x-2=3"},
        {"target": "eq: 3x+4=10"},
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    total, pretrain, sft = convert_jsonl(
        input_path=source,
        output_path=output,
        pretrain_count=4,
        sft_count=2,
    )

    converted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    pretrain_equations = [equation_from_record(record).compact for record in converted[:4]]
    sft_equations = [equation_from_record(record).compact for record in converted[4:]]

    assert (total, pretrain, sft) == (6, 4, 2)
    assert len(set(pretrain_equations)) == 4
    assert sft_equations == pretrain_equations[:2]
