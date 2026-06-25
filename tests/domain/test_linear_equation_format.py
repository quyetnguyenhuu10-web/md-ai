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


def test_pretrain_format_uses_id_prompt_and_transfer_steps_without_words():
    eq = parse_equation("2x - 2 = 3")

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


def test_sft_format_explains_transfer_sign_and_keeps_fraction():
    eq = parse_equation("2x - 2 = 3")

    assert sft_prompt(eq) == "Giải phương trình: 2x-2=3?"
    assert sft_target(eq) == "\n".join(
        [
            "Chuyển -2 sang vế phải và đổi dấu, ta được 2x=3+2.",
            "Tính vế phải, ta được 2x=5.",
            "Chia hai vế cho 2, ta được x=5/2.",
            "Vậy nghiệm là x=5/2.",
        ]
    )


def test_formats_reduce_divisible_or_reducible_fractions():
    plus = parse_equation("3x + 4 = 10")
    direct = parse_equation("4x = 28")

    assert reduced_solution(plus) == "2"
    assert "step: x=6/3\nstep: x=2\nans: x=2" in pretrain_target(plus)
    assert "Rút gọn phân số, ta được x=2." in sft_target(plus)

    assert reduced_solution(direct) == "7"
    assert pretrain_target(direct) == "\n".join(
        [
            "eq: 4x=28",
            "step: x=28/4",
            "step: x=7",
            "ans: x=7",
        ]
    )


def test_grade_answer_uses_reduced_solution():
    eq = parse_equation("3x+4=10")

    assert grade_answer(eq, "Vậy nghiệm là x=2.")
    assert not grade_answer(eq, "Vậy nghiệm là x=6/3.")


def test_convert_jsonl_builds_sft_from_pretrain_subset(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "converted.jsonl"
    records = [
        {"target": "eq: 2x-2=3"},
        {"target": "eq: 3x+4=10"},
        {"target": "eq: 4x=28"},
        {"target": "eq: 9x+9=99"},
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    total, pretrain, sft = convert_jsonl(
        input_path=source,
        output_path=output,
        pretrain_count=3,
        sft_count=2,
    )

    converted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert (total, pretrain, sft) == (5, 3, 2)
    assert [record["prompt"] for record in converted[:3]] == [
        "<eq_000001>\n",
        "<eq_000002>\n",
        "<eq_000003>\n",
    ]
    assert [record["prompt"] for record in converted[3:]] == [
        "Giải phương trình: 2x-2=3?",
        "Giải phương trình: 3x+4=10?",
    ]


def test_convert_jsonl_dedupes_pretrain_equations_and_fills_to_count(tmp_path):
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
