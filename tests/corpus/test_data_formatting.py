from __future__ import annotations

from mintdim_lab.corpus.formatting import prompt_already_has_answer_prefix, prompt_answer_prefix


def test_prompt_answer_prefix_matches_unit_build_template():
    assert prompt_answer_prefix("2+5+1=") == "#Prompt_user: 2+5+1=\n#Answer:\n"


def test_prompt_answer_prefix_keeps_preformatted_prompt():
    prompt = "#Prompt_user: 2+5+1=\n#Answer:\n"

    assert prompt_already_has_answer_prefix(prompt)
    assert prompt_answer_prefix(prompt) == prompt
