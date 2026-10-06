"""Official split parity and policy-visible data boundaries."""
from task import Question, official_split
from prep_data import pilot_questions, verl_rows


def official_rows():
    return [
        {"uid": f"{difficulty}-{i}", "question": f"Question {i}?", "answer": "543 million",
         "difficulty": difficulty, "source_files": "treasury_bulletin_1940_06.txt"}
        for difficulty, count in [("easy", 113), ("hard", 133)]
        for i in range(count)
    ]


def test_official_split_is_stable_and_hard_holdout_never_enters_pilot():
    rows = official_rows()
    train, heldout = official_split(rows, hard_train=90)
    again = official_split(list(reversed(rows)), hard_train=90)
    assert (train, heldout) == again
    assert len(train) == 203 and len(heldout) == 43
    assert all(q.uid.startswith("hard-") for q in heldout)
    assert not set(q.uid for q in train) & set(q.uid for q in heldout)
    difficulties = {row["uid"]: row["difficulty"] for row in rows}
    pilot = pilot_questions(train, difficulties)
    assert len(pilot) == 16
    assert sum(q.uid.startswith("easy-") for q in pilot) == 8
    assert all(q in train for q in pilot)


def test_gold_and_reference_filename_are_excluded_from_policy_prompt():
    question = Question(uid="q", question="What was the total?", answer="987654321 million",
                        source_files=("secret_reference_1940.txt",))
    row = verl_rows([question], max_turns=12, difficulties={"q": "easy"})[0]
    text = "\n".join(message["content"] for message in row["prompt"])
    assert "987654321" not in text and "secret_reference_1940" not in text
    assert "up to 12 steps" in text
    assert row["reward_model"]["ground_truth"] == question.answer
    assert "submit_report" in row["extra_info"]["tool_selection"]
