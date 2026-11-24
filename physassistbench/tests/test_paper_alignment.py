"""
Unit tests for the behaviours the paper specifies (no network, no MIMIC data).

Each test class names the paper section it pins down:
  - §5.1      16 tool calls per turn; fixed GPT-5.4-mini judge; EHR ground truth for the judge
  - §3.4      Pass@Turn / Pass@Session denominators
  - App. D.3  deterministic replay of pre-generated patient replies
  - App. D.4  prepare_to_answer recorded; omission = incomplete plan
  - App. H    Gate 1 rules (Table H2), Queryable Items block (H.2), tier text (Table H1),
              explicit-question rebuild (H.4), round-trip translation (A.4)

Run from the repo root:
    uv run pytest physassistbench/tests/test_paper_alignment.py -v
"""

import json
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from physassistbench import eval_runner
from physassistbench.eval_runner import (
    MAX_TOOL_CALLS,
    AnthropicChatAdapter,
    compute_tool_metrics,
    prepare_patient_session,
    run_agent_turn,
    _thinking_extra_body,
)
from physassistbench.phm.patient_agent_runtime import get_session
from physassistbench.tools.tool_registry import call_tool


def _act(name, obs=None, args=None, **extra):
    return {"action": {"name": name, "arguments": args or {}}, "observation": obs, **extra}


def _patient_obs(text, zh=None):
    obs = {"patient_response": text,
           "patient_responses_all_literacy": {"high": text, "medium": text + " (m)", "low": text + " (l)"}}
    if zh:
        obs["patient_response_zh"] = zh
        obs["patient_responses_all_literacy_zh"] = {"high": zh, "medium": zh, "low": zh}
    return obs


def _entry(eid, reply, session_id="shared_sid", tool="patient.get_symptom_history", zh=None):
    e = {"id": eid, "subject_id": 1, "persona_config": None,
         "answer_list": [[_act("Observation.search", {"resourceType": "Bundle", "total": 0})],
                         [_act(tool, _patient_obs(reply, zh)), _act("prepare_to_answer", "gold answer")]]}
    if session_id is not None:
        e["session_id"] = session_id
    return e


# ═════════════════════════════════════════════════════════════════════════════
# App. D.3 — deterministic replay of stored patient replies
# ═════════════════════════════════════════════════════════════════════════════

class TestPatientReplay:
    def test_entries_sharing_a_session_id_replay_their_own_reply(self):
        a, b = _entry("a", "reply of A"), _entry("b", "reply of B")
        for entry, expected in ((a, "reply of A"), (b, "reply of B"), (a, "reply of A")):
            prepare_patient_session(entry, entry["session_id"])
            obs = call_tool("patient.get_symptom_history",
                            {"subject_id": 1, "session_id": entry["session_id"], "query": "x"})
            assert obs == {"patient_response": expected}

    def test_no_llm_call_when_tool_has_no_stored_reply(self, monkeypatch):
        import physassistbench.pipeline.agents.llm_client as llm_client

        def _boom(*a, **k):
            raise AssertionError("patient LLM must not be invoked during evaluation")
        monkeypatch.setattr(llm_client, "llm_call", _boom)

        prepare_patient_session(_entry("a", "reply of A"), "sid")
        obs = call_tool("patient.get_medication_adherence",
                        {"subject_id": 1, "session_id": "sid", "drug": "warfarin"})
        assert obs == {"patient_response": get_session("sid")._NO_STORED_RESPONSE["en"]}
        # stored tool still replays, and no literacy variants are generated online
        obs = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        assert obs == {"patient_response": "reply of A"}

    def test_replay_is_repeatable(self):
        prepare_patient_session(_entry("a", "reply of A"), "sid")
        r1 = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        r2 = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        assert r1 == r2 == {"patient_response": "reply of A"}

    def test_literacy_variant_selected(self):
        prepare_patient_session(_entry("a", "reply"), "sid", health_literacy="low")
        obs = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        assert obs["patient_response"] == "reply (l)"

    def test_chinese_session_uses_chinese_reply_when_stored(self):
        prepare_patient_session(_entry("a", "reply", zh="中文回复"), "sid", language="zh")
        obs = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        assert obs["patient_response"] == "中文回复"

    def test_chinese_session_falls_back_to_english_reply(self):
        prepare_patient_session(_entry("a", "reply"), "sid", language="zh")
        obs = call_tool("patient.get_symptom_history", {"subject_id": 1, "session_id": "sid"})
        assert obs["patient_response"] == "reply"


# ═════════════════════════════════════════════════════════════════════════════
# §5.1 — at most 16 tool calls per turn
# ═════════════════════════════════════════════════════════════════════════════

class _FakeClient:
    """Issues `n_first` parallel tool calls, then plain text."""

    def __init__(self, n_first, tool="NoSuch.tool"):
        self.n_first, self.tool, self.calls = n_first, tool, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            tcs = [SimpleNamespace(id=f"c{i}", type="function", model_extra={},
                                   function=SimpleNamespace(name=self.tool, arguments="{}"))
                   for i in range(self.n_first)]
            msg = SimpleNamespace(content=None, tool_calls=tcs, model_extra={})
        else:
            msg = SimpleNamespace(content="final answer", tool_calls=None, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class TestToolCallBudget:
    def test_budget_value(self):
        assert MAX_TOOL_CALLS == 16

    def test_calls_beyond_budget_are_not_executed(self):
        client = _FakeClient(n_first=20)
        executed, answer, messages, _ = run_agent_turn(
            client, [{"role": "user", "content": "q"}], tools=[],
            subject_id=1, session_id="s", task_type="Data Gathering")
        real = [a for a in executed if not a.get("synthetic")]
        assert len(real) == MAX_TOOL_CALLS
        # every tool_call still got a tool message (API requirement)
        assert sum(1 for m in messages if m.get("role") == "tool") == 20
        assert answer == "final answer"
        # the model was not given another tool-calling step after the budget was spent
        assert client.calls[-1].get("tool_choice") == "none"

    def test_under_budget_runs_normally(self):
        client = _FakeClient(n_first=3)
        executed, answer, _, _ = run_agent_turn(
            client, [{"role": "user", "content": "q"}], tools=[],
            subject_id=1, session_id="s", task_type="Data Gathering")
        assert len([a for a in executed if not a.get("synthetic")]) == 3
        assert answer == "final answer"


# ═════════════════════════════════════════════════════════════════════════════
# App. D.4 — prepare_to_answer is recorded; omission = incomplete plan
# ═════════════════════════════════════════════════════════════════════════════

class TestPrepareToAnswer:
    GOLD = [_act("Observation.search"), _act("prepare_to_answer")]

    def test_called(self):
        m = compute_tool_metrics([_act("Observation.search"), _act("prepare_to_answer")],
                                 self.GOLD, "Information Lookup")
        assert m["prepare_to_answer_called"] is True and m["plan_complete"] is True

    def test_harness_inserted_call_does_not_count(self):
        m = compute_tool_metrics(
            [_act("Observation.search"), _act("prepare_to_answer", synthetic=True)],
            self.GOLD, "Information Lookup")
        assert m["prepare_to_answer_called"] is False and m["plan_complete"] is False

    def test_missing_gold_tool_is_incomplete(self):
        m = compute_tool_metrics([_act("prepare_to_answer")], self.GOLD, "Information Lookup")
        assert m["prepare_to_answer_called"] is True and m["plan_complete"] is False

    def test_runner_marks_auto_appended_call_synthetic(self):
        client = _FakeClient(n_first=1)
        executed, _, _, _ = run_agent_turn(
            client, [{"role": "user", "content": "q"}], tools=[],
            subject_id=1, session_id="s", task_type="Information Lookup")
        assert executed[-1]["action"]["name"] == "prepare_to_answer"
        assert executed[-1].get("synthetic") is True


# ═════════════════════════════════════════════════════════════════════════════
# §5.1 — reasoning enabled for every evaluated model; fixed GPT-5.4-mini judge
# ═════════════════════════════════════════════════════════════════════════════

PAPER_MODELS = [
    "claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-ve", "doubao-seed-1-8", "gpt-5.4-mini",
    "glm-5", "kimi-k2.6", "deepseek-v4-pro-thinking", "deepseek-v4-flash-thinking",
    "minimax-m2.7-ve", "qwen3.5-35b-a3b", "qwen3.5-27b", "qwen3.5-9b", "qwen3.5-4b",
]


class TestModelSettings:
    @pytest.mark.parametrize("name", PAPER_MODELS)
    def test_paper_models_run_with_reasoning(self, name):
        cfg = eval_runner.load_model_config(name)
        if cfg.get("api_type") == "gateway_responses":
            assert cfg.get("reasoning_effort") == "high"     # GPT-5 series
        else:
            assert cfg.get("thinking") is True

    def test_judge_is_gpt_5_4_mini(self):
        from physassistbench.run_eval import JUDGE_MODEL_NAME
        assert "gpt-5.4-mini" in eval_runner.load_model_config(JUDGE_MODEL_NAME)["model_id"]

    def test_thinking_styles(self):
        assert _thinking_extra_body("x", True, "vllm") == {"chat_template_kwargs": {"enable_thinking": True}}
        assert _thinking_extra_body("glm-5", True, "none") == {}
        assert _thinking_extra_body("glm-5", True) == {"thinking": {"type": "enabled"}}

    def test_anthropic_adapter_replays_thinking_blocks_verbatim(self):
        adapter = AnthropicChatAdapter.__new__(AnthropicChatAdapter)   # no client needed
        blocks = [{"type": "thinking", "thinking": "", "signature": "sig"},
                  {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]
        _, out = adapter._convert_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "_anthropic_content": blocks,
             "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "r"},
        ])
        assert out[1] == {"role": "assistant", "content": blocks}
        assert out[2]["content"][0]["type"] == "tool_result"


# ═════════════════════════════════════════════════════════════════════════════
# §5.1 — the judge receives the EHR ground truth
# ═════════════════════════════════════════════════════════════════════════════

class TestJudgeInput:
    def test_gold_tool_results_are_shown(self):
        from physassistbench.eval.rubric_eval import _format_tool_results
        bundle = {"resourceType": "Bundle", "total": 1, "entry": [{"resource": {
            "resourceType": "Observation", "code": {"text": "Creatinine"},
            "valueQuantity": {"value": 1.2, "unit": "mg/dL"}, "effectiveDateTime": "2150-01-01"}}]}
        text = _format_tool_results([
            _act("Observation.search", bundle, {"subject_id": 1, "code": "Creatinine"}),
            _act("patient.get_symptom_history", _patient_obs("I feel dizzy"), {"session_id": "s"}),
            _act("prepare_to_answer", "reference answer text"),
        ])
        assert "Creatinine: 1.2 mg/dL" in text and "I feel dizzy" in text
        assert "code=Creatinine" in text and "subject_id" not in text
        assert "reference answer text" not in text

    def test_falls_back_to_reference_summary(self):
        from physassistbench.eval.rubric_eval import _format_tool_results
        assert _format_tool_results([_act("prepare_to_answer", "summary")]) == "summary"


# ═════════════════════════════════════════════════════════════════════════════
# §3.4 — metric denominators
# ═════════════════════════════════════════════════════════════════════════════

class TestMainTable:
    def test_unscored_turn_counts_as_failed(self, tmp_path):
        from physassistbench.fill_main_table import compute
        turns = [{"test_entry_id": "e1", "difficulty_level": 1, "rubric_score": s}
                 for s in (1.0, 1.0, 1.0, None)]
        turns += [{"test_entry_id": "e2", "difficulty_level": 1, "rubric_score": 1.0}] * 4
        (tmp_path / "turn_results_final.json").write_text(json.dumps(turns))
        row = compute(str(tmp_path))["All"]
        assert row["N"] == 2
        assert row["@T0.6"] == pytest.approx(100 * 7 / 8)
        assert row["@S0.6"] == pytest.approx(50.0)
        assert row["Avg"] == pytest.approx(100 * (0.75 + 1.0) / 2)


# ═════════════════════════════════════════════════════════════════════════════
# App. H — generation pipeline rules
# ═════════════════════════════════════════════════════════════════════════════

def _plan(*names):
    return {"Action_List": [{"name": n, "arguments": {"subject_id": 1}} for n in names]}


class TestGateOne:
    WRITE = ["MedicationRequest.create", "ServiceRequest.create", "Flag.create",
             "MedicationRequest.search", "prepare_to_answer"]

    def test_write_turn_needs_exactly_one_write_call(self):
        from physassistbench.pipeline.generate_v2 import _validate_action_plan_v2 as v
        assert v(_plan("MedicationRequest.create", "prepare_to_answer"), self.WRITE)[0]
        assert not v(_plan("MedicationRequest.create", "Flag.create", "prepare_to_answer"), self.WRITE)[0]
        assert not v(_plan("MedicationRequest.search", "MedicationRequest.create", "prepare_to_answer"), self.WRITE)[0]
        assert not v(_plan("prepare_to_answer"), self.WRITE)[0]

    def test_discharge_generator_uses_the_same_rule(self):
        from physassistbench.pipeline.generate_discharge_planning import _validate_action_plan as v
        assert v(_plan("Flag.create", "prepare_to_answer"), self.WRITE)[0]
        assert not v(_plan("Flag.create", "Flag.create", "prepare_to_answer"), self.WRITE)[0]

    def test_patient_source_requires_only_patient_tools(self):
        from physassistbench.pipeline.agents.checker_planner import validate_plan
        tools = ["patient.get_symptom_history", "Observation.search", "prepare_to_answer"]
        ok, reason = validate_plan(
            "q", "Data Gathering",
            _plan("patient.get_symptom_history", "Observation.search", "prepare_to_answer"),
            tools, subject_id=1, tool_source="patient")
        assert not ok and "all tools" in reason


class TestGenerationHelpers:
    def test_queryable_items_block(self):
        from physassistbench.pipeline.ehr_prefetch import format_queryable_items
        block = format_queryable_items(
            {"Creatinine": 1, "Potassium": 4},
            {"Condition (diagnoses)": 3, "DiagnosticReport (radiology reports)": 1, "Empty": 0})
        lines = block.splitlines()
        assert lines[0] == "[QUERYABLE ITEMS]" and lines[-1] == "[END QUERYABLE ITEMS]"
        single = lines[lines.index(next(l for l in lines if "1 result (Information Lookup turns only):" in l)) + 1]
        multi = lines[lines.index(next(l for l in lines if "≥2 results" in l and l.endswith(":"))) + 1]
        assert "Creatinine" in single and "Potassium" not in single
        assert "Potassium (4)" in multi
        assert "Empty" not in block
        assert any("DiagnosticReport" in l and "Information Lookup turns only" in l for l in lines)

    def test_tier_text_always_contains_table_h1(self):
        from physassistbench.generate_all import DIFFICULTY_CONSTRAINTS, get_difficulty_constraints
        for scenario in ("diagnostic_workup", "med_safety", "treatment_response", "discharge_planning"):
            for tier in (1, 2, 3):
                merged = get_difficulty_constraints(scenario, tier)
                assert DIFFICULTY_CONSTRAINTS[tier]["constraint_text_en"] in merged["constraint_text_en"]
        assert "MANDATORY" in DIFFICULTY_CONSTRAINTS[3]["constraint_text_en"]

    def test_content_overlap_is_punctuation_insensitive(self):
        from physassistbench.pipeline.generate_v2 import _no_content_overlap
        assert not _no_content_overlap("INR?", "What is the latest INR")
        assert _no_content_overlap("Creatinine?", "What is the latest INR")
        assert not _no_content_overlap("it?", "What is the latest INR")   # no content words → no verdict

    def test_all_18_tools_in_every_fhir_session(self):
        from physassistbench.pipeline.encoder import encode_entry
        turn = {"user_question": "q", "task_type": "Information Lookup", "subtype": None,
                "executed_actions": [], "assistant_answer": "a"}
        entry = encode_entry("e", "LabInterp", 1, None, {"env_info": ""}, [turn] * 4, tool_set="fhir")
        names = {t["function"]["name"] for t in entry["tools"]}
        assert len(names) == 18
        assert {"MedicationRequest.create", "ServiceRequest.create", "Flag.create"} <= names


class TestRoundTripTranslation:
    def _patch(self, monkeypatch, verdicts):
        from physassistbench.pipeline.agents import translator
        calls = {"zh": 0, "verify": 0}

        def fake_llm(messages, **kw):
            system = messages[0]["content"]
            if system is translator._TRANSLATE_TO_ZH_PROMPT:
                calls["zh"] += 1
                return f"译文{calls['zh']}"
            if system is translator._TRANSLATE_TO_EN_PROMPT:
                return "back translation"
            calls["verify"] += 1
            return json.dumps({"equivalent": verdicts[calls["verify"] - 1], "issue": "drift"})
        monkeypatch.setattr(translator, "llm_call", fake_llm)
        return translator, calls

    def test_verified_translation_is_returned(self, monkeypatch):
        translator, calls = self._patch(monkeypatch, [True])
        assert translator.translate_query_to_zh_verified("Creatinine?") == "译文1"
        assert calls == {"zh": 1, "verify": 1}

    def test_rejected_translation_is_retried(self, monkeypatch):
        translator, calls = self._patch(monkeypatch, [False, True])
        assert translator.translate_query_to_zh_verified("Creatinine?") == "译文2"
        assert calls == {"zh": 2, "verify": 2}

    def test_patient_replies_get_chinese_versions(self, monkeypatch):
        from physassistbench.pipeline.agents import translator
        monkeypatch.setattr(translator, "translate_to_zh", lambda t: f"ZH[{t}]")
        acts = [_act("patient.get_symptom_history", _patient_obs("hello")), _act("Observation.search", {})]
        translator.add_zh_patient_responses(acts)
        obs = acts[0]["observation"]
        assert obs["patient_response_zh"] == "ZH[hello]"
        assert obs["patient_responses_all_literacy_zh"] == {
            "high": "ZH[hello]", "medium": "ZH[hello (m)]", "low": "ZH[hello (l)]"}
        assert "patient_response_zh" not in acts[1]["observation"]
