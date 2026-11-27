"""Unit tests for the two-layer Independent Verifier (SPEC §7.5, PLAN T12)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lightbox.agent import IndependentVerifier, VerifierLLMOutput
from lightbox.api import create_app
from lightbox.models import (
    SIGLIP2_MODEL_ID,
    GenerationResult,
    Message,
    ModelCaps,
    ModelRouter,
    ToolSchema,
    Usage,
)
from lightbox.services import LightboxServices, build_demo_dataset
from lightbox.tools import ToolContext, ToolRegistry


class _FakeVerifierModel:
    """Scripted fake reviewer model returning predetermined flagged photo IDs."""

    def __init__(
        self,
        *,
        flagged_ids: list[str] | None = None,
        summary: str = "Independent check complete.",
        model_id: str = "gemini-2.5-flash",
        provider: str = "gemini",
        family: str = "google",
    ) -> None:
        self.id = model_id
        self.provider = provider
        self.family = family
        self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=False)
        self.flagged_ids = list(flagged_ids or [])
        self.summary = summary
        self.received_messages: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        _ = (tools, max_tokens, effort)
        self.received_messages.append(messages)
        payload = {
            "flagged": [
                {"photo_id": pid, "reason": "Subject looks potentially sharp / intentional"}
                for pid in self.flagged_ids
            ],
            "summary": self.summary,
        }
        raw_text = json.dumps(payload, ensure_ascii=False)
        parsed: Any = (
            response_schema.model_validate(payload) if response_schema is not None else None
        )
        return GenerationResult(
            text=raw_text,
            parsed=parsed,
            usage=Usage(in_tokens=400, out_tokens=60),
            cost_usd=0.0001,
            provider=self.provider,
            model=self.id,
        )


def _clear_all_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LIGHTBOX_LOCAL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # Prevent reading developer's real macOS keychain during tests
    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda _var: None)


def test_verifier_model_selection_cross_family_fallback_and_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all_api_keys(monkeypatch)

    with LightboxServices.open(tmp_path / "home") as svcs:
        verifier = IndependentVerifier(svcs)

        # 1. No API keys configured -> cleanly skips Layer 2
        c0 = verifier.resolve_reviewer_model()
        assert c0.alias is None
        assert c0.skipped_reason == "no_api_key"

        # 2. Both Anthropic (agent) and Gemini (reviewer) keys -> picks cross-family Google model
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key-1234567890")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey123456")
        c1 = verifier.resolve_reviewer_model()
        assert c1.alias == "gemini-2-5-flash"
        assert c1.cross_family is True
        assert c1.spec is not None and c1.spec.family == "google"

        # 3. Only Anthropic key (single key) -> falls back to same-family agent model
        monkeypatch.delenv("GEMINI_API_KEY")
        c2 = verifier.resolve_reviewer_model()
        assert c2.alias == "claude-sonnet-5-5"
        assert c2.cross_family is False

        # 4. Local profile disables Layer 2 regardless of keys
        c3 = verifier.resolve_reviewer_model(profile="local")
        assert c3.alias is None
        assert c3.skipped_reason == "disabled_in_profile"


def test_verifier_layer1_curation_near_duplicates(tmp_path: Path) -> None:
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)

    with LightboxServices.open(tmp_path / "home") as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        photos = svcs.repo.list_photos(scan.session_id)
        p0, p1, p2 = photos[0], photos[1], photos[2]

        verifier = IndependentVerifier(svcs)

        # Without vectors -> cleanly skips and returns empty list
        assert verifier.check_curation_near_duplicates([p0.id, p1.id, p2.id]) == []

        # Insert synthetic 768-d vectors: p0 & p1 nearly identical (cos > 0.99), p2 orthogonal
        v_a = np.zeros(768, dtype=np.float32)
        v_a[0] = 1.0
        v_a_near = np.zeros(768, dtype=np.float32)
        v_a_near[0] = 0.99
        v_a_near[1] = 0.141
        v_b = np.zeros(768, dtype=np.float32)
        v_b[10] = 1.0

        svcs.vector_index.upsert(p0.primary_content_fp, SIGLIP2_MODEL_ID, v_a)
        svcs.vector_index.upsert(p1.primary_content_fp, SIGLIP2_MODEL_ID, v_a_near)
        svcs.vector_index.upsert(p2.primary_content_fp, SIGLIP2_MODEL_ID, v_b)

        pairs = verifier.check_curation_near_duplicates([p0.id, p1.id, p2.id])
        assert len(pairs) == 1
        assert {pairs[0].photo_id_a, pairs[0].photo_id_b} == {p0.id, p1.id}
        assert pairs[0].similarity > 0.94


def test_verifier_layer2_downgrades_flagged_photos_to_yellow_and_redrafts_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all_api_keys(monkeypatch)
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)

    with LightboxServices.open(tmp_path / "home") as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        initial_draft = svcs.plan.draft_obvious_rejects(scan.session_id)
        assert initial_draft.plan is not None
        assert len(initial_draft.items) >= 2

        flagged_pid = initial_draft.items[0].photo_id
        remaining_expected = [it.photo_id for it in initial_draft.items[1:]]

        # Inject a Fake cross-family reviewer that flags the first item
        fake_reviewer = _FakeVerifierModel(
            flagged_ids=[flagged_pid, "non_existent_hallucinated_id"],
            summary="1 suspicious shot spared for human review.",
        )
        router = ModelRouter(
            svcs.repo,
            svcs.app_paths,
            svcs.config,
            adapters={"reviewer": fake_reviewer},
        )
        svcs.override_model_router(router)

        verifier = IndependentVerifier(svcs)
        final_draft = asyncio.run(
            verifier.verify_and_finalize_draft(
                scan.session_id,
                initial_draft,
                constraints=["Keep intentional motion blur shots"],
            )
        )

        # 1. Single fresh-context call (system + user only), untrusted metadata wrapped
        assert len(fake_reviewer.received_messages) == 1
        sent = fake_reviewer.received_messages[0]
        assert [m.role for m in sent] == ["system", "user"]
        assert "<untrusted_metadata>" in sent[1].text
        assert "Keep intentional motion blur shots" in sent[1].text
        assert len(sent[1].images) == len(initial_draft.items)

        # 2. Flagged photo was downgraded to `yellow` (`unsure`) and removed from the final plan
        eff_flagged = svcs.repo.get_effective_verdict(flagged_pid)
        assert eff_flagged is not None
        assert eff_flagged.bucket == "yellow"
        assert "verifier_flagged" in eff_flagged.reason_codes

        assert final_draft.plan is not None
        assert final_draft.plan.status == "reviewed"
        assert final_draft.plan.id != initial_draft.plan.id
        assert [it.photo_id for it in final_draft.items] == remaining_expected

        # Initial pre-filter plan was marked stale
        old_plan = svcs.repo.get_plan(initial_draft.plan.id)
        assert old_plan is not None and old_plan.status == "stale"

        # 3. Verifier report captures Layer 2 status, cross-family flag, and flagged_photos (hallucinated ID ignored)
        report = final_draft.verifier_report
        assert report["layer"] == "two_layer"
        assert report["layer2"]["status"] == "flagged"
        assert report["layer2"]["cross_family"] is True
        assert report["layer2"]["flagged_count"] == 1
        assert len(report["flagged_photos"]) == 1
        assert report["flagged_photos"][0]["photo_id"] == flagged_pid


def test_verifier_respects_user_confirmed_rejects_and_all_flagged_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all_api_keys(monkeypatch)
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)

    with LightboxServices.open(tmp_path / "home") as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        photos = svcs.repo.list_photos(scan.session_id)
        normal_photo = next(
            p
            for p in photos
            if not p.is_protected
            and (v := svcs.repo.get_effective_verdict(p.id)) is not None
            and v.bucket == "normal"
        )
        svcs.review.set_verdict(normal_photo.id, bucket="red", source="user")

        # 1. Plan with ONLY user-confirmed reject -> Layer 2 skips with `all_user_confirmed`
        user_only_draft = svcs.plan.create_plan(scan.session_id, [normal_photo.id])
        fake_reviewer = _FakeVerifierModel(flagged_ids=[normal_photo.id])
        svcs.override_model_router(
            ModelRouter(
                svcs.repo,
                svcs.app_paths,
                svcs.config,
                adapters={"reviewer": fake_reviewer},
            )
        )
        verifier = IndependentVerifier(svcs)
        res_user = asyncio.run(
            verifier.verify_and_finalize_draft(scan.session_id, user_only_draft)
        )
        assert len(fake_reviewer.received_messages) == 0
        assert res_user.verifier_report["layer2"]["status"] == "skipped"
        assert res_user.verifier_report["layer2"]["skipped_reason"] == "all_user_confirmed"
        assert res_user.plan is not None and res_user.selected_count == 1

        # 2. If ALL machine-rejected photos in a plan are flagged -> plan becomes None, all downgraded to yellow
        machine_draft = svcs.plan.draft_obvious_rejects(scan.session_id)
        assert machine_draft.plan is not None
        machine_ids = [
            it.photo_id
            for it in machine_draft.items
            if (v := svcs.repo.get_effective_verdict(it.photo_id)) is not None
            and v.source != "user"
        ]
        single_machine_draft = svcs.plan.create_plan(scan.session_id, [machine_ids[0]])
        flag_all_model = _FakeVerifierModel(flagged_ids=[machine_ids[0]])
        svcs.override_model_router(
            ModelRouter(
                svcs.repo,
                svcs.app_paths,
                svcs.config,
                adapters={"reviewer": flag_all_model},
            )
        )
        res_empty = asyncio.run(
            IndependentVerifier(svcs).verify_and_finalize_draft(
                scan.session_id, single_machine_draft
            )
        )
        assert res_empty.plan is None
        assert res_empty.selected_count == 0
        assert len(res_empty.verifier_report["flagged_photos"]) == 1
        v_after = svcs.repo.get_effective_verdict(machine_ids[0])
        assert v_after is not None and v_after.bucket == "yellow"


def test_create_plan_tool_and_api_attach_verifier_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_all_api_keys(monkeypatch)
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)
    home_dir = tmp_path / "home"

    # 1. ToolRegistry `create_plan` returns enriched verifier_report
    reg = ToolRegistry()
    with LightboxServices.open(home_dir) as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        ctx = ToolContext(services=svcs, surface="mcp:test")
        tool_res = reg.execute(
            "lightbox_create_plan",
            {"session_id": scan.session_id, "action": "quarantine_move"},
            ctx,
        )
        assert not tool_res.is_error
        rep = tool_res.data["verifier_report"]
        assert rep["passed"] is True
        assert rep["layer2"]["status"] == "skipped"
        assert rep["layer2"]["skipped_reason"] == "no_api_key"

    # 2. FastAPI `POST /api/sessions/{id}/plans/obvious-rejects` runs two-layer verifier with injected model
    fake = _FakeVerifierModel(flagged_ids=[], summary="All rejects look clearly blurry.")
    app = create_app(
        app_paths=LightboxServices.open(home_dir).app_paths,
        port=8765,
        model_router_factory=lambda s: ModelRouter(
            s.repo,
            s.app_paths,
            s.config,
            adapters={"reviewer": fake},
        ),
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        auth = client.post(
            "/api/auth/session",
            headers={"Origin": "http://127.0.0.1:8765", "X-Lightbox-Local-Confirm": "1"},
        )
        assert auth.status_code == 200
        csrf = auth.json()["csrf_token"]
        resp = client.post(
            f"/api/sessions/{scan.session_id}/plans/obvious-rejects",
            headers={"Origin": "http://127.0.0.1:8765", "X-Lightbox-CSRF": csrf},
            json={},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["verifier_report"]["layer"] == "two_layer"
        assert data["verifier_report"]["layer2"]["status"] == "passed"
        assert data["verifier_report"]["layer2"]["model_id"] == "gemini-2.5-flash"
        assert data["plan"]["status"] == "reviewed"
        assert isinstance(VerifierLLMOutput(flagged=[], summary="ok"), VerifierLLMOutput)
