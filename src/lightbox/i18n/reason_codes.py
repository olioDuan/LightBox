"""Language-independent verdict reason codes and bilingual (zh-CN / en-US) renderer (SPEC §4.2–4.3)."""

from __future__ import annotations

from collections.abc import Sequence

# --- Technical failure reason codes (count toward tech_reject in T17 digest) ---
REASON_EXTREME_EXPOSURE = "extreme_exposure"
REASON_MOTION_BLUR = "motion_blur"
REASON_OUT_OF_FOCUS = "out_of_focus"
REASON_SEVERE_BLUR = "severe_blur"
REASON_BORDERLINE_BLUR = "borderline_blur"
REASON_LOW_RELATIVE_SHARPNESS = "low_relative_sharpness"
REASON_FACE_OUT_OF_FOCUS = "face_out_of_focus"
REASON_EYES_CLOSED = "eyes_closed"

# --- Burst selection reason codes (strictly separated from technical failures!) ---
REASON_BURST_NON_PICK = "burst_non_pick"
REASON_BURST_PICK = "burst_pick"

# --- Creative intent protection & exemption reason codes ---
REASON_IN_CAMERA_PROTECTED = "in_camera_protected"
REASON_IN_CAMERA_RATED = "in_camera_rated"
REASON_AEB_BRACKET = "aeb_bracket"
REASON_LONG_EXPOSURE_PROTECTED = "long_exposure_protected"
REASON_EXTREME_EV_COMP_PROTECTED = "extreme_ev_comp_protected"
REASON_SUSPECTED_ICM = "suspected_icm"
REASON_PANNING_PROTECTED = "panning_protected"
REASON_SHALLOW_DOF_SHARP = "shallow_dof_sharp"
REASON_SLOW_SHUTTER_NOTE = "slow_shutter_note"

# --- Human / override reason codes ---
REASON_USER_RESCUE = "user_rescue"
REASON_USER_REJECT = "user_reject"
REASON_USER_UNSURE = "user_unsure"
REASON_USER_RESET = "user_reset"
REASON_USER_RESTORED = "user_restored"

# --- VLM fallback reason codes ---
REASON_VLM_KEEP = "vlm_keep"
REASON_VLM_REJECT = "vlm_reject"
REASON_VLM_UNSURE = "vlm_unsure"

TECH_REJECT_REASON_CODES: frozenset[str] = frozenset(
    {
        REASON_EXTREME_EXPOSURE,
        REASON_MOTION_BLUR,
        REASON_OUT_OF_FOCUS,
        REASON_SEVERE_BLUR,
        REASON_BORDERLINE_BLUR,
        REASON_LOW_RELATIVE_SHARPNESS,
        REASON_FACE_OUT_OF_FOCUS,
        REASON_EYES_CLOSED,
    }
)

BURST_CULL_REASON_CODES: frozenset[str] = frozenset({REASON_BURST_NON_PICK})

EXEMPT_REASON_CODES: frozenset[str] = frozenset(
    {
        REASON_IN_CAMERA_PROTECTED,
        REASON_IN_CAMERA_RATED,
        REASON_AEB_BRACKET,
    }
)

REASON_MESSAGES: dict[str, dict[str, str]] = {
    REASON_EXTREME_EXPOSURE: {
        "zh-CN": "极黑或极白误拍（几乎无有效画面信息）",
        "en-US": "Extreme under- or over-exposure mis-shot (almost no visual detail)",
    },
    REASON_MOTION_BLUR: {
        "zh-CN": "明显方向性拖影（疑似手抖或运动模糊）",
        "en-US": "Noticeable directional motion blur or camera shake",
    },
    REASON_OUT_OF_FOCUS: {
        "zh-CN": "全图严重失焦模糊",
        "en-US": "Severely out of focus across the frame",
    },
    REASON_SEVERE_BLUR: {
        "zh-CN": "全图清晰度严重不足（绝对与同场相对均偏低）",
        "en-US": "Severely soft image (below both absolute and session-relative baselines)",
    },
    REASON_BORDERLINE_BLUR: {
        "zh-CN": "清晰度处于临界区间，建议人工复核",
        "en-US": "Borderline sharpness; kept for manual review",
    },
    REASON_LOW_RELATIVE_SHARPNESS: {
        "zh-CN": "相比同场同器材照片偏软，建议人工复核",
        "en-US": "Softer than session peers from the same camera/lens; kept for review",
    },
    REASON_FACE_OUT_OF_FOCUS: {
        "zh-CN": "主体人脸疑似失焦（仅标黄保留，不自动判废）",
        "en-US": "Subject face appears soft (marked yellow for review, never auto-red)",
    },
    REASON_EYES_CLOSED: {
        "zh-CN": "检测到人物闭眼（仅标黄保留，建议人工复核）",
        "en-US": "Subject appears to have eyes closed (marked yellow for review, never auto-red)",
    },
    REASON_BURST_NON_PICK: {
        "zh-CN": "已确认连拍组中未选中的合格次选帧",
        "en-US": "Non-pick frame from a confirmed burst group",
    },
    REASON_BURST_PICK: {
        "zh-CN": "连拍组推荐首选帧",
        "en-US": "Recommended pick frame in burst group",
    },
    REASON_IN_CAMERA_PROTECTED: {
        "zh-CN": "机内已加锁或文件只读，自动豁免保护",
        "en-US": "In-camera protected or read-only file; automatically exempt",
    },
    REASON_IN_CAMERA_RATED: {
        "zh-CN": "机内已标星或标记，自动豁免保护",
        "en-US": "In-camera rated or labeled; automatically exempt",
    },
    REASON_AEB_BRACKET: {
        "zh-CN": "包围曝光或对焦包围序列，自动豁免保护",
        "en-US": "Exposure or focus bracketing sequence; automatically exempt",
    },
    REASON_LONG_EXPOSURE_PROTECTED: {
        "zh-CN": "长曝光创作保护（快门 ≥ 1 秒），不判为明显废片",
        "en-US": "Long-exposure creative protection (shutter >= 1s); capped at yellow",
    },
    REASON_EXTREME_EV_COMP_PROTECTED: {
        "zh-CN": "刻意曝光补偿保护（|EV| ≥ 1.5），不判为明显废片",
        "en-US": "Deliberate exposure compensation protection (|EV| >= 1.5); capped at yellow",
    },
    REASON_SUSPECTED_ICM: {
        "zh-CN": "连续慢门软糊，疑似创意晃拍（ICM），降级为待复核",
        "en-US": "Consecutive slow-shutter soft frames (suspected ICM); downgraded to yellow",
    },
    REASON_PANNING_PROTECTED: {
        "zh-CN": "检测到追焦摇拍（主体清晰、背景方向性虚化），保留正常",
        "en-US": "Panning shot detected (sharp subject with directional background blur); kept normal",
    },
    REASON_SHALLOW_DOF_SHARP: {
        "zh-CN": "大光圈浅景深且主体局部清晰",
        "en-US": "Sharp subject with shallow depth of field",
    },
    REASON_SLOW_SHUTTER_NOTE: {
        "zh-CN": "快门慢于 1/焦距 安全快门（仅供参考）",
        "en-US": "Shutter slower than 1/focal-length guideline (informational only)",
    },
    REASON_USER_RESCUE: {
        "zh-CN": "人工救回保留",
        "en-US": "Rescued / kept by user",
    },
    REASON_USER_REJECT: {
        "zh-CN": "人工标记淘汰",
        "en-US": "Marked as reject by user",
    },
    REASON_USER_UNSURE: {
        "zh-CN": "人工标记待复核",
        "en-US": "Marked as uncertain by user",
    },
    REASON_USER_RESET: {
        "zh-CN": "人工重置判定",
        "en-US": "Verdict reset by user",
    },
    REASON_USER_RESTORED: {
        "zh-CN": "已从废片箱放回保留",
        "en-US": "Restored to keep from trash folder",
    },
    REASON_VLM_KEEP: {
        "zh-CN": "VLM 视觉复核建议保留",
        "en-US": "VLM triage recommended keeping",
    },
    REASON_VLM_REJECT: {
        "zh-CN": "VLM 视觉复核建议淘汰",
        "en-US": "VLM triage recommended rejecting",
    },
    REASON_VLM_UNSURE: {
        "zh-CN": "VLM 视觉复核仍存疑，保留供人工确认",
        "en-US": "VLM triage uncertain; kept for manual review",
    },
}


def render_reason(code: str, lang: str = "zh-CN") -> str:
    """Render a language-independent reason code into human-readable text."""
    entry = REASON_MESSAGES.get(code)
    if entry is None:
        return code
    normalized = "en-US" if lang.lower().startswith("en") else "zh-CN"
    return entry.get(normalized, entry["en-US"])


def render_reasons(codes: Sequence[str], lang: str = "zh-CN") -> list[str]:
    """Render a sequence of reason codes in the requested language."""
    return [render_reason(code, lang=lang) for code in codes]


def is_tech_reject_reason(codes: Sequence[str]) -> bool:
    """Return True if any code represents a genuine technical capture failure."""
    return any(code in TECH_REJECT_REASON_CODES for code in codes)


def is_burst_cull_reason(codes: Sequence[str]) -> bool:
    """Return True if codes indicate a burst non-pick without a technical failure."""
    code_set = set(codes)
    return bool(code_set & BURST_CULL_REASON_CODES) and not bool(
        code_set & TECH_REJECT_REASON_CODES
    )


__all__ = [
    "BURST_CULL_REASON_CODES",
    "EXEMPT_REASON_CODES",
    "REASON_AEB_BRACKET",
    "REASON_BORDERLINE_BLUR",
    "REASON_BURST_NON_PICK",
    "REASON_BURST_PICK",
    "REASON_EXTREME_EV_COMP_PROTECTED",
    "REASON_EXTREME_EXPOSURE",
    "REASON_EYES_CLOSED",
    "REASON_FACE_OUT_OF_FOCUS",
    "REASON_IN_CAMERA_PROTECTED",
    "REASON_IN_CAMERA_RATED",
    "REASON_LONG_EXPOSURE_PROTECTED",
    "REASON_LOW_RELATIVE_SHARPNESS",
    "REASON_MESSAGES",
    "REASON_MOTION_BLUR",
    "REASON_OUT_OF_FOCUS",
    "REASON_PANNING_PROTECTED",
    "REASON_SEVERE_BLUR",
    "REASON_SHALLOW_DOF_SHARP",
    "REASON_SLOW_SHUTTER_NOTE",
    "REASON_SUSPECTED_ICM",
    "REASON_USER_REJECT",
    "REASON_USER_RESCUE",
    "REASON_USER_RESET",
    "REASON_USER_RESTORED",
    "REASON_USER_UNSURE",
    "REASON_VLM_KEEP",
    "REASON_VLM_REJECT",
    "REASON_VLM_UNSURE",
    "TECH_REJECT_REASON_CODES",
    "is_burst_cull_reason",
    "is_tech_reject_reason",
    "render_reason",
    "render_reasons",
]
