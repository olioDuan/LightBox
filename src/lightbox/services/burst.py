"""BurstService: burst listing, pick recommendation, and confirmation with `burst_non_pick` (SPEC §4.2)."""

from __future__ import annotations

from dataclasses import dataclass

from lightbox.i18n import REASON_BURST_NON_PICK, REASON_BURST_PICK, is_tech_reject_reason
from lightbox.store import BurstMemberRecord, BurstRecord, LightboxRepository


@dataclass(frozen=True, slots=True)
class BurstDetail:
    burst: BurstRecord
    members: tuple[BurstMemberRecord, ...]


@dataclass(frozen=True, slots=True)
class ConfirmBurstResult:
    burst_id: str
    pick_photo_id: str
    non_pick_photo_ids: tuple[str, ...]
    confirmed: bool


class BurstService:
    """Manages burst group review, recommendations, and explicit confirmation."""

    def __init__(self, repo: LightboxRepository) -> None:
        self.repo = repo

    def get_burst_detail(self, burst_id: str) -> BurstDetail:
        burst = self.repo.get_burst(burst_id)
        if burst is None:
            raise ValueError(f"Unknown burst_id: {burst_id}")
        return BurstDetail(burst=burst, members=tuple(self.repo.list_burst_members(burst_id)))

    def find_burst_for_photo(self, session_id: str, photo_id: str) -> str | None:
        for burst in self.repo.list_bursts(session_id):
            for member in self.repo.list_burst_members(burst.id):
                if member.photo_id == photo_id:
                    return burst.id
        return None

    def list_bursts(
        self,
        session_id: str,
        *,
        include_quarantined: bool = False,
    ) -> list[BurstDetail]:
        bursts = self.repo.list_bursts(session_id)
        if include_quarantined:
            return [
                BurstDetail(burst=b, members=tuple(self.repo.list_burst_members(b.id)))
                for b in bursts
            ]
        quarantined = self.repo.list_quarantined_photo_ids(session_id)
        result: list[BurstDetail] = []
        for b in bursts:
            active_members = tuple(
                m for m in self.repo.list_burst_members(b.id) if m.photo_id not in quarantined
            )
            if len(active_members) >= 2:
                result.append(BurstDetail(burst=b, members=active_members))
        return result

    def recommend_burst_pick(
        self,
        burst_id: str,
        pick_photo_id: str,
        *,
        note: str | None = None,
    ) -> BurstDetail:
        """Record an AI/system recommendation for a burst without confirming it."""
        burst = self.repo.get_burst(burst_id)
        if burst is None:
            raise ValueError(f"Unknown burst_id: {burst_id}")
        members = self.repo.list_burst_members(burst_id)
        member_ids = {m.photo_id for m in members}
        if pick_photo_id not in member_ids:
            raise ValueError(f"Photo {pick_photo_id} is not a member of burst {burst_id}")

        for member in members:
            is_pick = member.photo_id == pick_photo_id
            self.repo.upsert_burst_member(
                burst_id,
                member.photo_id,
                rank=member.rank,
                is_pick=is_pick,
                ai_note=note if is_pick else member.ai_note,
            )
        return BurstDetail(burst=burst, members=tuple(self.repo.list_burst_members(burst_id)))

    def confirm_burst(
        self,
        burst_id: str,
        pick_photo_id: str,
        *,
        actor: str = "user",
    ) -> ConfirmBurstResult:
        """Confirm a burst group pick and tag qualified non-picks with `burst_non_pick` (PLAN T06).

        Strictly preserves technical failure verdicts (`motion_blur`, `out_of_focus`, etc.) and
        never overrides `exempt` photos so T17 shooting digest cleanly separates technical rejects
        from burst selection culls.
        """
        burst = self.repo.get_burst(burst_id)
        if burst is None:
            raise ValueError(f"Unknown burst_id: {burst_id}")
        members = self.repo.list_burst_members(burst_id)
        member_ids = {m.photo_id for m in members}
        if pick_photo_id not in member_ids:
            raise ValueError(f"Photo {pick_photo_id} is not a member of burst {burst_id}")

        self.repo.confirm_burst(burst_id, pick_photo_id)

        # Keep the pick as normal (unless exempt)
        pick_photo = self.repo.get_photo(pick_photo_id)
        if pick_photo is not None and not pick_photo.is_protected:
            self.repo.add_verdict(
                photo_id=pick_photo_id,
                source="user",
                bucket="normal",
                confidence=1.0,
                reason_codes=(REASON_BURST_PICK,),
                detail={"burst_id": burst_id, "actor": actor},
                increment_decision_version=False,  # already bumped atomically by confirm_burst
            )

        non_picks: list[str] = []
        for member in members:
            if member.photo_id == pick_photo_id:
                continue
            photo = self.repo.get_photo(member.photo_id)
            if photo is None or photo.is_protected:
                continue

            current = self.repo.get_effective_verdict(member.photo_id)
            if current is not None and current.bucket == "exempt":
                continue

            # Preserve existing technical failure codes if frame was genuinely blurred/mis-shot;
            # otherwise mark purely as `burst_non_pick` for T17 separation.
            if (
                current is not None
                and current.bucket == "red"
                and is_tech_reject_reason(current.reason_codes)
            ):
                non_picks.append(member.photo_id)
                continue

            self.repo.add_verdict(
                photo_id=member.photo_id,
                source="user",
                bucket="red",
                confidence=1.0,
                reason_codes=(REASON_BURST_NON_PICK,),
                detail={"burst_id": burst_id, "pick_photo_id": pick_photo_id, "actor": actor},
                increment_decision_version=False,  # already bumped atomically by confirm_burst
            )
            non_picks.append(member.photo_id)

        return ConfirmBurstResult(
            burst_id=burst_id,
            pick_photo_id=pick_photo_id,
            non_pick_photo_ids=tuple(non_picks),
            confirmed=True,
        )


__all__ = ["BurstDetail", "BurstService", "ConfirmBurstResult"]
