import type { PhotoItem } from "../api";
import type { Verdict } from "./ui";

/**
 * Finalized photos (kept, rejected, protected) — mirrors the Review view's rule so the
 * contact sheets on every page agree with the loupe.
 */
export function isPhotoFinalized(item: PhotoItem): boolean {
  if (item.bucket === "red") return true;
  if (item.bucket === "exempt" || item.is_protected) return true;
  if (
    item.bucket === "normal" &&
    item.source === "user" &&
    !item.reason_codes.includes("user_reset")
  ) {
    return true;
  }
  return false;
}

/**
 * Verdict color for a photo: reject red, unsure amber, protected neutral, kept green,
 * skipped dashed (only known inside a review pass), otherwise none.
 */
export function verdictForPhoto(item: PhotoItem, skipped = false): Verdict | null {
  if (item.bucket === "red") return "reject";
  if (item.bucket === "yellow") return "unsure";
  if (item.bucket === "exempt" || item.is_protected) return "protected";
  if (isPhotoFinalized(item)) return "keep";
  if (skipped) return "skip";
  return null;
}
