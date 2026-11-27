import {
  forwardRef,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ComponentType,
  type CSSProperties,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import type { IconProps } from "reicon-react";

/* ────────────────────────────────────────────────────────────────────────────
 * Icon — Reicon glyph at 18px / 1.5px stroke, centred in a transparent 24×24 box
 * ──────────────────────────────────────────────────────────────────────────── */

export type ReiconGlyph = ComponentType<IconProps>;

export function Icon({
  as: Glyph,
  className = "",
  title,
}: {
  as: ReiconGlyph;
  className?: string;
  title?: string;
}) {
  return (
    <span
      aria-hidden={title ? undefined : true}
      title={title}
      className={`inline-flex h-6 w-6 shrink-0 items-center justify-center ${className}`}
    >
      <Glyph size={18} strokeWidth={1.5} />
    </span>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Button
 * ──────────────────────────────────────────────────────────────────────────── */

type ButtonVariant = "primary" | "mark" | "secondary" | "ghost" | "quiet";
type ButtonSize = "sm" | "md" | "lg";

const VARIANT_CLASS: Record<ButtonVariant, string> = {
  primary: "bg-ink text-white hover:bg-black active:bg-[#2b2b2b]",
  mark: "bg-reject text-white hover:bg-reject-ink active:bg-[#9a2c23]",
  secondary: "bg-table text-ink hover:bg-table-2",
  ghost: "text-ink-2 hover:bg-table hover:text-ink",
  quiet: "text-ink-2 hover:text-ink",
};

const SIZE_CLASS: Record<ButtonSize, string> = {
  sm: "h-8 gap-1 rounded-chip px-3 text-sm",
  md: "h-10 gap-2 rounded-btn px-4 text-sm",
  lg: "h-12 gap-2 rounded-btn px-6 text-base",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: ReiconGlyph;
  iconRight?: ReiconGlyph;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, iconRight, className = "", children, type, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type ?? "button"}
      className={`inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap font-medium tabular-nums transition-colors duration-150 disabled:pointer-events-none disabled:opacity-40 ${VARIANT_CLASS[variant]} ${SIZE_CLASS[size]} ${icon && size !== "sm" ? "pl-3" : ""} ${className}`}
      {...rest}
    >
      {icon && <Icon as={icon} className={size === "sm" ? "-ml-1" : ""} />}
      {children}
      {iconRight && <Icon as={iconRight} className="-mr-1" />}
    </button>
  );
});

export const IconButton = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & {
    icon: ReiconGlyph;
    label: string;
    size?: "sm" | "md";
    variant?: "ghost" | "secondary" | "primary";
  }
>(function IconButton(
  { icon, label, size = "md", variant = "ghost", className = "", type, ...rest },
  ref,
) {
  const dims = size === "sm" ? "h-8 w-8 rounded-chip" : "h-10 w-10 rounded-btn";
  const tone =
    variant === "primary"
      ? "bg-ink text-white hover:bg-black"
      : variant === "secondary"
        ? "bg-table text-ink hover:bg-table-2"
        : "text-ink-2 hover:bg-table hover:text-ink";
  return (
    <button
      ref={ref}
      type={type ?? "button"}
      aria-label={label}
      title={label}
      className={`inline-flex shrink-0 items-center justify-center transition-colors duration-150 disabled:pointer-events-none disabled:opacity-30 ${dims} ${tone} ${className}`}
      {...rest}
    >
      <Icon as={icon} />
    </button>
  );
});

export function Kbd({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={`inline-flex h-5 min-w-5 items-center justify-center rounded-[6px] px-1 font-sans text-xs font-medium text-ink-3 ${className}`}
    >
      {children}
    </kbd>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Tabs — text segments; the active one sits on the table gray
 * ──────────────────────────────────────────────────────────────────────────── */

export interface TabItem<T extends string> {
  id: T;
  label: ReactNode;
  count?: number;
  attention?: boolean;
  verdict?: Verdict;
}

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  size = "md",
  wrap = true,
  variant = "plain",
  className = "",
}: {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  size?: "sm" | "md";
  wrap?: boolean;
  /** "segmented" sits the tabs in a table-gray tray with a white active pill. */
  variant?: "plain" | "segmented";
  className?: string;
}) {
  const segmented = variant === "segmented";
  return (
    <div
      role="tablist"
      className={`flex items-center gap-1 ${wrap ? "flex-wrap" : "flex-nowrap"} ${
        segmented ? "rounded-[14px] bg-table p-1" : ""
      } ${className}`}
    >
      {items.map((item) => {
        const active = item.id === value;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={`inline-flex items-center gap-2 whitespace-nowrap font-medium tabular-nums transition-[background-color,color,box-shadow] duration-150 ${
              segmented
                ? `h-8 rounded-[10px] px-3 text-sm focus-visible:outline-offset-0 ${
                    active
                      ? "bg-paper text-ink shadow-[0_1px_2px_rgba(24,24,24,0.08),0_0_0_0.5px_rgba(24,24,24,0.06)]"
                      : "text-ink-2 hover:bg-table-2 hover:text-ink"
                  }`
                : size === "sm"
                  ? `h-8 rounded-chip px-3 text-sm ${active ? "bg-table text-ink" : "text-ink-2 hover:text-ink"}`
                  : `h-10 rounded-btn px-4 text-sm ${active ? "bg-table text-ink" : "text-ink-2 hover:text-ink"}`
            }`}
          >
            {item.verdict && <StatusDot verdict={item.verdict} className="-mx-2" />}
            <span>{item.label}</span>
            {item.count !== undefined && (
              <span
                className={`text-sm font-normal ${
                  item.attention ? "font-medium text-reject" : active ? "text-ink-2" : "text-ink-3"
                }`}
              >
                {item.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Verdicts — four colors a photographer reads at a glance
 * keep = green, unsure = amber, skip = dashed gray, reject = red; protected = neutral lock
 * ──────────────────────────────────────────────────────────────────────────── */

export type Verdict = "keep" | "unsure" | "skip" | "reject" | "protected";

export const VERDICT_STYLE: Record<
  Verdict,
  { color: string; soft: string; ink: string; dashed?: boolean }
> = {
  keep: { color: "var(--color-keep)", soft: "var(--color-keep-soft)", ink: "var(--color-keep-ink)" },
  unsure: {
    color: "var(--color-unsure)",
    soft: "var(--color-unsure-soft)",
    ink: "var(--color-unsure-ink)",
  },
  skip: {
    color: "var(--color-skip)",
    soft: "var(--color-table)",
    ink: "var(--color-ink-2)",
    dashed: true,
  },
  reject: {
    color: "var(--color-reject)",
    soft: "var(--color-reject-soft)",
    ink: "var(--color-reject-ink)",
  },
  protected: { color: "var(--color-ink-3)", soft: "var(--color-table)", ink: "var(--color-ink-2)" },
};

/** 8px status dot (dashed ring for "skip"), centred in a 24px box like icons. */
export function StatusDot({ verdict, className = "" }: { verdict: Verdict; className?: string }) {
  const v = VERDICT_STYLE[verdict];
  return (
    <span
      aria-hidden="true"
      className={`inline-flex h-6 w-6 shrink-0 items-center justify-center ${className}`}
    >
      <span
        className="block h-2 w-2 rounded-full"
        style={
          v.dashed
            ? { boxShadow: "none", border: `1.5px dashed ${v.color}`, width: 10, height: 10 }
            : { background: v.color }
        }
      />
    </span>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Progress line
 * ──────────────────────────────────────────────────────────────────────────── */

export function ProgressLine({
  value,
  className = "",
  tone = "ink",
  indeterminate = false,
}: {
  value?: number;
  className?: string;
  tone?: "ink" | "mark" | "keep" | "unsure";
  indeterminate?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={`h-1 overflow-hidden rounded-full bg-table-2 ${indeterminate ? "progress-indeterminate" : ""} ${className}`}
    >
      {!indeterminate && (
        <div
          className={`h-full rounded-full transition-[width] duration-500 ease-out ${
            tone === "mark" ? "bg-reject" : tone === "keep" ? "bg-keep" : tone === "unsure" ? "bg-unsure" : "bg-ink"
          }`}
          style={{ width: `${pct}%` }}
        />
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Notices — floating pill messages, never a box inside the page flow
 * ──────────────────────────────────────────────────────────────────────────── */

export interface NoticeAction {
  label: string;
  onClick: () => void;
}

export function Notice({
  tone = "info",
  children,
  actions = [],
  onDismiss,
  dismissLabel = "×",
}: {
  tone?: "info" | "error";
  children: ReactNode;
  actions?: NoticeAction[];
  onDismiss?: () => void;
  dismissLabel?: string;
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className="fade-enter pointer-events-auto flex max-w-[min(720px,calc(100vw-32px))] items-center gap-4 rounded-[24px] bg-ink py-2 pl-5 pr-2 text-sm text-white"
    >
      {tone === "error" && <span className="h-2 w-2 shrink-0 rounded-full bg-mark" />}
      <span className="min-w-0 flex-1 py-1 leading-5">{children}</span>
      {actions.length > 0 && (
        <span className="flex shrink-0 items-center gap-1">
          {actions.map((a) => (
            <button
              key={a.label}
              type="button"
              onClick={a.onClick}
              className="h-8 rounded-chip px-3 text-sm font-medium text-white hover:bg-white/15"
            >
              {a.label}
            </button>
          ))}
        </span>
      )}
      {onDismiss && (
        <button
          type="button"
          aria-label={dismissLabel}
          onClick={onDismiss}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-chip text-white/60 hover:bg-white/15 hover:text-white"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M7 7l10 10M17 7 7 17" />
          </svg>
        </button>
      )}
    </div>
  );
}

export function NoticeDock({ children, bottom = 32 }: { children: ReactNode; bottom?: number }) {
  return (
    <div
      className="pointer-events-none fixed inset-x-0 z-50 flex flex-col items-center gap-2 px-4"
      style={{ bottom }}
    >
      {children}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Dialog shell
 * ──────────────────────────────────────────────────────────────────────────── */

export function Dialog({
  children,
  labelledBy,
  onDismiss,
  width = 560,
  zIndex = 50,
  className = "",
}: {
  children: ReactNode;
  labelledBy?: string;
  onDismiss?: () => void;
  width?: number;
  zIndex?: number;
  className?: string;
}) {
  useEffect(() => {
    if (!onDismiss) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  return (
    <div
      className="fade-enter fixed inset-0 flex items-center justify-center bg-[#181818]/30 p-4 backdrop-blur-[6px]"
      style={{ zIndex }}
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onDismiss?.();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className={`dialog-enter flex max-h-[88vh] w-full flex-col overflow-hidden rounded-dialog bg-paper ${className}`}
        style={{ maxWidth: width }}
      >
        {children}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Menu — desktop menu with open/close delays and a safe-triangle for submenus
 * ──────────────────────────────────────────────────────────────────────────── */

export interface MenuItem {
  key: string;
  label: ReactNode;
  icon?: ReactNode;
  hint?: ReactNode;
  onSelect?: () => void;
  disabled?: boolean;
  tone?: "default" | "mark";
  submenu?: MenuItem[];
  /** Adds breathing room above this item to separate groups (no divider lines). */
  groupStart?: boolean;
}

const SUBMENU_OPEN_DELAY = 120;
const SUBMENU_CLOSE_DELAY = 280;
const TRIANGLE_HOLD_MAX = 600;

type Point = { x: number; y: number };

function sign(p1: Point, p2: Point, p3: Point): number {
  return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

function pointInTriangle(p: Point, a: Point, b: Point, c: Point): boolean {
  const d1 = sign(p, a, b);
  const d2 = sign(p, b, c);
  const d3 = sign(p, c, a);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

function MenuRow({
  item,
  active,
  onEnter,
  onLeave,
  onActivate,
  onKeyDown,
  rowRef,
}: {
  item: MenuItem;
  active: boolean;
  onEnter: (e: React.MouseEvent) => void;
  onLeave?: (e: React.MouseEvent) => void;
  onActivate: () => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLButtonElement>) => void;
  rowRef?: (el: HTMLButtonElement | null) => void;
}) {
  return (
    <button
      ref={rowRef}
      type="button"
      role="menuitem"
      aria-haspopup={item.submenu ? "menu" : undefined}
      aria-expanded={item.submenu ? active : undefined}
      disabled={item.disabled}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onClick={onActivate}
      onKeyDown={onKeyDown}
      className={`flex h-10 w-full items-center gap-2 rounded-btn px-2 text-left text-sm transition-colors duration-100 focus:outline-none focus-visible:bg-table disabled:opacity-40 ${
        item.groupStart ? "mt-2" : ""
      } ${active ? "bg-table" : "hover:bg-table"} ${item.tone === "mark" ? "text-mark" : "text-ink"}`}
    >
      {item.icon ? (
        <span className="flex h-6 w-6 shrink-0 items-center justify-center">{item.icon}</span>
      ) : (
        <span className="w-1 shrink-0" />
      )}
      <span className="min-w-0 flex-1 truncate">{item.label}</span>
      {item.hint && <span className="shrink-0 text-sm text-ink-3">{item.hint}</span>}
      {item.submenu && (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mr-1 shrink-0 text-ink-3" aria-hidden="true">
          <path d="m10 7 5 5-5 5" />
        </svg>
      )}
    </button>
  );
}

export function Menu({
  anchorRef,
  open,
  onClose,
  items,
  align = "start",
  header,
  width = 240,
}: {
  anchorRef: RefObject<HTMLElement | null>;
  open: boolean;
  onClose: () => void;
  items: MenuItem[];
  align?: "start" | "end";
  header?: ReactNode;
  width?: number;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const subRef = useRef<HTMLDivElement | null>(null);
  const rowRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const subRowRefs = useRef<HTMLButtonElement[]>([]);
  const [rootPos, setRootPos] = useState<CSSProperties>({ visibility: "hidden" });
  const [subPos, setSubPos] = useState<CSSProperties>({ visibility: "hidden" });
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [focusSubFirst, setFocusSubFirst] = useState(false);

  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const holdTimer = useRef<number | null>(null);
  const apexRef = useRef<Point | null>(null);
  const pendingRef = useRef<MenuItem | null>(null);
  const pointerRef = useRef<Point>({ x: 0, y: 0 });

  const clearTimers = useCallback(() => {
    for (const t of [openTimer, closeTimer, holdTimer]) {
      if (t.current !== null) {
        window.clearTimeout(t.current);
        t.current = null;
      }
    }
  }, []);

  useEffect(() => {
    if (!open) {
      setOpenKey(null);
      apexRef.current = null;
      pendingRef.current = null;
      clearTimers();
    }
  }, [clearTimers, open]);

  useEffect(() => clearTimers, [clearTimers]);

  // Position the root panel beside its anchor; flip when it would leave the viewport.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const anchor = anchorRef.current?.getBoundingClientRect();
      const panel = rootRef.current;
      if (!anchor || !panel) return;
      const pw = panel.offsetWidth;
      const ph = panel.offsetHeight;
      let left = align === "end" ? anchor.right - pw : anchor.left;
      left = Math.max(8, Math.min(left, window.innerWidth - pw - 8));
      let top = anchor.bottom + 8;
      if (top + ph > window.innerHeight - 8) top = Math.max(8, anchor.top - ph - 8);
      setRootPos({ left, top, visibility: "visible" });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [align, anchorRef, open, items.length]);

  // Position the submenu beside its owner row.
  useLayoutEffect(() => {
    if (!open || !openKey) {
      setSubPos({ visibility: "hidden" });
      return;
    }
    const row = rowRefs.current.get(openKey)?.getBoundingClientRect();
    const root = rootRef.current?.getBoundingClientRect();
    const sub = subRef.current;
    if (!row || !root || !sub) return;
    const sw = sub.offsetWidth;
    const sh = sub.offsetHeight;
    let left = root.right + 4;
    if (left + sw > window.innerWidth - 8) left = root.left - sw - 4;
    let top = row.top - 8;
    top = Math.max(8, Math.min(top, window.innerHeight - sh - 8));
    setSubPos({ left, top, visibility: "visible" });
  }, [open, openKey]);

  useEffect(() => {
    if (focusSubFirst && openKey && subPos.visibility === "visible") {
      subRowRefs.current[0]?.focus();
      setFocusSubFirst(false);
    }
  }, [focusSubFirst, openKey, subPos.visibility]);

  // Outside click & Escape close the whole menu.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (rootRef.current?.contains(target)) return;
      if (subRef.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
        anchorRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [anchorRef, onClose, open]);

  // Focus the first row on open for keyboard users.
  useEffect(() => {
    if (!open) return;
    const id = window.requestAnimationFrame(() => {
      const first = items.find((i) => !i.disabled);
      if (first) rowRefs.current.get(first.key)?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(id);
  }, [items, open]);

  /** The safe triangle: apex where the pointer left the owner row, base = submenu's near edge. */
  const inSafeTriangle = useCallback((p: Point): boolean => {
    const apex = apexRef.current;
    const sub = subRef.current?.getBoundingClientRect();
    if (!apex || !sub) return false;
    const opensRight = sub.left >= apex.x - 4;
    const edgeX = opensRight ? sub.left : sub.right;
    const top = { x: edgeX, y: sub.top - 4 };
    const bottom = { x: edgeX, y: sub.bottom + 4 };
    // Nudge the apex back by a few px so tiny jitter at the start still counts as "heading over".
    const a = { x: apex.x + (opensRight ? -4 : 4), y: apex.y };
    return pointInTriangle(p, a, top, bottom);
  }, []);

  const commitHover = useCallback(
    (item: MenuItem) => {
      if (openTimer.current !== null) window.clearTimeout(openTimer.current);
      openTimer.current = window.setTimeout(() => {
        openTimer.current = null;
        apexRef.current = null;
        pendingRef.current = null;
        setOpenKey(item.submenu && !item.disabled ? item.key : null);
      }, SUBMENU_OPEN_DELAY);
    },
    [],
  );

  const handleRowEnter = useCallback(
    (item: MenuItem) => {
      if (closeTimer.current !== null) {
        window.clearTimeout(closeTimer.current);
        closeTimer.current = null;
      }
      if (item.key === openKey) {
        pendingRef.current = null;
        apexRef.current = null;
        if (openTimer.current !== null) window.clearTimeout(openTimer.current);
        return;
      }
      if (openKey && inSafeTriangle(pointerRef.current)) {
        // Heading toward the open submenu: hold it, remember where we'd go otherwise.
        pendingRef.current = item;
        if (holdTimer.current !== null) window.clearTimeout(holdTimer.current);
        holdTimer.current = window.setTimeout(() => {
          holdTimer.current = null;
          const pending = pendingRef.current;
          if (pending) commitHover(pending);
        }, TRIANGLE_HOLD_MAX);
        return;
      }
      commitHover(item);
    },
    [commitHover, inSafeTriangle, openKey],
  );

  const handleRootMove = useCallback(
    (e: React.MouseEvent) => {
      pointerRef.current = { x: e.clientX, y: e.clientY };
      if (pendingRef.current && !inSafeTriangle(pointerRef.current)) {
        const pending = pendingRef.current;
        if (holdTimer.current !== null) {
          window.clearTimeout(holdTimer.current);
          holdTimer.current = null;
        }
        commitHover(pending);
      }
    },
    [commitHover, inSafeTriangle],
  );

  const scheduleSubClose = useCallback(() => {
    if (!openKey) return;
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => {
      closeTimer.current = null;
      if (subRef.current?.matches(":hover")) return;
      setOpenKey(null);
      apexRef.current = null;
      pendingRef.current = null;
    }, SUBMENU_CLOSE_DELAY);
  }, [openKey]);

  const moveFocus = (list: HTMLButtonElement[], current: HTMLElement, delta: number) => {
    const enabled = list.filter((el) => !el.disabled);
    const idx = enabled.indexOf(current as HTMLButtonElement);
    const next = enabled[(idx + delta + enabled.length) % enabled.length];
    next?.focus();
  };

  if (!open) return null;

  const openItem = items.find((i) => i.key === openKey);
  const rootRows = items
    .map((i) => rowRefs.current.get(i.key))
    .filter((el): el is HTMLButtonElement => Boolean(el));

  return createPortal(
    <>
      <div
        ref={rootRef}
        role="menu"
        className="fade-enter fixed z-[70] rounded-panel bg-paper p-2 shadow-[0_8px_32px_rgba(24,24,24,0.12)]"
        style={{ ...rootPos, width }}
        onMouseMove={handleRootMove}
        onMouseLeave={scheduleSubClose}
      >
        {header && <div className="px-2 pb-2 pt-1">{header}</div>}
        {items.map((item) => (
          <MenuRow
            key={item.key}
            item={item}
            active={openKey === item.key}
            rowRef={(el) => {
              if (el) rowRefs.current.set(item.key, el);
              else rowRefs.current.delete(item.key);
            }}
            onEnter={(e) => {
              pointerRef.current = { x: e.clientX, y: e.clientY };
              handleRowEnter(item);
            }}
            onLeave={(e) => {
              if (item.key === openKey) apexRef.current = { x: e.clientX, y: e.clientY };
            }}
            onActivate={() => {
              if (item.submenu) {
                clearTimers();
                setOpenKey(item.key);
                return;
              }
              item.onSelect?.();
              onClose();
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                moveFocus(rootRows, e.currentTarget, 1);
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                moveFocus(rootRows, e.currentTarget, -1);
              } else if (e.key === "ArrowRight" && item.submenu) {
                e.preventDefault();
                clearTimers();
                setOpenKey(item.key);
                setFocusSubFirst(true);
              }
            }}
          />
        ))}
      </div>

      {openItem?.submenu && (
        <div
          ref={subRef}
          role="menu"
          className="fade-enter fixed z-[71] max-h-[60vh] overflow-y-auto rounded-panel bg-paper p-2 shadow-[0_8px_32px_rgba(24,24,24,0.12)]"
          style={{ ...subPos, width }}
          onMouseEnter={() => {
            apexRef.current = null;
            pendingRef.current = null;
            clearTimers();
          }}
          onMouseLeave={scheduleSubClose}
        >
          {openItem.submenu.map((sub, idx) => (
            <MenuRow
              key={sub.key}
              item={sub}
              active={false}
              rowRef={(el) => {
                if (el) subRowRefs.current[idx] = el;
              }}
              onEnter={() => undefined}
              onActivate={() => {
                sub.onSelect?.();
                onClose();
              }}
              onKeyDown={(e) => {
                const list = subRowRefs.current.filter(Boolean);
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  moveFocus(list, e.currentTarget, 1);
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  moveFocus(list, e.currentTarget, -1);
                } else if (e.key === "ArrowLeft") {
                  e.preventDefault();
                  const owner = rowRefs.current.get(openItem.key);
                  setOpenKey(null);
                  owner?.focus();
                }
              }}
            />
          ))}
        </div>
      )}
    </>,
    document.body,
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Thumbnail — photo tile framed in its verdict color; settled frames go gray
 * ──────────────────────────────────────────────────────────────────────────── */

export function Thumb({
  src,
  alt = "",
  verdict,
  pop = false,
  veiled = false,
  className = "",
  imgClassName = "",
  children,
}: {
  src: string;
  alt?: string;
  verdict?: Verdict | null;
  pop?: boolean;
  veiled?: boolean;
  className?: string;
  imgClassName?: string;
  children?: ReactNode;
}) {
  const v = verdict ? VERDICT_STYLE[verdict] : null;
  return (
    <div
      className={`relative overflow-hidden rounded-chip border-2 bg-table bg-clip-padding transition-[border-color] duration-200 ${
        pop ? "verdict-pop" : ""
      } ${className}`}
      style={{
        borderColor: v ? v.color : "transparent",
        borderStyle: v?.dashed ? "dashed" : "solid",
      }}
    >
      <img
        src={src}
        alt={alt}
        loading="lazy"
        draggable={false}
        className={`h-full w-full rounded-[5px] object-cover transition-[filter,opacity] duration-300 ${
          veiled ? "opacity-60 grayscale" : ""
        } ${imgClassName}`}
      />
      {children}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Empty / loading states
 * ──────────────────────────────────────────────────────────────────────────── */

export function EmptyState({
  title,
  children,
  className = "",
}: {
  title: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex flex-col items-center gap-6 py-24 text-center ${className}`}>
      <p className="text-base font-medium text-ink">{title}</p>
      {children && <div className="flex flex-wrap items-center justify-center gap-2">{children}</div>}
    </div>
  );
}

export function Loading({ label, className = "" }: { label: string; className?: string }) {
  return (
    <div className={`flex items-center justify-center gap-4 py-24 text-sm text-ink-2 ${className}`}>
      <ProgressLine indeterminate className="w-24" />
      <span className="sr-only">{label}</span>
    </div>
  );
}

/** Track ids whose verdict just changed so only those marks animate. */
export function useJustMarked(): [string | null, (id: string | null) => void] {
  const [id, setId] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const set = useCallback((next: string | null) => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    setId(next);
    if (next) {
      timer.current = window.setTimeout(() => setId(null), 1200);
    }
  }, []);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  return [id, set];
}
