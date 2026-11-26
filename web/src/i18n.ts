import { createContext, useContext } from "react";

export type Lang = "zh-CN" | "en-US";

const MESSAGES = {
  "zh-CN": {
    appName: "Lightbox",
    navSessions: "场次",
    navRestore: "废片箱",
    authPromptTitle: "进入本地选片工作区",
    authPromptDesc: "此服务仅在本机 127.0.0.1 运行。点击下方按钮建立本次浏览器会话。",
    authPromptAction: "确认进入",
    sessionsTitle: "拍摄场次",
    sessionsSubtitle: "选择本地照片目录开始初筛，或先运行示例相册体验流程。",
    pickFolder: "选择拍摄目录",
    scanPathPlaceholder: "或粘贴本地目录路径，例如 /Users/.../Photos",
    startScan: "开始扫描",
    runDemo: "运行内置示例",
    scanning: "正在扫描",
    pauseScan: "暂停",
    resumeScan: "继续",
    emptySessions: "还没有扫描记录",
    emptySessionsHint: "点击上方「选择拍摄目录」或「运行内置示例」开始第一场选片。",
    openReview: "进入选片",
    openShootFolder: "打开文件夹",
    autoScreen: "自动筛废片",
    autoScreening: "正在分析...",
    autoScreenResult: (name: string, red: number, yellow: number, total: number) => {
      if (red === 0 && yellow === 0) {
        return `「${name}」分析完成：共 ${total} 张，未发现明显拍糊或误拍废片。`;
      }
      const parts: string[] = [];
      if (red > 0) parts.push(`${red} 张明显废片（已放入待删除）`);
      if (yellow > 0) parts.push(`${yellow} 张存疑照片`);
      return `「${name}」分析完成：自动找出 ${parts.join("、")}。`;
    },
    resetReview: "重置选择",
    resetReviewConfirm:
      "确定重置当前目录内所有照片的选择标记吗？（已移入「废片箱」的照片保持不变，如需找回请点顶部「废片箱」一键放回原处。）",
    resetReviewDone: (name: string) =>
      `已重置「${name}」的选择记录，可点击「自动筛废片」或「进入选片」重新开始。`,
    oneClickQuarantine: "移入废片箱",
    oneClickWithCount: (n: number) => `移入废片箱（${n} 张）`,
    bucketAll: "待选",
    bucketAllPhotos: "全部",
    bucketRed: "待删除",
    bucketYellow: "存疑",
    bucketBursts: "连拍选帧",
    bucketExempt: "已保护",
    bucketNormal: "已保留",
    backToSessions: "返回场次",
    photoCount: (n: number) => `${n} 张`,
    sessionStatusDone: "已选完",
    sessionStatusUnsure: (n: number) => `已看完 · ${n} 张存疑待定`,
    sessionStatusReadyTrash: (n: number) => `已选完 · 待移出 ${n} 张废片`,
    sessionStatusInProgress: (unreviewed: number) => `还剩 ${unreviewed} 张未看`,
    sessionProgressRatio: (done: number, total: number) => `进度 ${done} / ${total}`,
    sessionKeptCount: (n: number) => `保留 ${n} 张`,
    sessionTotalScanned: (n: number) => `共 ${n} 张`,
    progressAllDone: "本轮已全部看完",
    progressDoneWithTrash: (rejects: number) => `已全部看完 · 还有 ${rejects} 张废片待移出`,
    progressDoneWithUnsure: (unsure: number) => `已全部看完 · 还有 ${unsure} 张存疑可复看`,
    progressSummary: (reviewed: number, unreviewed: number, rejects: number, unsure: number) => {
      const parts = [`已看 ${reviewed} 张`, `还剩 ${unreviewed} 张`];
      if (unsure > 0) parts.push(`存疑 ${unsure} 张`);
      if (rejects > 0) parts.push(`待删 ${rejects} 张`);
      return parts.join(" · ");
    },
    alreadyQuarantinedLink: (n: number) => `已移入废片箱 ${n} 张 →`,
    hideRejectedToggle: "淘汰后直接移出列表",
    rejectedToast: (n: number) => `已放入待删除（共 ${n} 张）`,
    skippedToast: (seq: string) => `已跳过 ${seq}，移到本轮末尾稍后再选`,
    reviewCompleteTitle: "这一轮全部选完了",
    reviewEndOfListTitle: "已浏览到本轮末尾",
    reviewCompleteDesc: (kept: number, unsure: number, rejects: number) => {
      const parts = [`保留 ${kept} 张`];
      if (unsure > 0) parts.push(`存疑 ${unsure} 张`);
      if (rejects > 0) parts.push(`待移出废片 ${rejects} 张`);
      return `${parts.join("，")}。`;
    },
    reviewRemainingDesc: (unreviewed: number, rejects: number) =>
      rejects > 0
        ? `还有 ${unreviewed} 张尚未决定（含跳过的照片），另有 ${rejects} 张待移出。`
        : `还有 ${unreviewed} 张照片尚未决定，可以继续查看。`,
    reviewCompleteConfirm: (n: number) => `将 ${n} 张废片移入待删除文件夹`,
    reviewCompleteUnsure: (n: number) => `去复看 ${n} 张存疑照片`,
    reviewCompleteClean: (kept: number) => `已确认保留 ${kept} 张，没有需要移出的废片。`,
    reviewCompleteStay: "继续浏览",
    reviewViewAll: "查看全部照片",
    reviewGoUnreviewed: (n: number) => `去看待选照片（${n} 张）`,
    lowResHint: "当前为低清内嵌预览（< 2000px），1:1 放大仅供参考",
    zoomOn: "1:1 放大中（Space / Z 返回）",
    zoomOff: "1:1 智能放大（Space / Z）",
    actionKeep: "保留",
    actionUnsure: "存疑",
    actionSkip: "跳过",
    actionReject: "淘汰",
    actionUndo: "撤销",
    stripKept: "✓ 保留",
    stripUnsure: "? 存疑",
    stripReject: "✕ 待删废片",
    stripProtected: "保护",
    stripSkipped: "跳过",
    rescueReasonPrompt: "已放回保留，可选记一下原因：",
    rescueReasons: [
      { tag: "actually_sharp", label: "主体其实清晰" },
      { tag: "intentional_blur", label: "刻意动感或慢门" },
      { tag: "good_moment", label: "表情或氛围很好" },
      { tag: "other", label: "其他" },
    ],
    sourceLabels: {
      engine: "自动初筛",
      user: "人工已选",
      rule: "偏好规则",
      vlm: "视觉复核",
      agent: "助手建议",
    } as Record<string, string>,
    noPhotosInBucket: "当前列表没有照片",
    quarantinedBadge: "已移入废片箱",
    protectedBadge: "机内保护",
    burstBanner: "连拍选帧模式：按 1 / 2 / 3 / 4 选择首选，按 → 确认并进入下一组（未确认的连拍默认保留）",
    burstNoGroups: "当前场次没有检测到高速连拍组",
    burstGroupLabel: (idx: number, total: number) => `第 ${idx} / ${total} 组连拍`,
    burstConfirmed: "已确认",
    burstUnconfirmed: "待确认",
    burstRecommended: "推荐首选",
    burstSelected: "当前选中",
    burstConfirmNext: "确认首选并下一组（→）",
    burstSyncZoom: "同步放大（Space）",
    modalTitle: "把废片移出原文件夹",
    modalSummary: (total: number) =>
      `即将把这 ${total} 张废片统一移入本地「待删除文件夹」，原相册立刻清爽。`,
    modalTechCount: (n: number) => `明显拍糊 / 误拍废片：${n} 张`,
    modalBurstCount: (n: number) => `连拍未选中的照片：${n} 张`,
    modalManualCount: (n: number) => `手动标记淘汰：${n} 张`,
    modalHighRatioWarn: (pct: string) =>
      `注意：本次移出占比达 ${pct}（超过 40%），建议先扫一眼下方缩略图确认无误。`,
    modalTargetDir: "待删除文件夹位置",
    modalSafetyPromise:
      "软件不会直接彻底抹掉原片。确认不需要后，点「打开文件夹」就能一键清空；如果反悔，随时能在「废片箱」放回原处。",
    modalCancel: "先取消",
    modalConfirm: "移入待删除文件夹",
    modalConfirmWithCount: (n: number) => `移入待删除文件夹（${n} 张）`,
    modalReturnToReview: (n: number) => `放回待选（${n} 张）`,
    modalReturnSuccess: (n: number) => `已将 ${n} 张照片取消待删标记，放回「待选」列表。`,
    modalSelectAll: "全选",
    modalDeselectAll: "取消全选",
    modalSelectedCount: (selected: number, total: number) => `已勾选 ${selected} / ${total} 张`,
    modalUncheckedStayHint: (n: number) => `未勾选的 ${n} 张暂留在原地，并在「待选」中标红提示`,
    modalReasonFallback: "标记为待删除",
    modalApplying: "正在移出...",
    modalEmpty: "当前没有标记为待删除的照片。",
    modalSuccess: (n: number) => `已将 ${n} 张废片移到待删除文件夹，原相册已清爽。`,
    openTrashFolder: "打开文件夹",
    goTrashBin: "去废片箱",
    insightsTitle: "废片箱（待删除文件夹）",
    insightsSubtitle:
      "从原相册移出的废片都在这里。确认不要了可以点「打开文件夹」直接删除清空；误移了随时点「放回原处」恢复并保留。",
    insightsEmpty: "废片箱是空的",
    insightsEmptyHint: "在选片页面把废片移入待删除文件夹后，会集中显示在这里。",
    planStatus: {
      draft: "草稿",
      reviewed: "待确认",
      approved: "待移出",
      applying: "移动中",
      applied: "在废片箱中",
      reverted: "已放回原处",
      stale: "已过期",
    } as Record<string, string>,
    restoreAll: "全部放回原处",
    restoreSingle: "放回原处",
    restoredBadge: "已放回保留",
    restoreSuccessBatch: (n: number) => `已将 ${n} 张照片放回原文件夹，并自动标记为保留。`,
    restoreSuccessSingle: "已放回原文件夹，并自动标记为保留。",
    fileCount: (n: number) => `${n} 个关联文件`,
    showDetails: "展开照片",
    hideDetails: "收起照片",
    loading: "加载中...",
    errorPrefix: "操作未完成：",
  },
  "en-US": {
    appName: "Lightbox",
    navSessions: "Sessions",
    navRestore: "Trash folder",
    authPromptTitle: "Enter local review workspace",
    authPromptDesc:
      "This server runs strictly on 127.0.0.1. Confirm to open your local browser session.",
    authPromptAction: "Continue",
    sessionsTitle: "Shoot sessions",
    sessionsSubtitle: "Choose a local photo folder to scan, or run the built-in sample shoot first.",
    pickFolder: "Choose photo folder",
    scanPathPlaceholder: "Or paste a local folder path, e.g. /Users/.../Photos",
    startScan: "Scan folder",
    runDemo: "Run sample shoot",
    scanning: "Scanning",
    pauseScan: "Pause",
    resumeScan: "Resume",
    emptySessions: "No sessions yet",
    emptySessionsHint: "Choose a local photo folder or run the sample shoot above to begin.",
    openReview: "Review",
    openShootFolder: "Open folder",
    autoScreen: "Auto-find rejects",
    autoScreening: "Analyzing...",
    autoScreenResult: (name: string, red: number, yellow: number, total: number) => {
      if (red === 0 && yellow === 0) {
        return `"${name}" analyzed (${total} photos): no obvious blurry or mis-shot rejects found.`;
      }
      const parts: string[] = [];
      if (red > 0) parts.push(`${red} obvious rejects`);
      if (yellow > 0) parts.push(`${yellow} unsure`);
      return `"${name}" analyzed: flagged ${parts.join(" and ")}.`;
    },
    resetReview: "Reset selections",
    resetReviewConfirm:
      "Reset selection marks for all photos currently in this folder? (Files already moved to the Trash folder stay untouched — restore them anytime from Trash folder.)",
    resetReviewDone: (name: string) =>
      `Selections reset for "${name}". You can run Auto-find rejects or enter Review.`,
    oneClickQuarantine: "Move to trash folder",
    oneClickWithCount: (n: number) => `Move to trash folder (${n})`,
    bucketAll: "To review",
    bucketAllPhotos: "All",
    bucketRed: "To remove",
    bucketYellow: "Unsure",
    bucketBursts: "Bursts",
    bucketExempt: "Protected",
    bucketNormal: "Kept",
    backToSessions: "Sessions",
    photoCount: (n: number) => `${n} photos`,
    sessionStatusDone: "Completed",
    sessionStatusUnsure: (n: number) => `Reviewed · ${n} unsure`,
    sessionStatusReadyTrash: (n: number) => `Reviewed · ${n} ready to remove`,
    sessionStatusInProgress: (unreviewed: number) => `${unreviewed} left to review`,
    sessionProgressRatio: (done: number, total: number) => `${done} / ${total} done`,
    sessionKeptCount: (n: number) => `${n} kept`,
    sessionTotalScanned: (n: number) => `${n} total`,
    progressAllDone: "Round complete — all photos reviewed",
    progressDoneWithTrash: (rejects: number) =>
      `All reviewed · ${rejects} rejects ready to move out`,
    progressDoneWithUnsure: (unsure: number) =>
      `All reviewed · ${unsure} unsure photos to revisit`,
    progressSummary: (reviewed: number, unreviewed: number, rejects: number, unsure: number) => {
      const parts = [`Reviewed ${reviewed}`, `${unreviewed} left`];
      if (unsure > 0) parts.push(`${unsure} unsure`);
      if (rejects > 0) parts.push(`${rejects} to remove`);
      return parts.join(" · ");
    },
    alreadyQuarantinedLink: (n: number) => `${n} in trash folder →`,
    hideRejectedToggle: "Hide after reject",
    rejectedToast: (n: number) => `Queued for trash folder (${n} total)`,
    skippedToast: (seq: string) => `Skipped ${seq} — moved to the end of this round`,
    reviewCompleteTitle: "Round complete",
    reviewEndOfListTitle: "Reached end of round",
    reviewCompleteDesc: (kept: number, unsure: number, rejects: number) => {
      const parts = [`${kept} kept`];
      if (unsure > 0) parts.push(`${unsure} unsure`);
      if (rejects > 0) parts.push(`${rejects} ready to remove`);
      return `${parts.join(", ")}.`;
    },
    reviewRemainingDesc: (unreviewed: number, rejects: number) =>
      rejects > 0
        ? `${unreviewed} photos still undecided, and ${rejects} ready to remove.`
        : `${unreviewed} photos still undecided.`,
    reviewCompleteConfirm: (n: number) => `Move ${n} rejects to trash folder`,
    reviewCompleteUnsure: (n: number) => `Revisit ${n} unsure photos`,
    reviewCompleteClean: (kept: number) => `${kept} photos kept. Nothing left to remove.`,
    reviewCompleteStay: "Keep browsing",
    reviewViewAll: "Browse all photos",
    reviewGoUnreviewed: (n: number) => `Review remaining (${n})`,
    lowResHint: "Low-resolution embedded preview (< 2000px); 1:1 zoom is approximate",
    zoomOn: "1:1 zoomed (Space / Z to exit)",
    zoomOff: "1:1 smart zoom (Space / Z)",
    actionKeep: "Keep",
    actionUnsure: "Unsure",
    actionSkip: "Skip",
    actionReject: "Reject",
    actionUndo: "Undo",
    stripKept: "✓ Kept",
    stripUnsure: "? Unsure",
    stripReject: "✕ Trash",
    stripProtected: "Locked",
    stripSkipped: "Skip",
    rescueReasonPrompt: "Kept. Optional note:",
    rescueReasons: [
      { tag: "actually_sharp", label: "Subject is sharp" },
      { tag: "intentional_blur", label: "Intentional motion / slow shutter" },
      { tag: "good_moment", label: "Great expression or mood" },
      { tag: "other", label: "Other" },
    ],
    sourceLabels: {
      engine: "Auto-screen",
      user: "Reviewed",
      rule: "Rule",
      vlm: "VLM review",
      agent: "Assistant",
    } as Record<string, string>,
    noPhotosInBucket: "No photos in this view",
    quarantinedBadge: "In trash folder",
    protectedBadge: "Protected",
    burstBanner:
      "Burst pick mode: press 1 / 2 / 3 / 4 to choose pick, press → to confirm and advance (unconfirmed bursts stay safe)",
    burstNoGroups: "No rapid burst groups detected in this session",
    burstGroupLabel: (idx: number, total: number) => `Burst ${idx} of ${total}`,
    burstConfirmed: "Confirmed",
    burstUnconfirmed: "Unconfirmed",
    burstRecommended: "Recommended",
    burstSelected: "Selected",
    burstConfirmNext: "Confirm pick & next (→)",
    burstSyncZoom: "Sync zoom (Space)",
    modalTitle: "Move rejects out of shoot folder",
    modalSummary: (total: number) =>
      `${total} rejected photos will be moved into a local trash folder so your shoot stays clean.`,
    modalTechCount: (n: number) => `Blurry / mis-shot rejects: ${n}`,
    modalBurstCount: (n: number) => `Confirmed burst non-picks: ${n}`,
    modalManualCount: (n: number) => `Manually rejected: ${n}`,
    modalHighRatioWarn: (pct: string) =>
      `Note: ${pct} of this session is marked for removal (> 40%). Please glance at the thumbnails below.`,
    modalTargetDir: "Trash folder path",
    modalSafetyPromise:
      "Lightbox never permanently deletes originals on its own. Open the folder to empty it whenever you're ready, or put any photo back in one click.",
    modalCancel: "Cancel",
    modalConfirm: "Move to trash folder",
    modalConfirmWithCount: (n: number) => `Move to trash folder (${n})`,
    modalReturnToReview: (n: number) => `Return to review (${n})`,
    modalReturnSuccess: (n: number) => `Unflagged ${n} photos and returned them to "To review".`,
    modalSelectAll: "Select all",
    modalDeselectAll: "Clear all",
    modalSelectedCount: (selected: number, total: number) => `${selected} of ${total} selected`,
    modalUncheckedStayHint: (n: number) =>
      `${n} unchecked stay flagged in red inside "To review"`,
    modalReasonFallback: "Marked to remove",
    modalApplying: "Moving...",
    modalEmpty: "No photos are currently marked to remove.",
    modalSuccess: (n: number) => `Moved ${n} rejects to the trash folder.`,
    openTrashFolder: "Open folder",
    goTrashBin: "Open trash view",
    insightsTitle: "Trash folder",
    insightsSubtitle:
      "Rejects moved out of your shoot folders live here. Open the folder to delete them permanently, or put any photo back in one click.",
    insightsEmpty: "Trash folder is empty",
    insightsEmptyHint: "Rejected photos you move out while reviewing will appear here.",
    planStatus: {
      draft: "Draft",
      reviewed: "Pending",
      approved: "Ready",
      applying: "Moving",
      applied: "In trash folder",
      reverted: "Put back",
      stale: "Superseded",
    } as Record<string, string>,
    restoreAll: "Put all back",
    restoreSingle: "Put back",
    restoredBadge: "Put back",
    restoreSuccessBatch: (n: number) => `Put ${n} photos back in their original folder and marked as kept.`,
    restoreSuccessSingle: "Put back in original folder and marked as kept.",
    fileCount: (n: number) => (n === 1 ? "1 file" : `${n} files`),
    showDetails: "Show photos",
    hideDetails: "Hide photos",
    loading: "Loading...",
    errorPrefix: "Could not complete action: ",
  },
};

export type Messages = (typeof MESSAGES)["zh-CN"];

export interface I18nContextValue {
  lang: Lang;
  setLang: (next: Lang) => void;
  t: Messages;
}

export const I18nContext = createContext<I18nContextValue>({
  lang: "zh-CN",
  setLang: () => undefined,
  t: MESSAGES["zh-CN"],
});

export function getMessages(lang: Lang): Messages {
  return MESSAGES[lang];
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext);
}
