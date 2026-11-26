# Lightbox Photo Culling Rubric (`culling_rubric.md`)

Use this rubric directly when evaluating uploaded photos in chat (**Mode A — Zero-Install**) and when reviewing crops or burst strips from local sessions (**Mode B — LightBox MCP**).

---

## 1. Core Safety Philosophy

1. **Zero false-discard bias**: Never lump an ambiguous photo into obvious rejects. Any borderline shot goes into **拿不准的照片 (Uncertain / Borderline)** with a concrete note on what to verify.
2. **Respect creative intent**: Unusual blur, darkness, or brightness can be deliberate photography rather than mistakes.
3. **Be honest about visibility**: Chat uploads are often downscaled. If fine eye sharpness or micro-motion blur cannot be verified with certainty, explicitly state that uncertainty instead of guessing.
4. **Match the user's actual question**: Lead with what the user asked for (e.g., put **明显废片 / 模糊照片** first and mandatory when the user asks to find blurry photos or clean up rejects; put **推荐选片** first when the user asks to pick N favorites).
5. **No lecturing**: State observable visual facts and selection tradeoffs objectively; never preach camera tutorials unless the user explicitly asks how to improve technique.

---

## 2. Six Evaluation Dimensions

Evaluate photos across these dimensions, weighted by the user's stated goal:

| Dimension | What Wins (Keep / Recommend) | What Loses (Flag as Blurry / Reject) |
| :--- | :--- | :--- |
| **1. Subject Focus & Sharpness** | Subject's eyes (for portraits/animals) or primary focal anchor are crisp; intentional shallow depth of field keeps the subject sharp while background melts smoothly. | Accidental camera shake across the entire frame, severe motion smear, or missed focus where the intended subject is clearly soft with no creative intent. |
| **2. Expression & Peak Moment** | Natural expression, both eyes open (or intentional closed-eye mood), flattering posture, decisive action peak. | Mid-blink, awkward mid-sentence mouth, subject unexpectedly turning away, accidental half-step clutter. |
| **3. Composition & Clean Edges** | Clear subject separation, intentional framing, level horizon (unless deliberate tilt), clean edges. | Unintentionally chopped chin/forehead/feet, severe unwanted intrusion blocking the subject, accidental ground/pocket misfire. |
| **4. Light, Exposure & Color** | Pleasant light on the subject, recoverable detail in key highlights/shadows, natural skin tones or cohesive mood. | Extreme pitch-black or blown-white misfire with virtually no usable visual information. |
| **5. Variety & Burst Deduplication** | When picking N photos, prefer complementary angles, expressions, or moments rather than near-identical burst frames. | Near-duplicate runner-up from the same burst (note clearly as a *burst duplicate*, NOT a technical blur failure). |
| **6. Fit for User's Request** | Directly matches the user's requested task, count, vibe, or target use. | Technically fine frame that misses the user's stated criteria. |

---

## 3. Creative Intent Protection Rules

Do **NOT** classify any of the following as an obvious reject / blurry waste; keep them or place them in **拿不准的照片 (Uncertain)** if ambiguous:

- **Shallow depth of field / macro / foreground bokeh**: Soft background or foreground is normal when the intended subject plane is sharp.
- **Panning (追焦摇拍)**: Subject is reasonably crisp while the background shows directional motion streaks.
- **Long exposure (长曝光)**: Silky water, light trails, or blurred moving crowds against a steady frame.
- **Intentional low-key / silhouette / high-key snow**: Strong dark or bright areas forming a deliberate graphic composition.
- **Intentional Camera Movement (ICM / 创意晃拍)**: Soft painterly streaks that appear deliberate across a series.

---

## 4. Intent-Adaptive Output Templates (Mode A — Chat Upload)

Always match the user's language (Chinese or English) and **adapt the section order to the user's intent**.

### Template 1: Finding Blurry / Bad Photos ("帮我把糊的照片挑出来" / "帮我去掉废片" / "Find blurry shots")
> **Mandatory rule:** When the user asks to find blurry photos or remove rejects, **明显废片 / 模糊照片** MUST come **FIRST** and is **REQUIRED** (never optional). If none are clearly blurry/unusable, explicitly say so rather than inventing flaws.

```markdown
### 明显废片 / 模糊照片（建议淘汰，共 2 张）
1. **#1（IMG_4820.JPG）** — 全图明显方向性拖影（手抖模糊），主体面部和边缘均失去清晰度。
2. **#4（IMG_4823.JPG）** — 严重失焦（焦点落在远处背景，前景主体整体发虚）且人物闭眼。

### 拿不准的照片（暂不判废，建议 100% 放大确认）
- **#3（IMG_4822.JPG）** — 缩略图下眼部清晰度略显临界，但整体姿态自然；建议原图放大看眼部是否合焦后再决定。
- **#6（IMG_4825.JPG）** — 有轻微慢门动态感，拿不准是否为刻意创作氛围；已保留不归入明显废片。

### 清晰可用照片（建议保留）
- **#2（IMG_4821.JPG）**、**#5（IMG_4824.JPG）**、**#7（IMG_4826.JPG）** 焦点扎实、画面干净，可放心保留。
```

### Template 2: Picking Standout Photos ("帮我挑 3 张" / "选几张发朋友圈" / "Pick the best 3")
```markdown
### 推荐选片（共 3 张）
1. **#2（IMG_4821.JPG）** — 人物眼神对焦清晰、表情自然放松，光线柔和且背景干净，是这组里最耐看的主图。
2. **#5（IMG_4824.JPG）** — 构图与 #2 形成景别互补，人物动态自然、画面通透，适合放在组图第二张。
3. **#7（IMG_4826.JPG）** — 氛围感最好，高光与暗部过渡舒服，与前两张视角不重复。

### 拿不准的照片（建议人工放大确认）
- **#3（IMG_4822.JPG）** — 表情比 #2 更生动，但聊天缩略图下眼部清晰度略逊于 #2；建议原图 100% 放大核对眼部是否合焦，若清晰可替换 #2。
- **#6（IMG_4825.JPG）** — 画面有轻微拖影，拿不准是否为刻意慢门氛围；已保留不按废片处理。

### 明显废片 / 连拍重复提醒（按需补充）
- **#1（IMG_4820.JPG）**：人物闭眼且明显手抖。
- **#4（IMG_4823.JPG）**：与 #2 为同机位连拍，清晰度和神态略弱于 #2（属连拍重复，非技术糊片）。
```

### Template 3: Comparing Burst / Similar Frames ("这几张连拍留哪张" / "Compare these shots")
```markdown
### 首选推荐
- **#2（IMG_4821.JPG）** — 这组连拍中眼部最锐、双目自然睁开、构图最稳的一帧。

### 拿不准的备选
- **#3（IMG_4822.JPG）** — 笑容略比 #2 放松，但预览图里眼部锐度看起来稍弱，建议原图 100% 对比后再二选一。

### 其余帧对比说明
- **#1（IMG_4820.JPG）**：人物半闭眼。
- **#4（IMG_4823.JPG）**：与 #2 高度重复且主体边缘略裁切。
```

### Identification Fallback Rule
- If the chat client exposes filenames, always show **both** upload order and filename (e.g., `#2（IMG_4821.JPG）`).
- If filenames are hidden by the client, clearly label photos by upload order (`#1`, `#2`, `#3`...) plus a short 3–5 word visual cue (e.g., `#2（穿白衬衫侧身回眸）`) so the user can locate each photo effortlessly.
