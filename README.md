
# Mining WhatsApp Course Groups for Sentiment Patterns

A desktop WhatsApp data collector (Electron + `@open-wa/wa-automate`) and an LLM-based sentiment pipeline (Python + OpenAI) used to analyze **>23,000 messages from 28 WhatsApp groups across two faculties at the Hebrew University of Jerusalem**, covering **~7,061 participants**. We test hypotheses about group culture (Psychology vs CS) and mood trends across a semester, and report results with effect sizes, cross-model checks, and weekly timelines.

---

## Highlights

* **Custom WhatsApp Collector**: Electron desktop app that authenticates, lists groups, and exports **rich JSON** (messages, participants, replies, reactions, metadata). Includes participant de-duplication and reply/reaction linking beyond basic exports.
* **LLM Sentiment at Scale**: Batched annotation over messages (Hebrew/English) with polarity, primary emotion, stress/uncertainty, helpfulness, gratitude/toxicity, evidence terms, plus post-processing smoothing.
* **Evaluation & Findings**

  * **Dataset**: **28 WhatsApp course groups** (two faculties, Hebrew University of Jerusalem), **23,619 messages**, **7,061 participants**.
  * *Psychology groups vs CS groups* on positivity, helpfulness, gratitude, message richness, and emoji warmth (large practical differences).
  * *No clear semester-long decline* in mood: weekly heatmaps/timelines show broadly stable emotional patterns.



---

## Tech Stack

* **Desktop**: Electron + `@open-wa/wa-automate`
* **Languages**: JavaScript (crawler/UI), Python (analysis)
* **LLM**: OpenAI chat models (batched)
* **Data & Viz**: pandas, numpy, matplotlib, seaborn

---


### 1) Desktop Collector (Electron)
* Authenticates to WhatsApp Web
* Lists all groups (with member counts)
* Lets you select groups to export
* Exports **enriched** JSON with replies/reactions & de-duplicated participants

Output Example:
<img width="1242" height="711" alt="image" src="https://github.com/user-attachments/assets/a470f7ed-5aed-4e99-98a9-d95f71f86fcf" />

Prefer not to run from source? Grab a packaged build from Releases


---

### 2) Sentiment Pipeline (Python)
* Load exported WhatsApp JSON.
* Send messages in batches of 10 to an LLM with a strict JSON schema.
* Apply deterministic post-processing (e.g., smoothing, reaction-aware nudges).

Output Example:
<img width="695" height="703" alt="image" src="https://github.com/user-attachments/assets/7d02cf29-dfb7-4e7c-913e-3d8283969547" />

---

## Key Results (Psychology vs CS)

| Metric                            | Psych |    CS | Psych Advantage | Direction        |
| --------------------------------- | ----: | ----: | --------------: | ---------------- |
| **Polarity (Positive Sentiment)** | 0.109 | 0.062 |      **+75.5%** | Higher is better |
| **Stress Levels**                 | 0.178 | 0.163 |       **+9.5%** | Lower is better  |
| **Helpfulness**                   | 0.214 | 0.112 |      **+92.2%** | Higher is better |
| **Help Request Rate**             | 0.226 | 0.171 |      **+31.8%** | Higher is better |
| **Gratitude Expression**          | 0.132 | 0.093 |      **+41.6%** | Higher is better |
| **Message Length (chars)**        | 76.01 | 60.27 |      **+26.1%** | Higher is better |
| **Message Emojis**                | 18.07 |  7.99 |     **+126.2%** | Higher is better |
| **Affectionate Emojis**           |  6.57 |  2.98 |     **+120.5%** | Higher is better |

<img width="5971" height="2964" alt="PSY_vs_CS_viz" src="https://github.com/user-attachments/assets/31aa477d-da90-4653-9755-0726be017d3e" />

![weeklyEmotionAndStress](https://github.com/user-attachments/assets/18708c41-f6c0-406a-bdc4-0425146a6763)

---

## Validation & Reliability

### Manual spot-checks:
Random samples were reviewed for polarity, primary emotion, stress/uncertainty, gratitude/helpfulness, and evidence terms. Findings (e.g., sarcasm, emoji-only posts) informed prompt and post-processing tweaks.


<img width="1393" height="468" alt="image" src="https://github.com/user-attachments/assets/ddd0d79c-6834-4091-8d5d-96d8c1325f41" />
### Cross-model check:
We annotated the same mini dataset with GPT-4o-mini and GPT-5. The distributions and overall trends matched. There were small, explainable gaps-for example, slightly different intensity for humor and anger-but these did not change the headline results.

<img width="906" height="304" alt="image" src="https://github.com/user-attachments/assets/809a05cb-4c55-40f2-b0e6-f12fac920a02" />


---

##  Repository Structure

```
.
├─ whatsapp-crawler-electron/        # Desktop data collector (Electron)
│  ├─ backend/
│  │  ├─ crawl-service.js            # Group discovery, message export, progress
│  │  ├─ enrichment.js               # Replies, reactions, sender resolution
│  │  ├─ participants.js             # Identity merging & participant info
│  │  ├─ exporter.js
│  │  ├─ messageUtils.js
│  │  └─ auth-process.js
│  ├─ renderer/                      # UI (HTML/JS/CSS)
│  ├─ main.js, preload.js
│  ├─ package.json
│  ├─ start.sh, build.sh
│
├─ setiment_analysis_LLM/            # LLM sentiment pipeline (name kept as-is)
│  ├─ sentiment_analysis_pipeline.py # Batched OpenAI annotation + post-pass
│  └─ SentimentAnalysis_output/      # Per-category JSON outputs
│
├─ WhatsappData/                     # Raw WhatsApp JSON exports (by category)
│  ├─ CS/
│  ├─ Psychology & Biology/
│  └─ General Courses and Groups/
│
└─ Results/                          # Plots, comparisons, heatmaps
   ├─ PSY_vs_CS_raw_data.csv
   ├─ PSY_vs_CS_viz.png
   ├─ Compare_ai_models/
   │  ├─ Evaluation/
   │  │  └─ mini_db_demo.sentiment_4o_mini.json
   │  └─ plots/
   │     └─ model_comparison_overview.png
   └─ heatmap/
      ├─ WeeklyEmotion_Semester_Timeline.jpg
      ├─ Weekly_sentiment_analysis.jpg
      └─ weeklyEmotionAndStress.jpg
```

---

##  Project team:

- **Yehonatan Ezra**
- **Natanel Richey**
- **Jonatan Vider**




See the project summary (PDF) for additional details.

---



