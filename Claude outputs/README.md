# Daily Job Hunt Agent — Muhammad Ahmad (Gill)

> **For the Claude agent reading this:** you are running a job search for Muhammad Ahmad. This file has everything you need: his profile, what he wants, where to search, how to score each job against his CV, and the exact Excel file to produce. Do the whole job end to end without asking questions. If something is unclear, make the sensible choice and write it down in the `Run Notes` sheet. The only thing you hand back is a finished `.xlsx` file and a short summary.

---

## 0. TL;DR — what to do on every run

1. Read **Section 1** (candidate profile) and **Section 2** (target roles, locations and **priority companies**).
1b. **Check a browser is available** (Section 3.0). Without one, LinkedIn, Indeed and Glassdoor can't be searched and the run will come back thin.
1c. **Go through every priority company in Section 2.3 first**, one by one, on its careers page and its LinkedIn Jobs tab. Record each one in the `Company Coverage` sheet, even when it has no matching jobs.
2. Then search every source in **Section 3** with the queries in **Section 4**. Only two kinds of job count: **(a) on-site or hybrid jobs in Lahore, Pakistan**, and **(b) fully remote jobs from anywhere in the world** that hire from Pakistan. **No on-site or hybrid jobs anywhere else**, including Karachi, Islamabad, Saudi Arabia, UAE, Europe and everywhere else.
3. Keep only jobs that pass the **relevance filter** in **Section 5**. Throw away anything outside his field.
4. For every job you keep, open the actual posting, pull the details, and score it with the **fit rubric** in **Section 6**.
5. Remove duplicates, and remove jobs already reported on earlier runs (**Section 8**).
6. Build the Excel workbook exactly as **Section 7** describes and save it as `Job_Openings_YYYY-MM-DD.xlsx`.
7. Reply with a 5-line summary: how many jobs were found, the top 5 by fit score, and any jobs that close within 7 days.

**Search window:** jobs posted in the **last 7 days** on job boards and LinkedIn. **Priority companies (Section 2.3) and company careers / ATS pages:** include **every currently open** matching role, whatever its posting date.

**Minimum volume: 40 jobs. Target: 50–100.** Don't stop searching just because the first sources ran dry. If you have fewer than 40 after all sources, follow the **widening steps in Section 5.1** in order until you reach 40 or run out of steps. Record what you did in `Run Notes`.

**Never make up** a job, a company, a link, a salary or a requirement. If a detail can't be verified, write `Not stated`. If the **posting date** is unclear but the page is live and accepting applications, **keep the job** and set `Date Posted` = `Unverified`. Don't drop it.

---

## 1. Candidate Profile (the "CV" to match against)

### 1.1 Identity and links
| Field | Value |
|---|---|
| Name | Muhammad Ahmad (Ahmad Gill) |
| Based in | Lahore, Pakistan (Pakistani national) |
| Email | m.ahmadgill01@gmail.com |
| Portfolio | https://ahmadgill.com |
| LinkedIn | https://linkedin.com/in/muhammad-ahmad-gill-427772262 |
| GitHub | https://github.com/Ahmad-Gill |
| Headline | AI & Data Science Specialist · ML Engineer · MLOps / DevOps · Full-Stack |

### 1.2 Education
| Degree | Institution | Dates | Notes |
|---|---|---|---|
| **MS Artificial Intelligence** | Lahore University of Management Sciences (LUMS) | 2025 – 2027 (in progress, 3rd semester as of Nov 2026) | Graduate-level AI coursework, see 1.5 |
| **BS Data Science** | FAST-NUCES, Lahore | Jun 2021 – Jun 2025 | FYP: CCTV video analytics (YOLO) |

> **Important:** he is a **full-time MS student until mid-2027** and works alongside his studies. He will **not relocate**. He only takes on-site / hybrid work in **Lahore**; everything else must be **100% remote**.

### 1.3 Work experience (~2.5 years total, from May 2024)
| Role | Company | Dates | What he did |
|---|---|---|---|
| **Data Scientist & Cloud Engineer** | GlowingSoft Technologies (Pakistan) | Aug 2025 – Present | Predictive ML models with cloud data pipelines; workflow automation with AI tools; Azure infrastructure with **Terraform (IaC)**; managed Azure storage, compute and **Azure ML** services. |
| **Junior DevOps Engineer** | Cybersoft North America (Pakistan) | Apr 2025 – Aug 2025 | **CI/CD in Azure DevOps**; Terraform provisioning; Python & Bash automation of data/DB pipelines (**−30% processing time**); Azure Cost Management (**−15% cloud cost**). |
| **Junior Data Scientist** | Coding Square Technologies (Pakistan) | May 2024 – May 2025 | Large-dataset preprocessing; **OCR models for government documents (+40% digitization accuracy)**; integrated AI into client systems. |
| **Data Science Intern** | GlowingSoft Technologies (Pakistan) | Jun 2024 – Aug 2024 | Predictive ML models, data analysis, automated reporting. |

**Seniority to target:** Junior, Associate, Mid-level, and "1–3 years" roles. Also include roles asking for "3–4 years" where his skills match very well (flag them as a stretch). Skip Senior/Staff/Principal/Lead roles that need 5+ years, and skip pure internships unless they are paid, AI-focused, and at a top-tier company.

### 1.4 Technical skills (from CV + verified GitHub + LUMS work)

**Languages:** Python (primary), C/C++, Java, JavaScript, SQL, Bash, R (basic)

**ML / DL / AI**
- Classical ML: scikit-learn, regression, SVM, clustering, supervised and unsupervised learning
- Deep Learning: **PyTorch**, TensorFlow, Keras; CNNs, U-Net, Transformers built from scratch
- **Computer Vision:** YOLO (v7), OpenCV, object detection, **semantic segmentation (U-Net)**, OCR, face detection, camera calibration
- **NLP / LLMs / GenAI:** wrote and trained a **Transformer language model from scratch** (tokenizer, training loop, hyperparameter ablations, decoding strategies); long-context extension, attention, positional encodings (Advanced GenAI course); translation, claim classification
- **LLM applications & agents:** **LangGraph**, agentic AI, **MCP (Model Context Protocol)**, RAG with local embeddings (bge-small), vector + keyword (FTS) retrieval, memory systems, guardrails
- **Local / self-hosted LLMs:** **llama.cpp**, GGUF models (Qwen 3.5 / 3.6 MoE), SSE streaming inference
- **Edge AI:** **Vision-Language Models (VLMs) on Raspberry Pi 5**, quantization, latency / memory / energy benchmarking, Metal / CPU inference
- **Audio AI:** deepfake / voice-clone audio detection (DL project), speech / TTS experiments
- Graph & spatio-temporal deep learning (LUMS course)
- **Robotics:** ROS 2 (Python launch files, perception nodes), vision-based heading controller

**MLOps / DevOps / Cloud**
- **Azure** (DevOps, ML, storage, compute, Cost Management), **Oracle Cloud** deployment
- **Terraform**, **Docker**, **CI/CD** (Azure DevOps, **GitHub Actions**), systemd deployments
- Data ingestion, ETL and **stream processing pipelines**, data preprocessing (MLOps assignments)
- Experiment tracking, evaluation harnesses, benchmarking, pytest test suites
- Linux, Bash, Git

**Backend / Full-Stack**
- **FastAPI** (auth, API keys, SQLite, streaming endpoints), Django, Node/Express
- **MERN stack** (MongoDB, Express, React, Node): auth, dashboards, chat, payments
- **Electron desktop apps** (cross-platform Clipboard app, macOS/Windows builds)
- SQLite / SQLCipher, REST API design, Postman
- Web scraping: Selenium, Python scrapers

### 1.5 Portfolio evidence (use to justify fit, e.g. "matches: EdgeLog project")
| Project | What it proves |
|---|---|
| **EdgeLog — event-triggered VLM logging on Raspberry Pi 5** (LUMS AI624) | Edge AI, VLMs, llama.cpp, quantization, benchmarking, CI, team engineering (5-person) |
| **Event-Triggered VLM Logging on Edge** | Qwen 3.5-2B GGUF, ChokePoint dataset, automated setup, AWQ / pruning / LoRA planned |
| **custom-llm-node** | Self-hosted LLM API: FastAPI, llama.cpp, API keys, SSE streaming, GitHub Actions CI/CD to Oracle Cloud |
| **Chotu — personalized AI assistant** | LLM agent architecture: planner, RAG, local embeddings, memory curation, guardrails, job queue, encrypted SQLite |
| **cs5326-pa1 — Transformer LM from scratch** | Deep understanding of LLM training: ablations, LR schedules, decoding |
| **create-your-own-story** | Small transformer story generator (PyTorch, tokenizers) |
| **Real-Time Semantic Segmentation + Vision Heading Controller** | CV + robotics control pipeline |
| **Sonara AI** | Audio / voice AI experiments |
| **Clipboard** | Electron cross-platform desktop app (frontend + backend) |
| **Thesis-Prepration** | Web scraper for faculty publications (data collection) |
| **CCTV Video Analytics (BS FYP)** | YOLO + OpenCV + TensorFlow, 92% detection accuracy |
| **AI Myth Buster** | NLP fact verification / claim classification |
| **MERN Flight Booking App** | Full-stack: auth, dashboards, chat, payments |
| **Brain Tumor MRI Detection** | Medical imaging DL |
| **Real-Time Face Detection + Email Alerts** | CV + automation |
| **Multi-Translation App** | NLP / translation APIs |
| **LUMS MLOps assignments** | Data ingestion, ETL stream processor, RL agent (Dino runner) |
| **LUMS Robotics** | ROS 2, YOLOv7, U-Net, camera calibration, perception |

### 1.6 Certifications
Google Data Analytics series (Foundations; Ask Questions; Prepare Data; Process Data; Analyze Data; Data Analysis with R) · Government of Pakistan: Machine Learning, AI and Data Science.

### 1.7 Known gaps (use these in "What to adjust")
- **AWS / GCP:** mostly Azure + Oracle; little AWS (SageMaker, EKS) or GCP (Vertex AI)
- **Kubernetes**, Kubeflow, MLflow at production scale: not on CV
- **Big data:** Spark, Databricks, Airflow, Kafka: limited / not on CV
- **TypeScript, Next.js:** not listed
- No published papers yet (MS in progress)
- ~2.5 yrs experience, so 5+ yr senior roles are a stretch
- Arabic language: not stated (some KSA roles ask for it)

---

## 2. Target Roles and Locations

### 2.1 Role titles to search (ranked by preference)
1. **AI Engineer** / Applied AI Engineer / GenAI Engineer / LLM Engineer
2. **Machine Learning Engineer** / ML Engineer / AI/ML Engineer
3. **Data Scientist** (ML-heavy, not dashboard-only)
4. **Computer Vision Engineer** / Edge AI Engineer / Embedded AI Engineer
5. **NLP Engineer** / Conversational AI Engineer / AI Agent Engineer
6. **Software Engineer (AI/ML)** / Python Backend Engineer (AI products)
7. **Full-Stack Engineer** (AI products, MERN / Python)

**Do NOT include:** MLOps / ML Platform / ML Infrastructure Engineer roles, DevOps / Cloud / SRE Engineer roles, or AI Research Engineer / Research Assistant roles. (MLOps or cloud skills appearing *inside* an AI/ML Engineer JD are fine. Only the job **title / main role** is excluded.)

### 2.2 Locations: only these two options
| Option | Where | Work mode allowed |
|---|---|---|
| **A** | **Lahore, Pakistan** | On-site, Hybrid, or Remote |
| **B** | **Anywhere else in the world** (other Pakistani cities, Saudi Arabia, UAE, GCC, Europe, US, Asia, etc.) | **Remote only** |

Rules for Option B (remote):
- Must be **fully remote**. "Remote for first 3 months then relocate", "remote within [country] only", or "hybrid" don't count.
- Must be open to candidates in **Pakistan**: "Remote – Worldwide / Anywhere", "Remote – EMEA / Asia / APAC", "Remote – Pakistan", or no residency restriction stated. Skip "Remote – US only / EU only / UK only / KSA residents only".
- Remote roles at Saudi / UAE / Gulf companies are welcome **as long as they are fully remote** and open to Pakistan.
- Contractor arrangements (Deel, Remote.com, Upwork-style contracts with a real company) are fine.

### 2.3 Priority companies (check EVERY one on EVERY run)

These are the employers he most wants. For each, find **all open roles that match Section 2.1** (Lahore office, or remote). Use **(a)** the company's own careers page, which is often a Workable / Lever / Greenhouse / BambooHR / Zoho Recruit / LinkedIn-hosted board (find it by searching `"<company> careers"`), **and (b)** the company's **LinkedIn Jobs** tab, **and (c)** a search `"<company>" jobs Lahore AI OR "machine learning" OR "data scientist"` (Rozee, Indeed PK, Glassdoor PK).

**Tier 1: his top choices (search hardest, list every matching role)**
| Company | Notes |
|---|---|
| **Systems Limited** | Large Lahore HQ; has data / AI / Gen AI practices. Check Lahore listings for AI, ML, Data Science, Python, Full-Stack. |
| **NETSOL Technologies** | Lahore HQ; AI / data teams for leasing and finance products. |
| **Devsinc** | Lahore; frequently hires AI / ML, Python, full-stack. |
| **10Pearls** | Has a Lahore office; AI, data and full-stack roles. |

**Tier 2: other large / well-known Pakistani tech employers with a Lahore office or remote roles**
Arbisoft · Motive · Afiniti · Confiz · Tkxel · Techlogix · Contour Software · i2c · VentureDive · Securiti · Xgrid · Tintash · Programmers Force · Xavor · Visionet · Educative · NorthBay Solutions · Remotebase · Kwanso · Folio3 · Emumba · Red Buffer · Datics AI · Siemens EDA (Mentor Graphics) Lahore · Teradata Pakistan · Ebryx · Zones · Jazz · Careem · Bazaar · Turing · Cloudways · Strategic Systems International · S&P Global Pakistan

For Tier 2, confirm the role is in **Lahore** or **remote**. Skip roles that are only at the Islamabad or Karachi office.

**Rules for priority companies**
- Seniority stretch: for **Tier 1**, include roles asking up to **5 years** (mark them as `Stretch`). For Tier 2, up to 4 years as usual.
- Role-name stretch: for Tier 1, also include "Associate / Software Engineer – Python", "Data Engineer" (only if ML-related), and "AI Trainee / Graduate AI Engineer" programs, since the title often hides AI work.
- Never drop a priority-company job only because its posting date is unclear.
- Record **every** Tier 1 and Tier 2 company in the `Company Coverage` sheet (Section 7), including ones with **0** matching roles, with the careers URL you checked. That way he can see why each company did or didn't show up.

---

## 3. Sources to Search

### 3.0 Use a browser: this is required for a full run

LinkedIn, Indeed, Glassdoor, Wellfound, Rozee and Bayt either need JavaScript or block plain page fetches. **LinkedIn is the single biggest source of Lahore and remote AI jobs.** Skipping it was the main reason an earlier run found only 10 jobs.

1. **First, check for browser tools**: Claude in Chrome (`mcp__claude-in-chrome__*`) or the desktop app's built-in browser (`*Claude_Browser__*`). Load them with ToolSearch if they're deferred.
2. **If a browser is available:** use it for LinkedIn, Indeed, Glassdoor, Rozee, Bayt and Wellfound. If Ahmad is already signed in to LinkedIn in that browser, that's fine: **read** the listings only. Never apply, message, follow, or change anything on his account.
3. **If no browser is available:** still try these public URLs with WebFetch before giving up:
   - LinkedIn public job search (past week = `f_TPR=r604800`, remote = `f_WT=2`, hybrid = `f_WT=3`, on-site = `f_WT=1`):
     `https://www.linkedin.com/jobs/search?keywords=AI%20Engineer&location=Lahore&f_TPR=r604800`
     `https://www.linkedin.com/jobs/search?keywords=Machine%20Learning%20Engineer&location=Pakistan&f_WT=2&f_TPR=r604800`
     `https://www.linkedin.com/jobs/search?keywords=AI%20Engineer&location=Worldwide&f_WT=2&f_TPR=r604800`
   - LinkedIn guest endpoint (returns plain HTML job cards; page with `&start=25`, `&start=50`, …):
     `https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=AI%20Engineer&location=Lahore&f_TPR=r604800&start=0`
   - If these are blocked too, put this **at the top of the final reply**: "⚠️ No browser: LinkedIn / Indeed / Glassdoor not searched, so results are incomplete. Re-run in the Claude desktop app with the Chrome extension or built-in browser."
4. Respect each site's terms: public listings only, no aggressive scraping, a short pause between page loads.
5. **Never ask for, type, or store any password or login.** Ahmad signs in to LinkedIn himself in the browser beforehand. If LinkedIn shows a login page, a CAPTCHA or a verification check, stop using LinkedIn for this run, note it in `Run Notes`, and continue with the other sources.

### 3.1 Global aggregators
- **LinkedIn Jobs**: search **each role title in 2.1** × **Lahore (all modes)** and × **Pakistan + Remote** and × **Worldwide + Remote**, with **Past week**, and Experience = Internship (paid) / Entry / Associate / Mid-Senior. Open **every** result page (25 per page) until results are older than 7 days or irrelevant.
- **Google Jobs** (search "`<title> jobs <location>`", the Google jobs panel)
- **Indeed** — pk.indeed.com, sa.indeed.com, ae.indeed.com, indeed.com
- **Glassdoor**
- **Wellfound (AngelList)** — startups, many remote
- **Y Combinator — Work at a Startup** (workatastartup.com)

### 3.2 Pakistan (Lahore any mode; other cities remote only)
- **Rozee.pk**
- **Mustakbil.com**
- **LinkedIn** (Pakistan filter)
- **Indeed Pakistan** (pk.indeed.com)
- **Careers pages:** all **priority companies in Section 2.3** (mandatory), plus Sadapay, NayaPay, Mindbridge, Antematter, Arpatech, DPL, Venturenox, CodeNinja, Musketeers Tech, Bricks Tech, Exper Labs
- **Other Pakistani boards:** hired.pk, Taraki Jobs, Qureos, Glassdoor PK, Jobz.pk (IT section only). Their "N days ago" labels are often wrong, so always open the posting itself to check it's live.
- **LUMS / NUST / FAST career portals** and LinkedIn posts by Pakistani tech recruiters ("hiring ML engineer Lahore")

### 3.3 Saudi Arabia and GCC (REMOTE roles only)
- **Bayt.com**
- **GulfTalent**
- **NaukriGulf**
- **LinkedIn** (Saudi Arabia / UAE / Qatar filter)
- **Indeed** Saudi / UAE
- **Careers pages (KSA):** SDAIA, NEOM, Aramco Digital, STC / stc Group, Elm, Lucidya, Mozn, Tamara, Tabby, Jahez, Salla, Lean Technologies, Foodics, KAUST, Tahakom, Humain (PIF AI company), Thiqah, Unifonic
- **Careers pages (UAE / Qatar):** G42, Core42, Presight AI, Inception, TII (Technology Innovation Institute), MBZUAI, Careem, Talabat, Noon, Emirates Group IT, QCRI, Ooredoo, Abu Dhabi AI firms

### 3.4 Remote-first boards
- **We Work Remotely**, **Remote OK**, **Remotive**, **Himalayas.app**, **Working Nomads**, **JustRemote**
- **Turing.com**, **Toptal**, **Arc.dev**, **Andela**, **Crossover**, **Deel / Remote.com job boards**
- **AI-specific:** aijobs.net (ai-jobs.net), MLOps Community job board, Hugging Face jobs, **Kaggle jobs**, "Who is hiring" (Hacker News monthly thread, filter REMOTE + ML/AI)

### 3.5 Remote-friendly company job boards (ATS sites)
- Greenhouse (`boards.greenhouse.io`), Lever (`jobs.lever.co`), Ashby (`jobs.ashbyhq.com`), Workable (`apply.workable.com`): search them with "remote" + role keywords
- **Arbeitnow**, **EU-Startups jobs**, **Otta / Welcome to the Jungle**: use their **remote** filter only

---

## 4. Search Queries

Run **every** query in 4.1 for **both** location groups (Lahore, and Remote). Use the **past 7 days** filter on every board that supports it. Company careers pages have no date filter: take every open matching role.

**Search budget (minimums per run):** at least **60 distinct searches / page loads**, made up of all 20 base queries × {Lahore, Remote}, plus every Tier 1 and Tier 2 company in Section 2.3, plus the `site:` queries in 4.2. Don't finish early because the first pages looked thin.

### 4.1 Base queries (combine with "Lahore" or "remote")
```
"AI Engineer"                      "Machine Learning Engineer"
"ML Engineer"                      "Generative AI Engineer"
"LLM Engineer"                     "Data Scientist" machine learning
"Computer Vision Engineer"         "Applied AI Engineer"
"AI/ML Engineer"                   "NLP Engineer"
"Edge AI Engineer"                 "AI Software Engineer"
"Python Developer" AI              "Full Stack Engineer" AI
"Junior Machine Learning Engineer" "Associate Data Scientist"
"RAG" engineer                     "AI Agent" developer
```

### 4.2 Example combined queries
```
AI Engineer jobs Lahore
Machine Learning Engineer Lahore
Data Scientist Lahore
Full Stack Developer AI Lahore
Machine Learning Engineer remote Pakistan
remote "AI Engineer" worldwide
remote ML engineer "Pakistan" OR "worldwide" OR "anywhere" OR "EMEA" OR "APAC"
Generative AI Engineer remote Saudi Arabia
LLM Engineer remote UAE
site:rozee.pk machine learning Lahore
site:bayt.com "machine learning engineer" remote
site:linkedin.com/jobs "AI Engineer" Lahore
site:boards.greenhouse.io "machine learning" remote
site:jobs.lever.co "AI engineer" remote
site:jobs.ashbyhq.com "ML engineer" remote
```
On LinkedIn, use **Location = Lahore** for any work mode, and **Location = Worldwide / Pakistan + "Remote" filter** for everything else.

### 4.3 Booster keywords (extra weight when present in a JD)
`PyTorch, LLM, RAG, LangGraph, LangChain, agents, MCP, llama.cpp, vLLM, quantization, edge, Raspberry Pi, Jetson, VLM, YOLO, OpenCV, segmentation, OCR, FastAPI, Azure, Terraform, Docker, CI/CD, GitHub Actions, MLOps, model deployment, inference optimization, Transformers, fine-tuning, LoRA`

---

## 5. Relevance Filter (apply before scoring)

### Keep a job if ALL are true
- The title or core duties involve **AI / ML / Data Science / MLOps / CV / NLP / LLMs**, **or** it's Software / Full-Stack / Backend / DevOps work **on an AI or data product**.
- Required experience is **≤ 4 years** (or not stated).
- Location is **Lahore (any mode)** **or** the job is **fully remote and open to candidates in Pakistan** (Section 2.2).
- **Still open**: the application link works and the page doesn't say "no longer accepting applications" or show an apply-by date that has passed.
- **Date:** posted within the last **7 days** on boards; any open date for priority companies and careers / ATS pages. **Unknown date + live page = keep**, marked `Unverified`.

### Drop a job if ANY are true
- **On-site or hybrid anywhere other than Lahore** (Karachi, Islamabad, Riyadh, Dubai, Europe, etc.), even with visa sponsorship or relocation.
- The role is mainly **MLOps / ML Platform / ML Infrastructure**, **DevOps / Cloud / SRE**, or **AI Research Engineer / Research Assistant**.
- Senior / Staff / Principal / Lead / Manager / Head-of with **5+ years** required.
- Pure data entry, data annotation / labeling gig, BI dashboard-only (Power BI / Tableau with no ML), or sales / pre-sales.
- Unrelated engineering (mechanical, civil, embedded-firmware-only with no ML, QA manual testing).
- Needs a **security clearance / citizenship** he doesn't have (US clearance, Saudi-nationals-only "Saudization" roles, UAE-nationals-only roles).
- Unpaid internships, "commission-only", or obvious scams (asks for fees, Telegram/WhatsApp-only contact, vague company with no website).
- Remote roles limited to US-only / EU-only / UK-only residents.
- PhD **required** (PhD "preferred" is fine, keep it).

---

### 5.1 Widening steps (use only if you have fewer than 40 jobs after all sources)
Apply these in order, re-checking the count after each one, and mark rows they add with `Widened: step N` in the Verification Notes column:
1. Extend the board search window from 7 to **14 days**, then to **30 days**.
2. Allow **4–5 years** experience at any company (as `Stretch`).
3. Add adjacent titles: "Python Developer" / "Backend Engineer" at AI-product companies, "Data Engineer" (ML pipelines), "AI Trainee / Graduate Program" (paid), "Prompt / LLM Application Developer".
4. Include remote jobs with **"Remote – Asia / APAC / Middle East"** limits as long as Pakistan isn't excluded.
5. Include Lahore-based **contract / part-time** AI work at real companies.

Never widen into: on-site outside Lahore, MLOps / DevOps / Research titles, 6+ years, nationals-only, or scam postings.

## 6. Fit Scoring Rubric (0–100)

Score each job by reading the **full job description**. Be honest. Don't inflate scores.

| Component | Weight | How to score |
|---|---|---|
| **Core skills match** | 35 | % of the JD's *required* technical skills he clearly has (Section 1.4) × 35 |
| **Role-type preference** | 15 | Ranks 1–2 in Section 2.1 = 15 · ranks 3–5 = 12 · ranks 6–7 = 9 |
| **Experience level match** | 15 | 0–3 yrs required = 15 · 3–4 yrs = 9 · not stated = 12 · 5+ = 0 (should have been filtered) |
| **Domain / project evidence** | 15 | A portfolio project in Section 1.5 directly shows the main duty (e.g. edge VLM, self-hosted LLM API, RAG agent, YOLO CV) = 15 · partial = 8 · none = 0 |
| **Location & schedule fit** | 10 | Remote with flexible hours or Asia-friendly timezone = 10 · Lahore hybrid = 10 · Lahore on-site = 8 · remote but requires heavy US/EU hour overlap = 6 |
| **Nice-to-haves / education** | 10 | MS AI in progress, cloud (Azure), Docker / CI/CD, preferred skills he has, language requirements met |

### Fit labels and "Chances"
| Score | Fit label | Chance of getting an interview |
|---|---|---|
| 85–100 | **Excellent** | High: apply today, tailor CV slightly |
| 70–84 | **Strong** | Good: apply with tailored CV + short cover note |
| 55–69 | **Moderate** | Fair: apply after closing 1–2 gaps in the CV wording |
| 40–54 | **Stretch** | Low: apply only if very interested, or skip |
| < 40 | Drop it, don't include in the sheet |

Adjust "Chances" down one level when: 200+ applicants are shown, it's a FAANG-tier company, Arabic is required, or it's a global remote role with very high competition. Adjust up one level when: it's a **Tier 1 priority company** (his referral / LUMS network is strongest there), it's a Lahore company / Lahore office, the remote role explicitly mentions Pakistan, or the posting is < 48 h old with few applicants.

### "What to adjust in CV" column: rules
Give **concrete, specific** advice in 2–4 bullet points, for example:
- "Move the EdgeLog (Raspberry Pi VLM) project to the top; JD stresses edge deployment and quantization."
- "Add 'LangGraph, MCP, RAG' to skills line; JD asks for agent frameworks."
- "Rename DevOps role bullets to emphasize ML model deployment (CI/CD for ML)."
- "Gap: AWS SageMaker required. Mention Azure ML as equivalent and note AWS is in progress / do AWS ML Specialty."
- "Add numbers: OCR +40% accuracy, −30% pipeline time, −15% cloud cost."

Never suggest he claim a skill he doesn't have. Gaps go in the `Missing Skills / Gaps` column.

---

## 7. Excel Output Specification

**Filename:** `Job_Openings_YYYY-MM-DD.xlsx` (use today's date)
Build it with `openpyxl` (or pandas + openpyxl for formatting).

### Sheet 1: `Jobs` (main sheet, one row per job, sorted by Fit Score descending)

| # | Column | Content |
|---|---|---|
| A | **#** | Row number |
| B | **Fit Score** | 0–100 (number) |
| C | **Fit Label** | Excellent / Strong / Moderate / Stretch |
| D | **Chances** | High / Good / Fair / Low, plus a short reason (≤ 15 words) |
| E | **Job Title** | Exactly as posted |
| F | **Role Category** | AI Engineer / ML Engineer / Data Scientist / CV / NLP-LLM / Software-AI / Full-Stack |
| G | **Company** | Company name. Add a **★** before Tier 1 names, e.g. "★ Systems Limited" |
| H | **Company Description** | 1–2 sentences: what the company does, size, industry (from their site / LinkedIn) |
| I | **Company Country** | Where the company is based (Pakistan / Saudi Arabia / UAE / USA / …) |
| J | **Job Location** | "Lahore" or "Remote" |
| K | **Work Mode** | **Remote / Hybrid / On-site** (Hybrid / On-site only allowed for Lahore) |
| L | **Employment Type** | Full-time / Part-time / Contract / Internship |
| M | **Experience Required** | e.g. "1–3 years", "Not stated" |
| N | **Salary** | As posted with currency, else "Not stated" |
| O | **Remote Eligibility / Timezone** | e.g. "Worldwide", "EMEA", "Pakistan", "GMT+3 ±3h", or "N/A (Lahore)" |
| P | **Job Description (Summary)** | 3–5 sentence summary of the role and main responsibilities |
| Q | **Key Requirements** | Required skills / qualifications, semicolon-separated |
| R | **Nice-to-Have** | Preferred skills, semicolon-separated |
| S | **Matching Skills** | Which of his skills match (from Section 1.4) |
| T | **Missing Skills / Gaps** | What the JD wants that he lacks |
| U | **Relevant Projects** | Which of his projects (Section 1.5) best prove fit |
| V | **How My CV Fits** | 2–3 sentences: an honest explanation of the match |
| W | **What to Adjust in CV** | 2–4 concrete bullet points (Section 6 rules) |
| X | **Schedule Fit** | ✅ Flexible / part-time / fits with MS, or ⚠️ Full-time fixed hours (note the hours if stated) |
| Y | **Date Posted** | YYYY-MM-DD (or "~N days ago" if only relative) |
| Z | **Application Deadline** | YYYY-MM-DD or "Not stated" |
| AA | **Source** | LinkedIn / Rozee / Bayt / Company site / … |
| AB | **Apply Link** | Direct URL to the posting (clickable hyperlink) |
| AC | **Company Website / LinkedIn** | URL |
| AD | **First Seen** | Date this job first appeared in a run |
| AE | **Status** | Leave as "New" (he'll update it: Applied / Interview / Rejected / Skip) |
| AF | **Priority Company** | Tier 1 / Tier 2 / — |
| AG | **Verification Notes** | How the job was confirmed live; `Unverified date`; `Widened: step N` if added by Section 5.1 |
| AH | **Score Breakdown** | e.g. "Skills 30/35 · Role 15 · Exp 15 · Projects 15 · Location 10 · Extras 8" |

### Sheet 2: `Top Picks`
**All Tier 1 company matches first** (whatever their score), then the top 15 other jobs by Fit Score, with only these columns: Fit Score, Job Title, Company, Company Country, Job Location, Work Mode, Chances, What to Adjust in CV, Apply Link.

### Sheet 3: `Summary`
- Total jobs found, broken down by **Lahore vs Remote**, **Company Country**, **Role Category**, and **Fit Label** (small tables with counts)
- Average fit score
- **Most frequent missing skills** across all jobs (top 10, with count). This tells him what to learn next.
- **Most requested skills** across all JDs (top 15, with count)
- Jobs closing within 7 days

### Sheet 4: `Company Coverage` (required)
One row per **Tier 1 and Tier 2 company** in Section 2.3:
`Tier · Company · Careers page URL checked · LinkedIn Jobs checked (Y/N) · Total open roles seen · Matching roles added · Why others were excluded (e.g. "3 Senior 6+ yrs; 2 Islamabad on-site; 1 DevOps") · Checked at (time)`
A company with 0 matches still gets a row, so he can see it was actually checked.

### Sheet 5: `Run Notes`
Run date and time (Asia/Karachi), **browser used (which one, or "none")**, total searches / page loads made, widening steps used (Section 5.1), sources searched (✅ / ❌ with reason if blocked), queries used, filters applied, number of jobs dropped and why (counts per reason), and any assumptions you made.

### Formatting rules
- Bold header row with a fill color, freeze the header row and the first 5 columns, auto-filter on all columns.
- Wrap text in long columns (H, P, Q, R, S, T, U, V, W); set sensible column widths (long-text columns ~50, others 12–25).
- Conditional fill on **Fit Score**: ≥ 85 green, 70–84 light green, 55–69 yellow, 40–54 orange.
- **Apply Link** and **Company Website** must be real clickable hyperlinks.
- Work Mode cell color: Remote = blue, Hybrid = purple, On-site = grey.
- No empty rows. Dates in ISO format.

---

## 8. De-duplication and Daily Tracking

- Keep a master file **`jobs_history.csv`** next to the Excel output, with columns: `job_key, company, title, country, apply_link, first_seen, last_seen, fit_score`.
- `job_key` = lower-cased `company | title | city`, with extra spaces and punctuation removed.
- On each run:
  1. Load `jobs_history.csv` if it exists.
  2. The same job posted on several sites counts as **one** row: prefer the company careers page link, then LinkedIn, then others.
  3. Jobs already in history → **don't** put them in today's `Jobs` sheet again. Update `last_seen` only. (Optional: add an `Older Still Open` sheet listing history jobs with fit ≥ 70 that are still open.)
  4. New jobs → add to the sheet **and** to history with `first_seen = today`.
- If the history file doesn't exist, this is the first run: use the past **14 days** on boards instead of 7.
- Repeat jobs from earlier runs that are **still open at Tier 1 / Tier 2 companies** go in a separate `Still Open (Priority)` sheet, so they never silently disappear.

---

## 9. Quality Checklist (do before you finish)

- [ ] Every row has a working **Apply Link** you actually opened (or that came straight from the listing).
- [ ] No invented data. Unknown = "Not stated".
- [ ] **40+ jobs**, or Section 5.1 widening fully used and explained in `Run Notes`.
- [ ] **Every** Tier 1 and Tier 2 company has a row in `Company Coverage`.
- [ ] LinkedIn was searched (browser or public URLs), or the ⚠️ warning is at the top of the reply.
- [ ] Every row is either **Lahore** or **fully Remote and open to Pakistan**. No on-site / hybrid jobs outside Lahore.
- [ ] No MLOps / ML Platform, DevOps / Cloud, or Research Engineer / Research Assistant roles.
- [ ] No senior 5+ yr roles, no off-field roles, no nationals-only roles.
- [ ] Every Fit Score follows the Section 6 rubric; spot-check 3 rows by recalculating.
- [ ] "What to Adjust in CV" is specific to each job (not copy-pasted across rows).
- [ ] Rows sorted by Fit Score (highest first).
- [ ] Jobs from **both** groups: Lahore and Remote. Remote should include Pakistani, Gulf and international companies. If a group gave zero results, say so in `Run Notes`.
- [ ] Workbook opens without errors; hyperlinks click through; filters work.
- [ ] `jobs_history.csv` updated.

---

## 10. Final Reply Format

After saving the file, reply with:

```
✅ Job scan complete — <date>
Found: <N> new relevant jobs (Lahore: x · Remote: x — of which PK cos: x · Gulf cos: x · Intl cos: x)
Top 5:
 1. <Score> — <Title> @ <Company> (<Company Country>, <Lahore/Remote>) — <Chances>
 2. ...
★ Tier 1: Systems <n> · NETSOL <n> · Devsinc <n> · 10Pearls <n>   |   Tier 2 companies with matches: <n>/<total>
🔎 Sources: browser=<name/none> · LinkedIn=<✅/❌> · searches made=<n> · widening used=<none/steps>
⏰ Closing within 7 days: <list or "none">
📈 Most common gap this run: <skill> (in N jobs)
File: Job_Openings_<date>.xlsx
```

---

## 11. Optional: run it every day automatically

Give a Claude agent this prompt on a daily schedule (for example 9:00 AM Asia/Karachi):

> "Read README.md in this folder and do the full daily job-hunt run it describes. Use jobs_history.csv for de-duplication. Save Job_Openings_<today>.xlsx and send me the summary."

**For the fullest results, run it from the Claude desktop app with the Chrome extension or the built-in browser turned on, and LinkedIn signed in.** Without a browser, LinkedIn / Indeed / Glassdoor mostly can't be read, and the job count drops sharply.

Keep `README.md` and `jobs_history.csv` in the same folder between runs. When his CV changes (new job, new skills, graduation in 2027), update **Section 1** and the scoring stays correct.
