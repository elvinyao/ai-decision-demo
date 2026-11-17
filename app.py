from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from jira_decision_service import DB_FILE, JiraDecisionEngine, JiraKnowledgeBase

app = FastAPI(title="Jira Decision AI Demo", version="1.0.0")
kb = JiraKnowledgeBase(DB_FILE)
engine = JiraDecisionEngine(kb)


class TicketInput(BaseModel):
    key: str = Field(..., min_length=1)
    component: str = Field(..., min_length=1)
    severity: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    description: str = ""


class EvaluateRequest(BaseModel):
    ticket: TicketInput
    api_key: str | None = None


class FeedbackRequest(BaseModel):
    ticket: TicketInput
    confirmed_same: bool
    confirmed_similar: bool
    reason: str = ""


@app.get("/api/v1/jira/kb")
def get_kb() -> dict:
    return {
        "records": kb._load(),
        "has_env_key": engine.has_configured_key(),
    }


@app.post("/api/v1/jira/evaluate")
def evaluate_ticket(req: EvaluateRequest) -> dict:
    try:
        return engine.evaluate_ticket(req.ticket.model_dump(), api_key=req.api_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"判定调用失败: {exc}") from exc


@app.post("/api/v1/jira/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    records = kb.add_feedback(
        ticket=req.ticket.model_dump(),
        confirmed_same=req.confirmed_same,
        confirmed_similar=req.confirmed_similar,
        reason=req.reason,
    )
    return {"records": records, "saved_key": req.ticket.key}


@app.post("/api/v1/jira/reset")
def reset_kb() -> dict:
    return {"records": kb.reset()}


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gemini Decision 1 · Jira 智能判定与反馈闭环 Demo</title>
  <style>
    :root {
      --bg: #f4f6f9;
      --surface: #ffffff;
      --surface-alt: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --yes-bg: #dcfce7;
      --yes-fg: #166534;
      --no-bg: #f1f5f9;
      --no-fg: #334155;
      --warn-bg: #fef3c7;
      --warn-fg: #92400e;
      --danger-bg: #fee2e2;
      --danger-fg: #991b1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", "Noto Sans CJK SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .topbar {
      background: #0f172a;
      color: #fff;
      padding: 14px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .status-mock { background: #334155; color: #fde68a; }
    .status-live { background: #065f46; color: #a7f3d0; }
    .key-box {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .key-box label { font-size: 12px; color: #cbd5e1; }
    .key-box input {
      background: #1e293b;
      border: 1px solid #334155;
      color: #fff;
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 12px;
      width: 240px;
    }
    .demo-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }
    .demo-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      margin-right: 4px;
    }
    button {
      cursor: pointer;
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      padding: 7px 13px;
      font-size: 13px;
      font-weight: 500;
      transition: all 0.15s ease;
    }
    button:hover:not(:disabled) { background: var(--surface-alt); border-color: #cbd5e1; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    button.primary {
      background: var(--primary);
      border-color: var(--primary);
      color: #fff;
    }
    button.primary:hover:not(:disabled) { background: var(--primary-hover); }
    button.step-btn.active {
      background: #eff6ff;
      border-color: var(--primary);
      color: var(--primary);
      font-weight: 600;
    }
    button.ghost { color: var(--muted); margin-left: auto; }
    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 24px;
      display: grid;
      grid-template-columns: 1.05fr 0.95fr 1.15fr;
      gap: 18px;
      align-items: start;
    }
    @media (max-width: 1120px) {
      .container { grid-template-columns: 1fr; }
      button.ghost { margin-left: 0; }
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .panel-title {
      margin: 0;
      font-size: 15px;
      font-weight: 600;
    }
    .panel-sub {
      font-size: 12px;
      color: var(--muted);
    }
    .kb-list { display: flex; flex-direction: column; gap: 10px; }
    .kb-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: var(--surface-alt);
      transition: border-color 0.2s ease, background 0.2s ease;
    }
    .kb-card.highlight {
      border-color: var(--primary);
      background: #eff6ff;
    }
    .kb-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .ticket-key {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 5px;
      background: #e2e8f0;
      color: #1e293b;
    }
    .meta-tag {
      font-size: 11px;
      color: var(--muted);
    }
    .kb-summary { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    .kb-note { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
    }
    .badge-yes { background: var(--yes-bg); color: var(--yes-fg); }
    .badge-no { background: var(--no-bg); color: var(--no-fg); }
    .badge-warn { background: var(--warn-bg); color: var(--warn-fg); }
    .badge-team { background: #e0e7ff; color: #3730a3; }
    .form-group { margin-bottom: 12px; }
    .form-group label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      margin-bottom: 4px;
    }
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    input[type="text"], select, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      background: #fff;
      color: var(--text);
    }
    textarea { resize: vertical; min-height: 72px; }
    .preset-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .preset-btn { font-size: 12px; padding: 4px 9px; }
    .result-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 12px;
      background: var(--surface-alt);
    }
    .result-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .result-title { font-size: 13px; font-weight: 600; }
    .metric-row {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
    }
    .bar-wrap {
      display: grid;
      grid-template-columns: 82px 1fr 48px;
      align-items: center;
      gap: 8px;
    }
    .bar-track {
      height: 8px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .fill-yes { background: #16a34a; }
    .fill-no { background: #64748b; }
    .fill-warn { background: #d97706; }
    .fill-conf { background: var(--primary); }
    .ref-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 8px;
    }
    .ref-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      padding: 6px 8px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 7px;
    }
    .feedback-box {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px dashed var(--border);
    }
    .toggle-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }
    .toggle-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      background: #fff;
    }
    .toast {
      margin-top: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      display: none;
    }
    .toast.show { display: block; }
    .toast.ok { background: var(--yes-bg); color: var(--yes-fg); }
    .toast.err { background: var(--danger-bg); color: var(--danger-fg); }
    .empty-state {
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      padding: 28px 12px;
      border: 1px dashed var(--border);
      border-radius: 10px;
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>Gemini Decision 1 · Jira 智能判定与反馈闭环</h1>
      <span id="engine-status" class="status-pill status-mock">Mock 本地模拟推理</span>
    </div>
    <div class="key-box">
      <label for="api-key-input">GEMINI_API_KEY (可选):</label>
      <input id="api-key-input" type="password" placeholder="留空则使用本地模拟，或自动读取环境变量" />
    </div>
  </header>

  <div class="demo-bar">
    <span class="demo-label">闭环演练流程：</span>
    <button class="step-btn" id="step-1-btn" onclick="runStep1()">① 初次判定 PROJ-205 (帳票1円ずれ)</button>
    <button class="step-btn" id="step-2-btn" onclick="runStep2()">② 注入人工反馈 (同件:不要 / 類似:要)</button>
    <button class="step-btn" id="step-3-btn" onclick="runStep3()">③ 判定类似新票 PROJ-310 (验证秒级生效)</button>
    <button class="ghost" onclick="resetKB()">↺ 重置知识库</button>
  </div>

  <main class="container">
    <!-- Column 1: Knowledge Base -->
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">1. 历史 Jira 判例库</h2>
          <div class="panel-sub">jira_knowledge_base.json · 动态 Few-Shot 检索源</div>
        </div>
        <span class="meta-tag" id="kb-count">0 条判例</span>
      </div>
      <div id="kb-list" class="kb-list"></div>
    </section>

    <!-- Column 2: Ticket Input -->
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">2. 待判定 Jira 工单</h2>
          <div class="panel-sub">模拟外部系统传入的新票据</div>
        </div>
      </div>

      <div class="preset-row">
        <button class="preset-btn" onclick="fillPreset('PROJ-205')">填入 PROJ-205 (帳票消費税)</button>
        <button class="preset-btn" onclick="fillPreset('PROJ-310')">填入 PROJ-310 (納品書割引)</button>
        <button class="preset-btn" onclick="fillPreset('PROJ-402')">填入 PROJ-402 (PayPay二重決済)</button>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="t-key">Ticket Key</label>
          <input id="t-key" type="text" />
        </div>
        <div class="form-group">
          <label for="t-severity">Severity</label>
          <select id="t-severity">
            <option value="Critical">Critical</option>
            <option value="Major">Major</option>
            <option value="Minor" selected>Minor</option>
          </select>
        </div>
      </div>

      <div class="form-group">
        <label for="t-component">Component (模块)</label>
        <input id="t-component" type="text" />
      </div>

      <div class="form-group">
        <label for="t-summary">Summary (摘要)</label>
        <input id="t-summary" type="text" />
      </div>

      <div class="form-group">
        <label for="t-description">Description (详细描述)</label>
        <textarea id="t-description"></textarea>
      </div>

      <button id="eval-btn" class="primary" style="width: 100%" onclick="evaluateCurrent()">执行 AI 判定</button>
      <div id="eval-toast" class="toast"></div>
    </section>

    <!-- Column 3: Evaluation Result & Feedback Loop -->
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">3. 判定结果与人工反馈闭环</h2>
          <div class="panel-sub">一次调用返回：同件調査 + 類似調査 + 担当チーム</div>
        </div>
        <span id="result-mode-tag" class="meta-tag">等待判定</span>
      </div>

      <div id="result-area">
        <div class="empty-state">
          点击上方「① 初次判定 PROJ-205」或左侧「执行 AI 判定」开始演示。
        </div>
      </div>

      <div class="feedback-box">
        <div class="panel-title" style="font-size: 14px; margin-bottom: 8px;">人工审核 / 纠错反馈写入</div>
        <div class="toggle-grid">
          <label class="toggle-card">
            <span>同件調査 (Same Case)</span>
            <input id="fb-same" type="checkbox" />
          </label>
          <label class="toggle-card">
            <span>類似調査 (Similar Case)</span>
            <input id="fb-similar" type="checkbox" checked />
          </label>
        </div>
        <div class="form-group">
          <label for="fb-reason">判定理由 / 社内ルールメモ</label>
          <textarea id="fb-reason" style="min-height: 60px;" placeholder="例如：帳票PDFは都度生成のためDB同件調査は不要だが、類似調査のみ必須..."></textarea>
        </div>
        <button id="fb-btn" style="width: 100%" onclick="submitFeedback()">提交人工反馈至知识库</button>
        <div id="fb-toast" class="toast"></div>
      </div>
    </section>
  </main>

  <script>
    const PRESETS = {
      "PROJ-205": {
        key: "PROJ-205",
        component: "帳票出力モジュール (Report)",
        severity: "Minor",
        summary: "請求書PDFの消費税端数処理が一部の取引先で1円ずれる",
        description: "外貨建て取引の請求書PDF出力時のみ、切り捨てではなく四捨五入になっている。"
      },
      "PROJ-310": {
        key: "PROJ-310",
        component: "帳票出力モジュール (Report)",
        severity: "Minor",
        summary: "納品書PDFの割引額計算で小数点以下の丸め誤差が発生",
        description: "割引率に小数を含む場合、PDF表示上の合計金額が1円ずれることがある。"
      },
      "PROJ-402": {
        key: "PROJ-402",
        component: "決済モジュール (Payment)",
        severity: "Critical",
        summary: "PayPay決済完了コールバック遅延時に重複引き落としが発生",
        description: "タイムアウト後の再試行制御で同一注文IDに対して二重請求される。"
      }
    };

    const DEMO_FEEDBACK_REASON = "帳票PDFは都度生成のためDB同件調査は不要だが、見積書や納品書でも同じ端数計算関数を使っている可能性が高いため類似調査のみ必須とする社内ルール。";

    let hasEnvKey = false;
    let lastSavedKey = null;

    function esc(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function updateEnginePill(activeMode) {
      const pill = document.getElementById("engine-status");
      const userKey = document.getElementById("api-key-input").value.trim();
      const live = activeMode ? activeMode === "gemini-decision-1" : (hasEnvKey || Boolean(userKey));
      pill.className = "status-pill " + (live ? "status-live" : "status-mock");
      pill.textContent = live ? "Gemini Decision 1 实时 API" : "Mock 本地模拟推理";
    }

    document.getElementById("api-key-input").addEventListener("input", () => updateEnginePill());

    function setActiveStep(stepNum) {
      [1, 2, 3].forEach((n) => {
        document.getElementById(`step-${n}-btn`).classList.toggle("active", n === stepNum);
      });
    }

    function fillPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      document.getElementById("t-key").value = p.key;
      document.getElementById("t-component").value = p.component;
      document.getElementById("t-severity").value = p.severity;
      document.getElementById("t-summary").value = p.summary;
      document.getElementById("t-description").value = p.description;
    }

    function readTicketForm() {
      return {
        key: document.getElementById("t-key").value.trim(),
        component: document.getElementById("t-component").value.trim(),
        severity: document.getElementById("t-severity").value,
        summary: document.getElementById("t-summary").value.trim(),
        description: document.getElementById("t-description").value.trim()
      };
    }

    function showToast(id, msg, isErr) {
      const el = document.getElementById(id);
      el.textContent = msg;
      el.className = "toast show " + (isErr ? "err" : "ok");
    }

    function hideToast(id) {
      const el = document.getElementById(id);
      el.className = "toast";
    }

    function renderKB(records) {
      document.getElementById("kb-count").textContent = `${records.length} 条判例`;
      const list = document.getElementById("kb-list");
      list.innerHTML = records.map((r) => {
        const same = r.labels?.needs_same_case_inv;
        const sim = r.labels?.needs_similar_case_inv;
        const hl = lastSavedKey && r.key === lastSavedKey ? " highlight" : "";
        return `
          <div class="kb-card${hl}">
            <div class="kb-top">
              <span class="ticket-key">${esc(r.key)}</span>
              <span class="meta-tag">${esc(r.component)} · ${esc(r.severity)}</span>
            </div>
            <div class="kb-summary">${esc(r.summary)}</div>
            <div class="badge-row">
              <span class="badge ${same ? "badge-yes" : "badge-no"}">同件調査: ${same ? "要" : "不要"}</span>
              <span class="badge ${sim ? "badge-yes" : "badge-no"}">類似調査: ${sim ? "要" : "不要"}</span>
            </div>
            <div class="kb-note">${esc(r.note || "")}</div>
          </div>
        `;
      }).join("");
    }

    function verdictBadgeClass(verdict) {
      if (verdict.startsWith("YES")) return "badge-yes";
      if (verdict.startsWith("NO")) return "badge-no";
      return "badge-warn";
    }

    function barFillClass(verdict) {
      if (verdict.startsWith("YES")) return "fill-yes";
      if (verdict.startsWith("NO")) return "fill-no";
      return "fill-warn";
    }

    function renderNoulCard(title, data) {
      const probPct = Math.round(data.prob_yes * 100);
      const confPct = Math.round(data.confidence * 100);
      return `
        <div class="result-card">
          <div class="result-head">
            <span class="result-title">${esc(title)}</span>
            <span class="badge ${verdictBadgeClass(data.verdict)}">${esc(data.verdict)}</span>
          </div>
          <div class="metric-row">
            <div class="bar-wrap">
              <span>Prob (Yes)</span>
              <div class="bar-track"><div class="bar-fill ${barFillClass(data.verdict)}" style="width: ${probPct}%"></div></div>
              <span>${data.prob_yes.toFixed(3)}</span>
            </div>
            <div class="bar-wrap">
              <span>Confidence</span>
              <div class="bar-track"><div class="bar-fill fill-conf" style="width: ${confPct}%"></div></div>
              <span>${data.confidence.toFixed(3)}</span>
            </div>
          </div>
        </div>
      `;
    }

    function renderResult(res) {
      updateEnginePill(res.mode);
      document.getElementById("result-mode-tag").textContent = `${res.ticket_key} · ${res.mode}`;
      const teamConfPct = Math.round(res.routing.confidence * 100);
      const refsHtml = (res.referenced_case_details || []).map((c) => `
        <div class="ref-item">
          <div>
            <span class="ticket-key">${esc(c.key)}</span>
            <span style="margin-left: 6px;">${esc(c.summary)}</span>
          </div>
          <span class="meta-tag">相似度 ${c.similarity.toFixed(3)}</span>
        </div>
      `).join("");

      document.getElementById("result-area").innerHTML = `
        ${renderNoulCard("同件調査 (needs_same_case_inv)", res.same_case_investigation)}
        ${renderNoulCard("類似調査 (needs_similar_case_inv)", res.similar_case_investigation)}
        <div class="result-card">
          <div class="result-head">
            <span class="result-title">担当チーム振り分け (assigned_team)</span>
            <span class="badge badge-team">${esc(res.routing.top_team)}</span>
          </div>
          <div class="metric-row">
            <div class="bar-wrap">
              <span>Confidence</span>
              <div class="bar-track"><div class="bar-fill fill-conf" style="width: ${teamConfPct}%"></div></div>
              <span>${res.routing.confidence.toFixed(3)}</span>
            </div>
          </div>
        </div>
        <div class="result-card">
          <div class="result-head" style="margin-bottom: 4px;">
            <span class="result-title">命中相似判例 (Few-Shot 参考)</span>
            <span class="meta-tag">Top ${res.referenced_cases.length}</span>
          </div>
          <div class="ref-list">${refsHtml}</div>
        </div>
      `;
    }

    async function loadKB() {
      const resp = await fetch("/api/v1/jira/kb");
      const data = await resp.json();
      hasEnvKey = data.has_env_key;
      updateEnginePill();
      renderKB(data.records);
    }

    async function evaluateCurrent() {
      hideToast("eval-toast");
      const btn = document.getElementById("eval-btn");
      btn.disabled = true;
      btn.textContent = "判定中...";
      try {
        const apiKey = document.getElementById("api-key-input").value.trim() || null;
        const resp = await fetch("/api/v1/jira/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticket: readTicketForm(), api_key: apiKey })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "请求失败");
        renderResult(data);
      } catch (err) {
        showToast("eval-toast", err.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = "执行 AI 判定";
      }
    }

    async function submitFeedback() {
      hideToast("fb-toast");
      const btn = document.getElementById("fb-btn");
      btn.disabled = true;
      try {
        const ticket = readTicketForm();
        const resp = await fetch("/api/v1/jira/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticket,
            confirmed_same: document.getElementById("fb-same").checked,
            confirmed_similar: document.getElementById("fb-similar").checked,
            reason: document.getElementById("fb-reason").value.trim()
          })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "反馈写入失败");
        lastSavedKey = data.saved_key;
        renderKB(data.records);
        showToast("fb-toast", `已将 ${data.saved_key} 人工判定写入知识库，下次判定秒级生效！`, false);
      } catch (err) {
        showToast("fb-toast", err.message, true);
      } finally {
        btn.disabled = false;
      }
    }

    async function runStep1() {
      setActiveStep(1);
      fillPreset("PROJ-205");
      await evaluateCurrent();
    }

    async function runStep2() {
      setActiveStep(2);
      fillPreset("PROJ-205");
      document.getElementById("fb-same").checked = false;
      document.getElementById("fb-similar").checked = true;
      document.getElementById("fb-reason").value = DEMO_FEEDBACK_REASON;
      await submitFeedback();
    }

    async function runStep3() {
      setActiveStep(3);
      fillPreset("PROJ-310");
      await evaluateCurrent();
    }

    async function resetKB() {
      setActiveStep(0);
      lastSavedKey = null;
      hideToast("eval-toast");
      hideToast("fb-toast");
      const resp = await fetch("/api/v1/jira/reset", { method: "POST" });
      const data = await resp.json();
      renderKB(data.records);
      document.getElementById("result-mode-tag").textContent = "等待判定";
      document.getElementById("result-area").innerHTML = `
        <div class="empty-state">知识库已重置为初始状态（PROJ-101 / PROJ-102）。点击「① 初次判定 PROJ-205」重新演练。</div>
      `;
      fillPreset("PROJ-205");
    }

    fillPreset("PROJ-205");
    loadKB();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE
