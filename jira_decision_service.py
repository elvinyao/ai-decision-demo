import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from google import genai

# =====================================================================
# 1. 模拟「训练数据库」(存储历史 Jira 票据与最终人工确认的标签)
# 实际生产中可以替换为 PostgreSQL (pgvector) / ChromaDB / Elasticsearch
# =====================================================================
DB_FILE = Path(__file__).parent / "jira_knowledge_base.json"

INITIAL_TRAINING_DATA: list[dict[str, Any]] = [
    {
        "key": "PROJ-101",
        "component": "決済モジュール (Payment)",
        "severity": "Critical",
        "summary": "クレジットカード決済時にタイムアウトで二重課金が発生",
        "description": "特定の時間帯にAPIゲートウェイが504を返し、リトライ処理で二重決済になる。",
        "labels": {
            "needs_same_case_inv": True,     # 涉及金额异常，必须查线上是否已有其他受害订单（同件調査）
            "needs_similar_case_inv": True,  # 必须排查其他支付方式如PayPay是否有同样重试漏洞（類似調査）
        },
        "note": "金銭影響があるため、過去ログの同件洗い出しと他決済手段の水平展開調査が必須。",
    },
    {
        "key": "PROJ-102",
        "component": "管理画面UI (Frontend)",
        "severity": "Minor",
        "summary": "ユーザー一覧画面のCSV出力ボタンの文言タイポ",
        "description": "「ダウンロード」が「ダウンロド」になっている。",
        "labels": {
            "needs_same_case_inv": False,
            "needs_similar_case_inv": False,
        },
        "note": "単純な表示崩れ・誤字のため追加調査不要。",
    },
]

DECISION_QUESTIONS = {
    "needs_same_case_inv": {
        "type": "noul",
        "instructions": (
            "参考 historical_reference_tickets 的判例标准，"
            "判断 target_new_jira_ticket 是否需要执行『同件調査』（排查生产环境日志/DB中是否已经发生过相同问题的数据或受害用户）？"
        ),
        "criteria": {
            "true_description": "存在数据不一致、金额错误、状态卡死或线上潜在已发生脏数据的风险，或历史相似票据要求同件調査。",
            "false_description": "纯UI显示问题、文案错误、尚未上线的测试环境问题，无需排查线上历史数据。",
        },
    },
    "needs_similar_case_inv": {
        "type": "noul",
        "instructions": (
            "参考 historical_reference_tickets 的判例标准，"
            "判断 target_new_jira_ticket 是否需要执行『類似調査』（水平排查其他类似画面、接口或共通组件是否存在同类代码缺陷）？"
        ),
        "criteria": {
            "true_description": "属于共通逻辑漏洞、权限校验缺失、框架级配置错误，可能在系统其他模块也写了类似代码。",
            "false_description": "单一页面特有的独立拼写错误或孤立问题，不具备水平展开排查的必要。",
        },
    },
    "assigned_team": {
        "type": "choice",
        "instructions": "这张新 Jira 票应该优先分派给哪个小组跟进？",
        "criteria": {
            "backend_team": "API、バッチ処理、決済、データベース、権限ロジック関連の不具合。",
            "frontend_team": "画面表示、CSS崩れ、ボタン文言、ブラウザ固有のUI不具合。",
            "infra_team": "ネットワーク、サーバーダウン、クラウド設定関連。",
            "other": "上記に当てはまらない、または情報不足で判断できない場合。",
        },
    },
}


class JiraKnowledgeBase:
    """管理历史 Jira 判例与人工反馈数据的简易存储类"""

    def __init__(self, filepath: Path = DB_FILE):
        self.filepath = filepath
        if not self.filepath.exists():
            self._save(INITIAL_TRAINING_DATA)

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.filepath.read_text(encoding="utf-8"))

    def _save(self, data: list[dict[str, Any]]) -> None:
        self.filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> list[dict[str, Any]]:
        """重置知识库为初始训练数据"""
        self._save(INITIAL_TRAINING_DATA)
        return self._load()

    def similarity_score(self, new_ticket: dict[str, Any], record: dict[str, Any]) -> float:
        # shortcut: SequenceMatcher for RAG similarity, upgrade to gemini-embedding-001 + pgvector over ~500 tickets
        query_text = f"{new_ticket.get('component', '')} {new_ticket.get('summary', '')} {new_ticket.get('description', '')}"
        rec_text = f"{record.get('component', '')} {record.get('summary', '')} {record.get('description', '')}"
        comp_bonus = 0.3 if record.get("component") == new_ticket.get("component") else 0.0
        return SequenceMatcher(None, query_text, rec_text).ratio() + comp_bonus

    def retrieve_similar_cases(self, new_ticket: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        """检索与新 Jira 票最相似的历史票据（模拟向量检索 RAG）"""
        records = self._load()
        ranked = sorted(records, key=lambda r: self.similarity_score(new_ticket, r), reverse=True)
        return ranked[:top_k]

    def add_feedback(
        self,
        ticket: dict[str, Any],
        confirmed_same: bool,
        confirmed_similar: bool,
        reason: str = "",
    ) -> list[dict[str, Any]]:
        """接收外部系统的人工审核/纠错反馈，实时写入判例库"""
        records = self._load()
        feedback_entry = {
            "key": ticket["key"],
            "component": ticket["component"],
            "severity": ticket["severity"],
            "summary": ticket["summary"],
            "description": ticket["description"],
            "labels": {
                "needs_same_case_inv": bool(confirmed_same),
                "needs_similar_case_inv": bool(confirmed_similar),
            },
            "note": reason or "外部システムからの担当者フィードバック結果",
        }
        records = [r for r in records if r["key"] != ticket["key"]]
        records.append(feedback_entry)
        self._save(records)
        print(f"\n[Feedback Saved] 票据 {ticket['key']} 的人工判定已写入知识库，下次推理立即生效！")
        return records


# =====================================================================
# 2. Gemini Decision 1 判定服务（无 API Key 时自动切换本地模拟）
# =====================================================================
def _make_verdict(prob: float, conf: float) -> str:
    if conf < 0.65 or (0.35 <= prob <= 0.65):
        return "HUMAN_REVIEW (模型拿不准，转人工确认)"
    return "YES (自動判定: 要調査)" if prob > 0.65 else "NO (自動判定: 不要)"


def _infer_team_mock(ticket: dict[str, Any]) -> str:
    text = f"{ticket.get('component', '')} {ticket.get('summary', '')} {ticket.get('description', '')}".lower()
    if any(k in text for k in ("frontend", "ui", "画面", "css", "ボタン", "ブラウザ", "タイポ", "文言")):
        return "frontend_team"
    if any(k in text for k in ("infra", "ネットワーク", "サーバーダウン", "クラウド", "インフラ", "dns", "vpc", "k8s")):
        return "infra_team"
    if any(k in text for k in ("backend", "api", "決済", "payment", "データベース", "db", "権限", "バッチ", "帳票", "report", "pdf", "税", "割引", "計算")):
        return "backend_team"
    return "other"


class JiraDecisionEngine:
    def __init__(self, kb: JiraKnowledgeBase, api_key: str | None = None):
        self.kb = kb
        self.api_key = api_key.strip() if api_key else None

    def _resolve_api_key(self, override_key: str | None = None) -> str | None:
        key = override_key if override_key is not None else (self.api_key or os.environ.get("GEMINI_API_KEY"))
        return key.strip() if key and key.strip() else None

    def _evaluate_mock(
        self,
        new_ticket: dict[str, Any],
        scored_cases: list[tuple[dict[str, Any], float]],
    ) -> tuple[float, float, float, float, str, float]:
        top_case, top_score = scored_cases[0] if scored_cases else (None, 0.0)
        if top_case and top_score >= 0.50:
            labels = top_case["labels"]
            same_prob = 0.91 if labels["needs_same_case_inv"] else 0.08
            sim_prob = 0.91 if labels["needs_similar_case_inv"] else 0.08
            same_conf, sim_conf = 0.90, 0.90
        else:
            # 无高相似历史判例时，模型主动降低置信度以触发 HUMAN_REVIEW 分流
            same_prob, same_conf = 0.52, 0.55
            sim_prob, sim_conf = 0.58, 0.59

        team = _infer_team_mock(new_ticket)
        return same_prob, same_conf, sim_prob, sim_conf, team, 0.88

    def evaluate_ticket(self, new_ticket: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
        similar_cases = self.kb.retrieve_similar_cases(new_ticket, top_k=3)
        scored_cases = [(case, self.kb.similarity_score(new_ticket, case)) for case in similar_cases]

        effective_key = self._resolve_api_key(api_key)
        if effective_key:
            client = genai.Client(api_key=effective_key, http_options={"api_version": "v1beta"})
            shared_input = json.dumps(
                {
                    "historical_reference_tickets": similar_cases,
                    "target_new_jira_ticket": new_ticket,
                },
                ensure_ascii=False,
                indent=2,
            )
            result = client.decisions.create(
                model="gemini-decision-1",
                input=shared_input,
                questions=DECISION_QUESTIONS,
            )
            answers = result.answers or {}
            same_ans = answers["needs_same_case_inv"]
            sim_ans = answers["needs_similar_case_inv"]
            team_ans = answers["assigned_team"]
            same_prob, same_conf = same_ans.noul, same_ans.confidence
            sim_prob, sim_conf = sim_ans.noul, sim_ans.confidence
            top_team, team_conf = team_ans.choice, team_ans.confidence
            mode = "gemini-decision-1"
        else:
            same_prob, same_conf, sim_prob, sim_conf, top_team, team_conf = self._evaluate_mock(
                new_ticket, scored_cases
            )
            mode = "mock"

        return {
            "mode": mode,
            "ticket_key": new_ticket["key"],
            "same_case_investigation": {
                "verdict": _make_verdict(same_prob, same_conf),
                "prob_yes": round(same_prob, 3),
                "confidence": round(same_conf, 3),
            },
            "similar_case_investigation": {
                "verdict": _make_verdict(sim_prob, sim_conf),
                "prob_yes": round(sim_prob, 3),
                "confidence": round(sim_conf, 3),
            },
            "routing": {
                "top_team": top_team,
                "confidence": round(team_conf, 3),
            },
            "referenced_cases": [c["key"] for c in similar_cases],
            "referenced_case_details": [
                {
                    "key": case["key"],
                    "component": case["component"],
                    "severity": case["severity"],
                    "summary": case["summary"],
                    "labels": case["labels"],
                    "note": case.get("note", ""),
                    "similarity": round(score, 3),
                }
                for case, score in scored_cases
            ],
        }


# =====================================================================
# 3. 完整业务流演示：初次判断 -> 外部系统人工反馈 -> 再次判断类似新票
# =====================================================================
if __name__ == "__main__":
    if DB_FILE.exists():
        DB_FILE.unlink()

    kb = JiraKnowledgeBase(DB_FILE)
    engine = JiraDecisionEngine(kb)

    ticket_a = {
        "key": "PROJ-205",
        "component": "帳票出力モジュール (Report)",
        "severity": "Minor",
        "summary": "請求書PDFの消費税端数処理が一部の取引先で1円ずれる",
        "description": "外貨建て取引の請求書PDF出力時のみ、切り捨てではなく四捨五入になっている。",
    }

    print("=== Step 1: 新 Jira 票 (PROJ-205) 初次判定 ===")
    decision_1 = engine.evaluate_ticket(ticket_a)
    print(json.dumps(decision_1, ensure_ascii=False, indent=2))

    kb.add_feedback(
        ticket=ticket_a,
        confirmed_same=False,
        confirmed_similar=True,
        reason="帳票PDFは都度生成のためDB同件調査は不要だが、見積書や納品書でも同じ端数計算関数を使っている可能性が高いため類似調査のみ必須とする社内ルール。",
    )

    ticket_b = {
        "key": "PROJ-310",
        "component": "帳票出力モジュール (Report)",
        "severity": "Minor",
        "summary": "納品書PDFの割引額計算で小数点以下の丸め誤差が発生",
        "description": "割引率に小数を含む場合、PDF表示上の合計金額が1円ずれることがある。",
    }

    print("\n=== Step 2: 注入反馈后，对另一张类似新票 (PROJ-310) 进行判定 ===")
    decision_2 = engine.evaluate_ticket(ticket_b)
    print(json.dumps(decision_2, ensure_ascii=False, indent=2))

    assert "PROJ-205" in decision_2["referenced_cases"], "PROJ-205 应作为最相似参考判例被检索出来"
    if decision_2["mode"] == "mock":
        assert decision_1["same_case_investigation"]["verdict"].startswith("HUMAN_REVIEW")
        assert decision_2["same_case_investigation"]["verdict"].startswith("NO")
        assert decision_2["similar_case_investigation"]["verdict"].startswith("YES")
