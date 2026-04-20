from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Task, Crew, Process, LLM

from config import settings
from agents.prompts import (
    PLANNER_SYSTEM_PROMPT,
    WORKER_SYSTEM_PROMPT,
    AGI_CONTROLLER_SYSTEM_PROMPT,
    AGI_REFLECTOR_SYSTEM_PROMPT,
    INTENT_CHECK_PROMPT,
    INTENT_CRITIC_SYSTEM_PROMPT,
    QUALITY_CRITIC_SYSTEM_PROMPT,
    REVERSE_QUALITY_CRITIC_SYSTEM_PROMPT,
    REVERSE_CRITIC_SYSTEM_PROMPT,
    IMPROVER_SYSTEM_PROMPT,
)


def build_llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url="http://localhost:11434",
    )

def build_agent(agent_key: str) -> Agent:
    llm = build_llm()

    specs = {
        "planner": {
            "role": "Planner",
            "goal": "タスクを安全かつ明確な手順に分解する",
            "backstory": "計画立案に長けたAI。失敗の再発防止を重視する。",
            "system_template": PLANNER_SYSTEM_PROMPT,
        },
        "worker": {
            "role": "Worker",
            "goal": "与えられた意図に沿って成果物を作成する",
            "backstory": "通常タスクでも逆タスクでも、与えられた意図を最優先に実行するAI。",
            "system_template": WORKER_SYSTEM_PROMPT,
        },
        "agi_controller": {
            "role": "AGIController",
            "goal": "目標を安全で実行可能な認知サイクルに分解する",
            "backstory": "真のAGIではないが、目標分析、制約整理、実行方針の作成を担当する制御役。",
            "system_template": AGI_CONTROLLER_SYSTEM_PROMPT,
        },
        "agi_reflector": {
            "role": "AGIReflector",
            "goal": "出力を検証し、次の改善方針を決める",
            "backstory": "過去タスク混入や目的逸脱を検出し、次サイクルに必要な最小改善を示す反省役。",
            "system_template": AGI_REFLECTOR_SYSTEM_PROMPT,
        },
        "intent_critic": {
            "role": "IntentCritic",
            "goal": "指示遵守だけを厳しく評価する",
            "backstory": "依頼と出力の一致度だけを見る厳格な審査役。",
            "system_template": INTENT_CRITIC_SYSTEM_PROMPT,
        },
        "quality_critic": {
            "role": "QualityCritic",
            "goal": "出力品質を評価する",
            "backstory": "成果物の質と構造を評価する査読者。",
            "system_template": QUALITY_CRITIC_SYSTEM_PROMPT,
        },
        "reverse_quality_critic": {
            "role": "ReverseQualityCritic",
            "goal": "逆タスクの出力品質を評価する",
            "backstory": "壊れた成果物の教材価値と欠陥の多様性を評価する査読者。",
            "system_template": REVERSE_QUALITY_CRITIC_SYSTEM_PROMPT,
        },
        "reverse_critic": {
            "role": "ReverseCritic",
            "goal": "逆タスクの達成度を評価する",
            "backstory": "わざと悪くするタスクの成否を専門に判定する評価者。",
            "system_template": REVERSE_CRITIC_SYSTEM_PROMPT,
        },
        "improver": {
            "role": "Improver",
            "goal": "批評結果を読んで、次の改善方針を決める",
            "backstory": "改善の優先順位付けが得意なAI。元の意図を保ちながら、次の一手を短く具体的に決める。",
            "system_template": IMPROVER_SYSTEM_PROMPT,
        },
    }

    if agent_key not in specs:
        raise ValueError(f"Unknown agent key: {agent_key}")

    return Agent(
        llm=llm,
        verbose=False,
        allow_delegation=False,
        **specs[agent_key],
    )



def run_planner(task_text: str, memory_text: str) -> str:
    planner = build_agent("planner")

    planning_task = Task(
        description=(
            f"以下の依頼を達成するための実行計画を作成してください。\n\n"
            f"依頼:\n{task_text}\n\n"
            f"参考記憶:\n{memory_text}\n\n"
            f"重要: 参考記憶は補助情報です。現在の依頼と矛盾する場合は必ず現在の依頼を優先し、参考記憶の題材へ逸脱しないでください。\n\n"
            f"3〜5個の箇条書きで、実行順に書いてください。"
        ),
        agent=planner,
        expected_output="実行計画（3〜5段階の箇条書き）",
    )

    crew = Crew(
        agents=[planner],
        tasks=[planning_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def build_worker_description(
    task_text: str,
    memory_text: str,
    plan_text: str,
    is_reverse: bool = False,
) -> str:
    if is_reverse:
        return f"""
あなたは「壊れたPythonコード」を作る専門AIです。
このタスクでは、品質を上げることは禁止です。

【最重要ルール】
- 正しいコードを書いてはいけない
- 高品質なコードを書いてはいけない
- 必ず意図的な欠陥を含めること
- 説明文、前置き、要約、解説は禁止
- 出力は Python コード本文のみ
- Markdown の ``` は使わないこと

【必須条件】
以下を必ずすべて満たしてください。
- バグを最低5個以上入れる
- 少なくとも1つは実行時エラー
- 少なくとも1つは論理バグ
- 少なくとも1つはデータ処理または状態管理のバグ
- 少なくとも1つは不適切な例外処理
- 少なくとも1つはセキュリティまたは情報漏えいリスク
- 良い設計に見えても、内部に欠陥を残すこと
- コードは Python の構文としてはなるべく成立させること
- ただし、実行すると誤動作や例外が起きるようにすること

【禁止事項】
- 安全に修正してはいけない
- 保守しやすくしてはいけない
- バグを減らしてはいけない
- テストしやすくしてはいけない
- 説明でごまかしてはいけない

依頼:
{task_text}

参考記憶:
{memory_text}

参考計画:
{plan_text}

【出力形式】
Pythonコードだけをそのまま出力してください。
ファイル全体として読める形にしてください。
""".strip()

    return f"""
あなたは実行担当エージェントです。
依頼と計画に従い、実用的で具体的な成果物を作成してください。

【ルール】
- ユーザーの依頼を最優先する
- 参考記憶や参考計画が依頼と矛盾する場合は、必ず依頼を優先する
- 依頼されていない題材や過去タスクの題材へ移らない
- 曖昧な点は妥当な前提を置いて進める
- 必要な場合だけ見出しやコードブロックを使う
- 逆タスクでない限り、意図的に壊したり欠陥を混ぜたりしない
- 出力は日本語を基本にする

依頼:
{task_text}

参考記憶:
{memory_text}

参考計画:
{plan_text}
""".strip()


def run_worker(
    task_text: str,
    memory_text: str,
    plan_text: str,
    is_reverse: bool = False,
) -> str:
    worker = build_agent("worker")

    worker_task = Task(
        description=build_worker_description(
            task_text=task_text,
            memory_text=memory_text,
            plan_text=plan_text,
            is_reverse=is_reverse,
        ),
        agent=worker,
        expected_output=(
            "意図的に複数の欠陥を含むPythonコード本文のみ"
            if is_reverse
            else "ユーザー依頼に沿った成果物"
        ),
    )

    crew = Crew(
        agents=[worker],
        tasks=[worker_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_agi_controller(task_text: str, memory_text: str) -> str:
    controller = build_agent("agi_controller")

    controller_task = Task(
        description=(
            f"以下の依頼をAGI-modeの認知サイクルとして整理してください。\n\n"
            f"依頼:\n{task_text}\n\n"
            f"参考記憶:\n{memory_text}\n\n"
            f"参考記憶は補助情報です。現在の依頼と矛盾する場合は無視してください。\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=controller,
        expected_output=(
            'JSON: {"objective": "...", "assumptions": [...], "subgoals": [...], '
            '"risks": [...], "success_criteria": [...], "worker_instruction": "..."}'
        ),
    )

    crew = Crew(
        agents=[controller],
        tasks=[controller_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_agi_reflector(
    task_text: str,
    plan_text: str,
    output_text: str,
    review_text: str,
) -> str:
    reflector = build_agent("agi_reflector")

    reflection_task = Task(
        description=(
            f"以下のAGI-modeサイクル結果を評価してください。\n\n"
            f"元の依頼:\n{task_text}\n\n"
            f"計画:\n{plan_text}\n\n"
            f"成果物:\n{output_text}\n\n"
            f"既存レビュー:\n{review_text}\n\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=reflector,
        expected_output=(
            'JSON: {"score": 0-100, "done": true/false, "strengths": [...], '
            '"weaknesses": [...], "next_instruction": "..."}'
        ),
    )

    crew = Crew(
        agents=[reflector],
        tasks=[reflection_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_intent_checker(task_text: str) -> str:
    planner = build_agent("planner")

    intent_task = Task(
        description=(
            f"{INTENT_CHECK_PROMPT}\n\n"
            f"以下のユーザーの指示の意図を分析してください。\n\n"
            f"指示:\n{task_text}\n\n"
            f"以下のJSON形式で返してください:\n"
            f'{{"intent": "...", "is_reverse_task": true/false, "success_criteria": "..."}}'
        ),
        agent=planner,
        expected_output="JSON形式の意図分析",
    )

    crew = Crew(
        agents=[planner],
        tasks=[intent_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_intent_critic(task_text: str, output_text: str) -> str:
    intent_critic = build_agent("intent_critic")

    critic_task = Task(
        description=(
            f"以下の成果物が、ユーザーの依頼にどれだけ忠実かを評価してください。\n\n"
            f"依頼:\n{task_text}\n\n"
            f"成果物:\n{output_text}\n\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=intent_critic,
        expected_output=(
            'JSON: {"score": 0-100, "strengths": [...], "weaknesses": [...], '
            '"next_action": "..."}'
        ),
    )

    crew = Crew(
        agents=[intent_critic],
        tasks=[critic_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_quality_critic(task_text: str, output_text: str, is_reverse: bool = False) -> str:
    agent_key = "reverse_quality_critic" if is_reverse else "quality_critic"
    critic = build_agent(agent_key)
    subject = "逆タスク成果物の品質" if is_reverse else "成果物の品質"

    critic_task = Task(
        description=(
            f"以下の{subject}を評価してください。\n\n"
            f"依頼:\n{task_text}\n\n"
            f"成果物:\n{output_text}\n\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=critic,
        expected_output=(
            'JSON: {"score": 0-100, "strengths": [...], "weaknesses": [...], '
            '"next_action": "..."}'
        ),
    )

    crew = Crew(
        agents=[critic],
        tasks=[critic_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_reverse_critic(task_text: str, output_text: str) -> str:
    reverse_critic = build_agent("reverse_critic")

    critic_task = Task(
        description=(
            f"以下の成果物が、逆タスクとしてどれだけ適切に壊れているか評価してください。\n\n"
            f"依頼:\n{task_text}\n\n"
            f"成果物:\n{output_text}\n\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=reverse_critic,
        expected_output=(
            'JSON: {"score": 0-100, "strengths": [...], "weaknesses": [...], '
            '"next_action": "..."}'
        ),
    )

    crew = Crew(
        agents=[reverse_critic],
        tasks=[critic_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_improver(task_text: str, review_text: str) -> str:
    improver = build_agent("improver")

    improver_task = Task(
        description=(
            f"以下の元タスクと評価結果を読み、次回の改善方針を決めてください。\n\n"
            f"元タスク:\n{task_text}\n\n"
            f"評価結果:\n{review_text}\n\n"
            f"必ずJSONのみで返してください。"
        ),
        agent=improver,
        expected_output=(
            'JSON: {"priority_fixes": [...], "keep": [...], "improved_prompt": "..."}'
        ),
    )

    crew = Crew(
        agents=[improver],
        tasks=[improver_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result).strip()
