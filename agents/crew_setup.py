from __future__ import annotations

from functools import lru_cache
import os


def _configure_crewai_environment() -> None:
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    # TYPE2_CREWAI_HOME が明示されている場合だけ HOME を上書きする。
    # 以前は常に HOME を CWD 配下へ差し替えていたため、同一プロセス内の
    # 他コンポーネントが影響を受けていた。
    override = os.environ.get("TYPE2_CREWAI_HOME")
    if override:
        os.environ["HOME"] = override


_configure_crewai_environment()


from crewai import Agent, Task, Crew, Process, LLM
from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages

from config import settings
from agents.prompts import (
    PLANNER_SYSTEM_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REFLECTIVE_CONTROLLER_SYSTEM_PROMPT,
    REFLECTIVE_REFLECTOR_SYSTEM_PROMPT,
    INTENT_CHECK_PROMPT,
    INTENT_CRITIC_SYSTEM_PROMPT,
    QUALITY_CRITIC_SYSTEM_PROMPT,
    REVERSE_QUALITY_CRITIC_SYSTEM_PROMPT,
    REVERSE_CRITIC_SYSTEM_PROMPT,
    IMPROVER_SYSTEM_PROMPT,
)
from utils.prompt_safety import INJECTION_DISCLAIMER, wrap_untrusted


set_suppress_tracing_messages(True)


@lru_cache(maxsize=1)
def build_llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
    )


_agent_cache: dict[str, Agent] = {}


def reset_agent_caches() -> None:
    """テスト時や設定差し替え時に LLM/Agent の内部キャッシュを初期化する。"""
    build_llm.cache_clear()
    _agent_cache.clear()


def build_agent(agent_key: str) -> Agent:
    if agent_key in _agent_cache:
        return _agent_cache[agent_key]
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
        "reflective_controller": {
            "role": "ReflectiveController",
            "goal": "目標を安全で実行可能な認知サイクルに分解する",
            "backstory": "真のAGIではないが、目標分析、制約整理、実行方針の作成を担当する制御役。",
            "system_template": REFLECTIVE_CONTROLLER_SYSTEM_PROMPT,
        },
        "reflective_reflector": {
            "role": "ReflectiveReflector",
            "goal": "出力を検証し、次の改善方針を決める",
            "backstory": "過去タスク混入や目的逸脱を検出し、次サイクルに必要な最小改善を示す反省役。",
            "system_template": REFLECTIVE_REFLECTOR_SYSTEM_PROMPT,
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
        "intent_checker": {
            "role": "IntentChecker",
            "goal": "ユーザー指示の意図を正確に分類し、逆タスクかどうかを判定する",
            "backstory": "指示の本来の目的を見抜くことに特化したAI。",
            "system_template": INTENT_CHECK_PROMPT,
        },
    }

    if agent_key not in specs:
        raise ValueError(f"不明なエージェントキー: {agent_key}")

    agent = Agent(
        llm=llm,
        verbose=False,
        allow_delegation=False,
        **specs[agent_key],
    )
    _agent_cache[agent_key] = agent
    return agent



def run_planner(task_text: str, memory_text: str) -> str:
    planner = build_agent("planner")

    planning_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の依頼を達成するための実行計画を作成してください。\n\n"
            f"依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"参考記憶:\n{wrap_untrusted(memory_text, 'memory')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def build_worker_description(
    task_text: str,
    memory_text: str,
    plan_text: str,
    is_reverse: bool = False,
) -> str:
    wrapped_task = wrap_untrusted(task_text, "task")
    wrapped_memory = wrap_untrusted(memory_text, "memory")
    wrapped_plan = wrap_untrusted(plan_text, "plan")
    if is_reverse:
        return f"""
{INJECTION_DISCLAIMER}

このタスクは「意図的に欠陥を含む壊れたPythonコード」を作ることです。
正しいコードを書くことは禁止です。

【必須条件】
- 実行時エラー・論理バグ・状態管理バグ・例外処理の欠陥・セキュリティ問題を少なくとも各1つ含める
- Pythonの構文として成立させるが、実行すると誤動作や例外が起きるようにする
- コメントでバグを説明・ラベリングしない（通常の開発者コメント程度にする）
- トップレベルで危険な副作用（ファイル操作・外部コマンド）を実行しない

【出力形式】
Pythonコード本文のみ。説明文・前置き・Markdownコードブロック(```)は不要。

依頼:
{wrapped_task}

参考記憶:
{wrapped_memory}

参考計画:
{wrapped_plan}
""".strip()

    return f"""
{INJECTION_DISCLAIMER}

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
{wrapped_task}

参考記憶:
{wrapped_memory}

参考計画:
{wrapped_plan}
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_reflective_controller(task_text: str, memory_text: str) -> str:
    controller = build_agent("reflective_controller")

    controller_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の依頼を反省モードの認知サイクルとして整理してください。\n\n"
            f"依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"参考記憶:\n{wrap_untrusted(memory_text, 'memory')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_reflective_reflector(
    task_text: str,
    plan_text: str,
    output_text: str,
    review_text: str,
) -> str:
    reflector = build_agent("reflective_reflector")

    reflection_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の反省モードサイクル結果を評価してください。\n\n"
            f"元の依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"計画:\n{wrap_untrusted(plan_text, 'plan')}\n\n"
            f"成果物:\n{wrap_untrusted(output_text, 'output')}\n\n"
            f"既存レビュー:\n{wrap_untrusted(review_text, 'review')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_intent_checker(task_text: str) -> str:
    intent_agent = build_agent("intent_checker")

    intent_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下のユーザーの指示の意図を分析してください。\n\n"
            f"指示:\n{wrap_untrusted(task_text, 'task')}"
        ),
        agent=intent_agent,
        expected_output=(
            'JSON: {"intent": "...", "is_reverse_task": true/false, "success_criteria": "..."}'
        ),
    )

    crew = Crew(
        agents=[intent_agent],
        tasks=[intent_task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()


def run_intent_critic(task_text: str, output_text: str) -> str:
    intent_critic = build_agent("intent_critic")

    critic_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の成果物が、ユーザーの依頼にどれだけ忠実かを評価してください。\n\n"
            f"依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"成果物:\n{wrap_untrusted(output_text, 'output')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_quality_critic(task_text: str, output_text: str, is_reverse: bool = False) -> str:
    agent_key = "reverse_quality_critic" if is_reverse else "quality_critic"
    critic = build_agent(agent_key)
    subject = "逆タスク成果物の品質" if is_reverse else "成果物の品質"

    critic_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の{subject}を評価してください。\n\n"
            f"依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"成果物:\n{wrap_untrusted(output_text, 'output')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_reverse_critic(task_text: str, output_text: str) -> str:
    reverse_critic = build_agent("reverse_critic")

    critic_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の成果物が、逆タスクとしてどれだけ適切に壊れているか評価してください。\n\n"
            f"依頼:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"成果物:\n{wrap_untrusted(output_text, 'output')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()

def run_improver(task_text: str, review_text: str) -> str:
    improver = build_agent("improver")

    improver_task = Task(
        description=(
            f"{INJECTION_DISCLAIMER}\n\n"
            f"以下の元タスクと評価結果を読み、次回の改善方針を決めてください。\n\n"
            f"元タスク:\n{wrap_untrusted(task_text, 'task')}\n\n"
            f"評価結果:\n{wrap_untrusted(review_text, 'review')}\n\n"
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
        tracing=False,
    )

    result = crew.kickoff()
    return str(result).strip()
