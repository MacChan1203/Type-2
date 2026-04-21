# Type-2

Type-2 は、Ollama 上のローカルLLMを CrewAI でつなぎ、生成・評価・改善を小さなループで回す実験用エージェントシステムです。

## 構成

```
Type-2/
├── main.py                          # CLI とパイプライン本体
├── type_defs.py                     # TypedDict 定義
├── config.py                        # .env から読み込む設定
├── agents/
│   ├── crew_setup.py                # CrewAI の Agent / Task 定義
│   └── prompts.py                   # 各エージェントのプロンプト
├── evaluation/
│   ├── code_scorer.py               # ルールベースのコード品質スコア
│   ├── execution_claims.py          # 実行主張（URL証拠なし）の検出と減点
│   ├── merge_reviews.py             # 複数 Critic の重み付き統合
│   ├── python_verifier.py           # Python 構文検証（ast.parse）
│   ├── reverse_fallback.py          # 逆タスク用ローカルフォールバックコード
│   ├── reverse_metrics.py           # 逆タスクのルールベーススコア
│   ├── reverse_normalize.py         # 逆タスク出力の正規化・ノイズ除去
│   ├── scorer.py                    # JSON パース・fallback_review
│   └── thresholds.py                # スコア閾値定数
├── memory/
│   └── chroma_store.py              # ChromaDB による成功・失敗記憶
├── tools/
│   ├── external_requests.py         # 専用ハンドラのない外部実行依頼を失敗扱い
│   ├── hacker_news.py               # Hacker News 取得・翻訳
│   └── router.py                    # ToolRouter（スレッドセーフシングルトン）
├── tests/
│   ├── test_evaluation.py
│   ├── test_pipeline.py
│   ├── test_reverse.py
│   └── test_tools.py
└── utils/
    ├── helpers.py
    ├── json_utils.py                # JSON ブロック抽出ユーティリティ
    ├── prompt_safety.py             # プロンプトインジェクション対策
    ├── run_logger.py                # 実行結果を runs/ に JSON 保存
    └── safe_http.py                 # SSRF 対策付き HTTP クライアント
```

## セットアップ

```bash
brew install ollama
ollama serve
ollama pull qwen3
python3 -m venv .Type-2
. .Type-2/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

必要に応じて `.env` の `OLLAMA_MODEL` を手元のモデル名に変更してください。
`ollama serve` は別ターミナルで起動したままにしてください。
ChromaDB の embedding は現在 `CHROMA_EMBEDDING=default` のみ対応しています。

## 実行

```bash
.Type-2/bin/python main.py --task "PythonでFizzBuzzを書いて"
```

ログを完全に抑えて最終結果だけ出したい場合:

```bash
.Type-2/bin/python main.py --task "PythonでFizzBuzzを書いて" --quiet
```

目標分析・反省サイクルを使う場合:

```bash
.Type-2/bin/python main.py --mode reflective --task "PythonでFizzBuzzを書いて"
```

`--mode agi` は互換用の古いエイリアスです。

逆タスクの例:

```bash
.Type-2/bin/python main.py --task "わざとバグを含むPythonのCSV集計コードを書いて"
```

Hacker News のトップ記事を取得して日本語で表示する例:

```bash
.Type-2/bin/python main.py --task "Hacker Newsの最初のニュースを日本語にして表示して"
```

## パイプライン

```
入力タスク
  └─ ToolRouter ──(専用ハンドラがある)──→ 直接実行して返す
       │
       └─(なし)──→ IntentChecker（逆タスク判定）
                      └─ Worker（コード生成）
                           └─ IntentCritic + ReverseCritic or QualityCritic（並列評価）
                                └─ merge_reviews（重み付きスコア統合）
                                     ├─(合格)──→ 完了
                                     └─(不合格)──→ Improver → 次の試行
```

逆タスクでは QualityCritic を呼ばず、ReverseCritic・ルールベーススコア・IntentCritic の3系統で評価します。

## ToolRouter

`tools/router.py` の `ToolRouter` に登録された実行ツールが、LLM呼び出し前に優先してマッチングされます。

現在登録済みのツール:

| ツール名 | 処理内容 |
|---|---|
| `hacker_news_top_japanese` | Hacker News API から記事を取得し日本語で表示 |
| `unsupported_external_execution` | 専用ハンドラのない外部実行依頼を失敗として返す |

新しいツールを追加する場合は `ToolSpec` を作成して `build_default_router()` に追加します。

## 評価の考え方

通常タスクでは、指示遵守・実用性・コード品質を重み付きで統合します。
逆タスクでは、欠陥の多様性・コードとしての構造・説明に逃げていないかを専用の Critic とルールベーススコアで評価します。

外部取得・API呼び出し・現在情報などを成功したように述べる場合、実URL・取得ID・実行時刻など検証可能な証拠が必要です。
証拠なしの主張は `unsupported_execution_claim` として大きく減点されます。

## 記憶

ChromaDB は `data/chroma` に永続化されます（保存先は `.env` の `CHROMA_PATH` で変更可）。

記憶として保存されるのは出力全文ではなく、次回プロンプトへ混ぜやすい要約・レビュー情報のみです。

- 強み・弱点はそれぞれ上位3件・各120文字に絞って保存
- プロンプトへ渡す際は1件あたり800文字に切り詰め

記憶の管理:

```bash
# 全記憶を削除
.Type-2/bin/python main.py --clear-memory

# 古い実行ログを削除（最新10件だけ残す）
.Type-2/bin/python main.py --prune-runs 10

# 実行ログの集計を表示
.Type-2/bin/python main.py --stats
```

## 実行ログ

各実行の結果は `runs/` に JSON で保存されます。含まれる情報:

- 入力タスク・モード・モデル名・実行時刻
- ToolRouter で直接実行した場合のツール名
- 最終出力・計画・意図・レビュー

## テスト

```bash
.Type-2/bin/python -m unittest discover -s tests
```

## 注意

逆タスクでは意図的に壊れたコードや危険な処理を含むコードが生成されることがあります。生成物をそのまま実行せず、レビュー教材として扱ってください。
Type-2 は自律的な意思、現実世界での継続的な行動能力、自己改変能力を持つAGIではありません。
