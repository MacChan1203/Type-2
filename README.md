# Type-2

Type-2 は、Ollama 上のローカルLLMを CrewAI でつなぎ、生成・評価・改善を小さなループで回す実験用エージェントシステムです。

## 構成

- `main.py`: CLI とパイプライン本体
- `agents/crew_setup.py`: CrewAI の Agent / Task 定義
- `agents/prompts.py`: Planner / Worker / Critic / Improver のプロンプト
- `tools/`: Hacker News 取得など、LLMではなく実際に実行する処理と ToolRouter
- `evaluation/`: JSON パース、品質スコア、逆タスクスコア、レビュー統合
- `memory/chroma_store.py`: ChromaDB による成功・失敗記憶
- `utils/run_logger.py`: 実行結果を `runs/` にJSON保存するロガー
- `config.py`: `.env` から読み込む設定

## セットアップ

```bash
brew install ollama
ollama serve
ollama pull gemma4:e4b
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

`--quiet` は、内部の Planner / Worker / Critic ログだけでなく、下位処理が標準出力や標準エラーへ書いたノイズも吸収し、最後の成果物だけを表示します。

AGI風の目標分析・反省サイクルを使う場合:

```bash
.Type-2/bin/python main.py --mode agi --task "PythonでFizzBuzzを書いて"
```

Hacker News のトップ記事を取得して日本語で表示する例:

```bash
.Type-2/bin/python main.py --mode agi --task "午後8時54分になったら、Hacker Newsの最初のニュースを日本語にして表示して"
```

この種類の外部取得タスクは、LLMに「取得したふり」をさせず、Hacker News API と記事URLを直接取得する専用ハンドラで処理します。
指定時刻が未来の場合はその時刻まで待機し、すでに過ぎている場合は即時実行します。
本文取得や翻訳にはネットワーク接続と Ollama の起動が必要です。

## ToolRouter

Type-2 は、実行できる処理を `tools/router.py` の `ToolRouter` に登録します。
現在は `hacker_news_top_japanese` が登録済みです。
新しい実行ツールを足す場合は、次の3点を持つ `ToolSpec` を追加します。

- `name`: ツール名
- `can_handle(task)`: その依頼を処理できるか判定する関数
- `run(task, verbose)`: 実際に処理して文字列を返す関数

これにより、`main.py` は個別ツールの詳細を知らず、対応できる依頼だけを実行ツールへ渡します。

逆タスクの例:

```bash
.Type-2/bin/python main.py --task "わざとバグを含むPythonのCSV集計コードを書いて"
```

## 評価の考え方

通常タスクでは、指示遵守・実用性・構造・安全性を評価します。
逆タスクでは、通常評価と混ざらないように、欠陥の多様性・コードとしてのまとまり・説明に逃げていないかを評価します。
外部サイト取得、APIアクセス、現在情報、時刻実行などを成功したように述べる場合は、実URL、取得ID、実行時刻、失敗理由などの検証可能な証拠が必要です。
証拠なしの「取得しました」「実行しました」は `unsupported_execution_claim` として大きく減点されます。

## 実行ログ

Type-2 は各実行の最終結果を `runs/` にJSONで保存します。
保存先は `.env` の `RUNS_PATH` で変更できます。
ログには次の情報が含まれます。

- 入力タスク、モード、モデル名、実行時刻
- ToolRouterで直接実行した場合のツール名
- 最終出力
- 計画、意図、レビュー、補助メタデータ

このログは、あとから失敗原因や改善履歴を追うための運用記録です。

## AGI Mode

`--mode agi` は、真のAGIではありません。Type-2 の通常パイプラインに、次の制御ループを重ねる実験モードです。

1. 目標分析
2. 前提・制約・リスクの整理
3. Worker への具体指示生成
4. 実行
5. Critic と Reflector による評価
6. 必要なら次サイクルで改善

このモードは、長めの調査、設計、複数条件を満たす生成タスクに向いています。単純なコードスニペット生成では通常モードの方が安定する場合があります。

## 記憶

ChromaDB は `data/chroma` に永続化されます。保存される記憶は出力全文ではなく、次回のプロンプトに混ぜても扱いやすい短い要約とレビュー情報です。

## テスト

```bash
.Type-2/bin/python -m unittest discover -s tests
```

## 注意

逆タスクでは意図的に壊れたコードや危険な処理を含むコードが生成されることがあります。生成物をそのまま実行せず、レビュー教材として扱ってください。
Type-2 は自律的な意思、現実世界での継続的な行動能力、自己改変能力を持つAGIではありません。
