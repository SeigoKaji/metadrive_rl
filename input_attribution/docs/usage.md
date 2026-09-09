# 実行方法

解析用設定は学習用 TOML へ追加せず、`input_attribution/configs/` の解析設定として管理します。既存の仮想環境の Python を使い、SB3、PyTorch、MetaDrive を無条件に更新しません。一般手順では利用する環境の Python を `PYTHON` に設定し、設定の `[model].path` を手元の学習済みモデルへ合わせます。259 の公式接続では実モデルと実環境が必要です。MetaDriveなしの fake adapter では接続契約と保存形式だけを検証します。`/home/seigo/workspace/metadrive_rl/metadrive-rl/.venv/bin/python3` と `/home/seigo/workspace/metadrive_rl/metadrive-rl-input-attribution/models/official_baseline.zip`（SHA256: `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`）は過去の実走行検証環境を記録した値で、一般手順の固定パスではありません。

## 最短の実行

Linux/Ubuntu（既存環境をそのまま使用）:

```bash
PYTHON=/path/to/existing/python
$PYTHON -m input_attribution check --config input_attribution/configs/input_attribution_official_259_variants.toml
$PYTHON -m input_attribution run --config input_attribution/configs/input_attribution_official_259_variants.toml
```

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m input_attribution check --config input_attribution/configs/input_attribution_official_259_variants.toml
.\.venv\Scripts\python.exe -m input_attribution run --config input_attribution/configs/input_attribution_official_259_variants.toml
```

`run` は check、通常走行収集、①-A、設定で指定した①-B、主レポート生成を順に行います。`run` では③ IG は常に未実行（skipped）として残ります。設定の `[ig].enabled` は Captum 依存の利用可否を check するためだけの項目で、実際の IG は下記の `ig` サブコマンドを対象 episode/step と baseline とともに明示して実行します。check はモデル、設定、adapter、schema、入力次元・dtype・範囲・全 index の被覆、離散 Action、前処理、代表確率再現、対象レーン、パターンの解決状態を人間向けと JSON で出します。`input_attribution/schemas/custom_262_template.json` に未確定フィールドが残る設定は成功扱いにしません。

## 終了コードと失敗の扱い

CLIの終了コードは、0を要求した処理が完了した状態、1を必要な実験または検証が失敗した状態、2をparserの構文・引数エラー（またはトップレベルで分類できないCLIエラー）として扱います。`run` と `closed-loop` は標準出力・標準エラーと `status.json` に `completed`、`failed`、`aborted`、`runtime_failure` の件数を残します。pattern内の runtime failure は終了コード1になり、他のpatternで保存できた結果と失敗phase・理由・途中件数は同じrunへ保存されます。

到達、衝突、道路逸脱、環境仕様上の自然終了は、実行が正常に進んだ記録として `completed` に含めます。no-op と明示条件外のskipもruntime failureには数えません。主要なmetric summaryは自然終了したepisodeだけを母数にし、failed・aborted・budget-censored episodeの値と実行状態は `diagnostic_metric_summary` と `execution_groups` に分けて保持します。missing telemetryやpartial measurementは0へ補完せず、failure phase、既知の測定件数、全体のunknown/partialとともに表示します。任意のIGが未実行でも、成功したA/Bの結果や終了状態は変更しません。

`report` は保存済みrunを読むだけの再生成です。`--output-dir` の有無にかかわらず、元runの `status.json`、manifest、生データを更新せず、生成先へreport成果物だけを書き出します。失敗したstageの診断を表示するときも、同じrunの古い成功stageへ置き換えません。

## 部分実行と保存結果からの再解析

```bash
# Linux（保存済み通常観測を使うコマンドは環境を起動しない）
PYTHON=/path/to/existing/python
$PYTHON -m input_attribution collect --config input_attribution/configs/input_attribution_official_259_variants.toml
$PYTHON -m input_attribution offline --run-dir outputs/input_attribution/<experiment>/<model>/<run_id>
$PYTHON -m input_attribution closed-loop --run-dir outputs/input_attribution/<experiment>/<model>/<run_id> --patterns P00,P02_heading_neutral
$PYTHON -m input_attribution report --run-dir outputs/input_attribution/<experiment>/<model>/<run_id>

# 旧runを変更せず、表示だけを別ディレクトリへ再生成
$PYTHON -m input_attribution report --run-dir outputs/input_attribution/<experiment>/<model>/<old_run_id> --output-dir /tmp/input-attribution-report-revision

# 新しいA（①-A）設定を保存観測から計算。新しい子runが表示される
$PYTHON -m input_attribution offline --run-dir outputs/input_attribution/<experiment>/<model>/<old_run_id> --config input_attribution/configs/input_attribution_official_259_variants.toml

# 子runに対する新しいB（①-B）は、選択した条件で環境を走り直す
$PYTHON -m input_attribution closed-loop --run-dir outputs/input_attribution/<experiment>/<model>/<child_run_id> --patterns P00,P02_heading_neutral

# 任意 IG（Captum が利用可能な環境だけ）
$PYTHON -m input_attribution ig --run-dir outputs/input_attribution/<experiment>/<model>/<run_id> --episode 0 --steps 12,20 --baseline episode-0:0
```

## 259 preset の主経路と B の選択

新しい259次元の主経路は `input_attribution/configs/input_attribution_official_259_variants.toml` です。source-confirmed な型付き置換を `full_episode` へ適用します。旧 `input_attribution/configs/input_attribution_official_259.toml` と `input_attribution/configs/input_attribution_official_259_legacy_freeze.toml` は、初期 saved-reference 条件を再現するための過去設定です。旧設定を新 preset と同じ条件として扱ったり、既存 run の条件を上書きしたりしません。

新 preset の標準 ①-B は15 patternで、速度・操舵・履歴はそれぞれ `P03_speed_reference`、`P03_steering_reference`、`P03_history_group_reference` を使います。`P03_speed_fixed_level`、`P03_steering_neutral`、`P03_history_group_neutral` などの `full_episode` 版は標準 B には含めず、追加診断として選択した場合だけ実行します。標準 B を259入力すべての全区間評価済みとは解釈しません。

定義済み pattern の実行数を変更せず、追加診断を明示的に選択する例です。

```bash
# 新しい run を作り、選択した B を実行
$PYTHON -m input_attribution run \
  --config input_attribution/configs/input_attribution_official_259_variants.toml \
  --patterns P00,P03_speed_fixed_level,P03_steering_neutral,P03_history_group_neutral

# 保存済み run で選択した B だけを実行
$PYTHON -m input_attribution closed-loop \
  --run-dir outputs/input_attribution/<experiment>/<model>/<run_id> \
  --patterns P00,P03_speed_fixed_level,P03_steering_neutral,P03_history_group_neutral
```

`run --patterns` と `closed-loop --patterns` は実行対象 B の選択です。動画の対象は `[video].patterns` で別に指定します。

PowerShell では `.venv\Scripts\python.exe` と Windows のパス区切りを使います。IG の baseline 指定形式は設定と `--help` の表示を優先します。Captumを追加する場合も、既存環境へ無条件に更新せず、同梱の任意 requirements-ig と独立した仮想環境/一時 wheel を使って互換性を確認します。

`report` は MetaDrive、学習済みモデル、Captum を import せず、run directory に保存された JSON/JSONL/CSV/NPZ と媒体メタデータだけを読みます。出力は次の場所です。

```text
outputs/input_attribution/<experiment>/<model>/<run_id>/
├── manifest.json                 # 実行・hash・seed・コマンド
├── resolved_config.json
├── status.json                   # 完了/失敗/未実行
├── 00_reference/                 # 観測と通常走行
├── 01_offline/                   # ①-A
├── 02_closed_loop/               # ①-B
├── 03_ig/                        # ③（任意）
├── report.html                   # 外部 CDN なし、inline SVG
├── report.md
├── summary.csv                   # UTF-8 BOM
├── report_manifest.json
└── reports/<report_id>/           # 再生成時の追加 view（root 初版を保持）
```

過去の run directory は新しい run id で作成し、実験結果を上書きしません。最初の report は run root に保存し、同じ run の再生成は `reports/<report_id>/` へ保存して root 初版を残します。`--output-dir` を指定した report は旧runの status/manifest/生データを変更せず、指定先だけへ出力します。`offline --run-dir PARENT --config NEW` は旧runを読み取り専用で検証し、モデルhash、入力意味・順序・正規化・結合条件、前処理hashが一致するときだけ、通常観測をコピーした子runへ①-Aを保存します。patternのvariant追加は許可されますが、入力の意味や順序を変えた設定は拒否します。Bの新条件は子runで closed-loop を走り直してください。

動画を作る場合は設定の `video.enabled = true` と `video.patterns = ["P00", "P02_heading_neutral"]` を指定します。`video.patterns` は config に宣言した pattern ID のうち動画を保存するものだけを選ぶ配列で、未指定なら実行対象の全 pattern です。動画を全て無効にする場合は `video.enabled = false` とします。`closed-loop --patterns P00,P02_heading_neutral` は走行する介入 pattern の選択であり、動画選択とは別です。個別LiDAR全件などを動画の既定対象にしません。

旧初期参照を再現する場合は `input_attribution/configs/input_attribution_official_259.toml` または `input_attribution/configs/input_attribution_official_259_legacy_freeze.toml` を明示的に選びます。前者は旧公式構成、後者は初期 reference 条件を凍結した構成です。どちらも新しい全域variant設定 `input_attribution_official_259_variants.toml` と同じ pattern 条件として扱わず、レポートでも条件付き試験として区別します。

## レポートの読み方

①-A と①-B は別見出しの比較です。①-A の行動変更割合、元 Action の確率差(pp)、JSは判断の変化を示します。①-B の横ずれRMS、最大値、逸脱、到達、進行度、速度、終了理由は走行結果です。P00との差分は横ずれRMSと進行度を同じ条件で表示します。停車して横ずれだけが小さい走行を改善とは結論しません。

表の「適用件数」「実変更件数」「no-op件数」は別の母数です。元観測が置換値と同じ no-op は未変更として明示し、重要度0とは解釈しません。有限値がないもの、未実行、依存不足、テレメトリ欠測は N/A と理由を表示します。1シナリオの到達を一般化した成功率や統計的有意性として扱いません。

`report_offline_bars.svg` は①-Aの棒グラフ、`report_closed_loop_lateral.svg` は①-Bの横ずれ時系列、`report_closed_loop_trajectory.svg` は取得できた世界座標軌跡、`report_ig.svg` は③の補足です。画像・動画の失敗は数値結果を成功扱いにする理由にせず、リンクと失敗理由だけを残します。映像の frame と行動決定前入力の時刻対応は保存された media metadata を確認します。

`report_closed_loop_speed.svg` は post-step 時刻の速度、`report_closed_loop_steering.svg` は pre-action 時刻の操舵を pattern/episode 別の系列として描きます。有限な telemetry がない場合は空系列を0で補完せず、N/A として表示します。
