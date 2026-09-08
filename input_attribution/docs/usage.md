# 実行方法

解析用設定は学習用 TOML へ追加せず、`input_attribution/configs/` の解析設定として管理します。既存の仮想環境の Python を使い、SB3、PyTorch、MetaDrive を無条件に更新しません。259 の公式接続では実モデルと実環境が必要です。MetaDriveなしの fake adapter では接続契約と保存形式だけを検証します。

## 最短の実行

Linux/Ubuntu:

```bash
.venv/bin/python -m input_attribution check --config input_attribution/configs/input_attribution_official_259.toml
.venv/bin/python -m input_attribution run --config input_attribution/configs/input_attribution_official_259.toml
```

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m input_attribution check --config input_attribution/configs/input_attribution_official_259.toml
.\.venv\Scripts\python.exe -m input_attribution run --config input_attribution/configs/input_attribution_official_259.toml
```

`run` は check、通常走行収集、①-A、設定で指定した①-B、主レポート生成を順に行います。`run` では③ IG は常に未実行（skipped）として残ります。設定の `[ig].enabled` は Captum 依存の利用可否を check するためだけの項目で、実際の IG は下記の `ig` サブコマンドを対象 episode/step と baseline とともに明示して実行します。check はモデル、設定、adapter、schema、入力次元・dtype・範囲・全 index の被覆、離散 Action、前処理、代表確率再現、対象レーン、パターンの解決状態を人間向けと JSON で出します。`custom_262_schema_template` に未確定フィールドが残る設定は成功扱いにしません。

## 部分実行と保存結果からの再解析

```bash
# Linux
.venv/bin/python -m input_attribution collect --config input_attribution/configs/input_attribution_official_259.toml
.venv/bin/python -m input_attribution offline --run-dir outputs/input_attribution/<experiment>/<model>/<run_id>
.venv/bin/python -m input_attribution closed-loop --run-dir outputs/input_attribution/<experiment>/<model>/<run_id> --patterns P00,P02_heading_reference
.venv/bin/python -m input_attribution report --run-dir outputs/input_attribution/<experiment>/<model>/<run_id>

# 任意 IG（Captum が利用可能な環境だけ）
.venv/bin/python -m input_attribution ig --run-dir outputs/input_attribution/<experiment>/<model>/<run_id> --episode 0 --steps 12,20 --baseline episode-0:0
```

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

過去の run directory は新しい run id で作成し、実験結果を上書きしません。最初の report は run root に保存し、同じ run の再生成は `reports/<report_id>/` へ保存して root 初版を残します。設定、model hash、schema hash、pattern hash、観測 hash が manifest と一致しない部分実行は停止します。

## レポートの読み方

①-A と①-B は別見出しの比較です。①-A の行動変更割合、元 Action の確率差(pp)、JSは判断の変化を示します。①-B の横ずれRMS、最大値、逸脱、到達、進行度、速度、終了理由は走行結果です。P00との差分は横ずれRMSと進行度を同じ条件で表示します。停車して横ずれだけが小さい走行を改善とは結論しません。

表の「適用件数」「実変更件数」「no-op件数」は別の母数です。元観測が置換値と同じ no-op は未変更として明示し、重要度0とは解釈しません。有限値がないもの、未実行、依存不足、テレメトリ欠測は N/A と理由を表示します。1シナリオの到達を一般化した成功率や統計的有意性として扱いません。

`report_offline_bars.svg` は①-Aの棒グラフ、`report_closed_loop_lateral.svg` は①-Bの横ずれ時系列、`report_closed_loop_trajectory.svg` は取得できた世界座標軌跡、`report_ig.svg` は③の補足です。画像・動画の失敗は数値結果を成功扱いにする理由にせず、リンクと失敗理由だけを残します。映像の frame と行動決定前入力の時刻対応は保存された media metadata を確認します。

`report_closed_loop_speed.svg` は post-step 時刻の速度、`report_closed_loop_steering.svg` は pre-action 時刻の操舵を pattern/episode 別の系列として描きます。有限な telemetry がない場合は空系列を0で補完せず、N/A として表示します。
