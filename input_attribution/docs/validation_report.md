# 実装・検証記録

この文書は実際に実行した検証と、移植先で残る確認を分けて記録します。合成方策・fake環境の結果をMetaDrive車両の性能として扱いません。

## 変更境界

開始時HEADは `0184eb26509eb33997229d0aa99c0b8939ec6a1e`、ブランチは `input_attribution`、開始時作業ツリーに変更はありませんでした。追加フォルダは `input_attribution/` です。既存の学習・評価・環境生成・設定・requirements・テストを編集せず、MetaDrive本体とインストール済みライブラリを編集していません。モデルを複製・再学習していません。

## 実行環境とモデル

- Python 3.12.3、NumPy 2.5.2、PyTorch 2.13.0、Stable-Baselines3 2.9.0、MetaDrive 0.4.3、Gymnasium 1.3.0。
- 使用した既存Python: `../metadrive_rl-main/.venv/bin/python3`。
- 既存モデル: `../metadrive_rl-main/models/official_baseline.zip`。
- モデルSHA256: `254b19aea772480133e19eb5db68b0fb5e1bdadfa221890a68494c8eb5d513e6`。
- 学習記録と現在の `configs/official.toml` のSHA256が一致: `385bfe6cf30c94e74ec90368d020e41857bf738c604688131a6220033eee909b`。
- 実観測259次元float32、単一Discrete(9)、学習時のBox観測前処理、外部VecNormalizeなしを確認。
- 公式接続では観測・行動に関係するMetaDriveの6ソースのbyte hashと、全259 indexの意味対応を照合します。導入版が異なる場合に次元数だけで受け入れません。

## 任意IGの依存

通常のPython環境にはCaptumを追加していません。通常環境で主工程を実行し、Captumがない場合のIG失敗が主結果を破損しないことをテストしています。

Captum 0.9.0は依存を更新せず一時wheelとして取得し、そのwheelだけを一時的な `PYTHONPATH` で参照して検証しました。wheel SHA256は `cda38e1d42c37591d71560bb2f5813ac4cae8d8543afac45279c7efd5945997e` です。線形の既知IG、非線形の既知積分、経路途中のargmax変化でも説明対象actionが固定されること、カテゴリの固定、completeness、重み不変・直接API契約の8テストを実行しました。

## 262次元の検証範囲

実機262次元のモデル・観測生成実装は手元にありません。実機262次元で検証済みとはしていません。fake262では追加3特徴をindex `7, 145, 260` に明示的に置き、別フォルダへコピーしたパッケージからcheck・収集・全入力①-A・指定①-B・レポートまで実行するテストを用意しています。

移植先では解析TOML、`adapters/port_template.py` を基にした接続、入力スキーマを編集します。追加3特徴のindex、符号、正規化、無効値、参照更新規則と、残りの全入力の順序を既存ソースで確認してください。外部正規化の有無も未設定から明示します。未解決テンプレートは実行前に停止します。

## 再実行

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests input_attribution/tests -q -p no:cacheprovider
python3 -m input_attribution check --config outputs/input_attribution_validation/official_validated_config.json --probe --probe-steps 3
python3 -m input_attribution run --config outputs/input_attribution_validation/official_validated_config.json
python3 -m input_attribution.pack --source-root . --output outputs/input_attribution_portable.zip
```

`official_validated_config.json` はこのPCの既存モデル・学習時設定を絶対パスで参照する実測用設定です。移植ZIPには実験出力とモデルを含めず、汎用の解析設定を `input_attribution/configs/` に同梱します。

## 新規ファイル一覧と責務

CLI/事前チェック: `cli.py`, `config.py`, `checks.py`。入力意味と置換: `schema.py`, `interventions.py`, `schemas/`, `configs/`。保存・比較: `artifacts.py`, `collection.py`, `offline.py`, `closed_loop.py`, `trajectory_metrics.py`, `metrics.py`, `policy.py`。任意IG: `integrated_gradients.py`, `requirements-ig.txt`。環境接続: `adapters/`。表示・移植: `reporting.py`, `pack.py`, `PORTABLE_FILES.txt`, `docs/`。検証: `tests/`。

```text
input_attribution/PORTABLE_FILES.txt
input_attribution/__init__.py
input_attribution/__main__.py
input_attribution/adapters/__init__.py
input_attribution/adapters/fake.py
input_attribution/adapters/fixtures.py
input_attribution/adapters/metadrive.py
input_attribution/adapters/port_template.py
input_attribution/artifacts.py
input_attribution/checks.py
input_attribution/cli.py
input_attribution/closed_loop.py
input_attribution/collection.py
input_attribution/config.py
input_attribution/configs/fake_259.toml
input_attribution/configs/fake_262_resolved.toml
input_attribution/configs/input_attribution_262_template.toml
input_attribution/configs/input_attribution_official_259.toml
input_attribution/docs/acceptance_matrix.md
input_attribution/docs/copilot_porting_prompt.md
input_attribution/docs/experiment_design.md
input_attribution/docs/implementation_baseline.json
input_attribution/docs/porting.md
input_attribution/docs/usage.md
input_attribution/docs/validation_report.md
input_attribution/integrated_gradients.py
input_attribution/interventions.py
input_attribution/metrics.py
input_attribution/offline.py
input_attribution/pack.py
input_attribution/policy.py
input_attribution/reporting.py
input_attribution/requirements-ig.txt
input_attribution/schema.py
input_attribution/schemas/README.md
input_attribution/schemas/__init__.py
input_attribution/schemas/custom_262_template.json
input_attribution/schemas/fake_262_resolved.json
input_attribution/schemas/official_259.json
input_attribution/tests/test_adapters.py
input_attribution/tests/test_cli.py
input_attribution/tests/test_core.py
input_attribution/tests/test_ig.py
input_attribution/tests/test_reporting.py
input_attribution/tests/test_runtime.py
input_attribution/trajectory_metrics.py
```

## 最終テスト結果

2026-09-08の最終コードで `python3 -m pytest tests input_attribution/tests -q -p no:cacheprovider` を実行し、**253 passed, 2 skipped (31.76秒)**。既存171テストを含みます。2 skipは通常環境にCaptumがないためです。一時wheelを参照する任意IGテストは別途 **8 passed**。ログは `outputs/input_attribution_validation/release_pytest.log` と `delivery_captum_pytest.log` です。

別ディレクトリへpackageのみをコピーする259/262 CLIテスト、環境を起動しないoffline、MetaDrive/SB3/PyTorch/Captum importを禁止したreport再生成、IG依存失敗時の主結果保持、再解析ID分離、モデル/schema/パターン/前処理/コード/観測hash不一致の拒否を含みます。

## 実MetaDrive 259次元の最終実験

納品用run: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f`。通常走行127 stepを保存し、①-Aの272パターン（全259入力の行、P00・設定group・整合groupの13行）、①-Bの明示した6パターン、主HTML/MD/CSVを生成しました。IGを実行する前に主工程とレポートが成功しています。

解析用設定は `outputs/input_attribution_validation/official_validated_config.json`。既存学習時設定のmap C、scenario seed 5、RL seed 0、交通量0、horizon 500を維持しています。参照置換は保存実観測 `episode-0:60` を使い、同一道路区間・開始時目標レーンordinalの一致を要求します。通常127観測のうち40観測で適用可能、残り87観測は理由付きskipです。各reference patternは39件を実変更、1件がno-opでした。

P00再走行は初期状態と全127 stepの参照traceが一致し、mismatch 0。収集時の確率と保存観測からの再計算が一致し、方策fingerprint・モデルSHA256は不変です。float32の一括推論と単観測推論の計算順序差を避けるため、確率評価は収集と同じ1観測単位で行います。

| パターン | step数 | 横ずれRMS m | 最大絶対値 m | 進行度 m | 平均速度 m/s | 到達 | 終了理由 |
|---|---:|---:|---:|---:|---:|---|---|
| P00 | 127 | 1.929347 | 3.017748 | 156.038969 | 12.284736 | はい | arrive_dest |
| P01_road_boundaries_reference | 91 | 2.216884 | 7.096743 | 97.795399 | 10.788536 | いいえ | out_of_road |
| P02_heading_reference | 127 | 1.601734 | 3.017748 | 156.407465 | 12.309249 | はい | arrive_dest |
| P04_current_lane_lateral_reference | 127 | 2.139771 | 3.437311 | 155.675600 | 12.260185 | はい | arrive_dest |
| P05_navigation_reference | 96 | 1.352124 | 3.017748 | 102.034829 | 10.531974 | いいえ | out_of_road |
| P06_lidar_all_no_detection | 127 | 1.929347 | 3.017748 | 156.038969 | 12.284736 | はい | arrive_dest |

横ずれは全走行で未加工の目標レーンテレメトリから取得し、有効率100%。P00は到達したものの目標レーン逸脱1回・6.0秒、最初の逸脱時刻6.8秒です。P01とP05は道路外で早期終了しました。P05の小さいRMSを性能改善と結論せず、到達・進行度・速度・時間を同じ表に表示します。対象は1シナリオであり、一般化した成功率や因果的重要度を主張しません。

①-AのP01は適用可能40観測で行動変更40%、元の選択行動確率の変化は **−15.5232879917 pp**。LiDAR全体と方向groupは全127観測がno-opであり、今回の条件では影響を評価できません。全272行の確率差と6走行の物理指標について、CSVとcanonical JSONの一致を追加検査しました。

HTMLのlocal link 14件は欠落0、外部CDN/resourceなし。CSVはUTF-8 BOM。通常走行と①-Bの計7 GIFは600×600で、全frameにpattern/episode/step/post時刻を表示し、frame mapとの件数が一致します。生成した図はグループ/個別入力比較、横ずれ・操舵・速度時系列、世界座標軌跡です。日本語フォントがないSVG環境では英語labelと日本語対応表を使います。実測の大きい保存JSONからのレポート生成は約1～2分かかります。検査の詳細は `outputs/input_attribution_validation/release_artifact_checks.json` に保存しました。

## 実モデルの任意IG

基準は保存済み `episode-0:70`、対象はstep 75・80。道路区間、開始時目標レーン、現在所属レーンとカテゴリ値が同じであることを確認しました。説明対象actionは各時刻の元argmaxで固定し、今回は両時刻ともaction 8です。Captum gausslegendre 64点、許容絶対残差1e-4、再計算0回で収束し、重み不変です。

| step | sum(IG) | F(x)-F(baseline) | 絶対completeness残差 |
|---:|---:|---:|---:|
| 75 | -0.325488496572 | -0.325488209724 | 2.86847352982e-07 |
| 80 | -0.954959951341 | -0.954959750175 | 2.01165676117e-07 |

生結果: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f/03_ig/20260908T105422Z-4e54f9c2/result.json`。IGは主順位へ合算せず補足に表示します。基準と同じ入力のIG=0を「未使用」と解釈しません。

## fake262と未検証事項

最終fake262 run: `outputs/input_attribution/fake_262_resolved/fake_policy_262/20260908T105202Z-17bf2917`。check・収集・全入力①-A・指定①-B・主レポートが成功しています。非末尾index 7/145/260を持つ合成例であり、実262モデルの車両性能は未検証です。実機262、Windowsでの実行、他シナリオ・他MetaDrive版・外部正規化を持つ移植先は、移植先の既存ソースとモデルでcheck/probeを行う必要があります。長い学習・再学習は行っていません。

## 変更境界の最終確認

`git diff HEAD --name-only` は空、`git status --short` は新規 `input_attribution/` のみ。既存ファイル・学習済みモデル・学習時設定を変更していません。実験結果は既存のignore対象 `outputs/` に別IDで保存しています。途中で停止した実験や表示修正前のrunも削除せず保持し、納品用runは上記IDで区別しています。

## 納品成果物の場所

- 主レポート: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f/report.html`（同じ場所にMD/CSV）。
- IG補足版: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f/reports/report-20260908T105438Z/report.html`。対象step・固定action・実観測baseline・F値・IG合計・出力差・残差・積分点数を評価ごとに表示します。異なる時刻のIGを一つの評価として合算しません。
- 初版のHTML/MD/CSVをハッシュで照合し、補足版生成後も不変であることを確認済みです。
- 移植ZIP: `outputs/input_attribution_portable.zip`。コード・文書・設定・スキーマ・テストの46ファイルのみを収録。モデル・実験結果・venv・cache・font・第三者repositoryは除外します。
- ZIP全entryのSHA256一覧: `outputs/input_attribution_validation/portable_manifest.json`。ZIPは2回生成してbyte SHA256の一致を確認します。

このPCの保存結果にIGを追加するコマンド（Captumが利用できる既存/一時環境で実行）:

```bash
python3 -m input_attribution ig \
  --run-dir outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f \
  --episode 0 --steps 75,80 --baseline episode-0:70
python3 -m input_attribution report \
  --run-dir outputs/input_attribution/official_baseline_259/official_baseline/20260908T105108Z-cbc5448f
```
