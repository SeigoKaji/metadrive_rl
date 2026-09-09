# 検証レポート（異常系・欠測・有効フラグ修正）

## 今回の結果（2026-09-09）

基準コミットと現在のHEADは `cf8801e09f5270b8e72e1e870f86fbbfe37bf328`、ブランチは `input_attribution` です。開始時の未コミット差分はありませんでした。今回の修正は未コミットの差分として保持しています。下記の過去履歴にある309 passed、旧モデルSHA、旧run、pack検査は今回の検証結果へ流用していません。

最終コードを固定した全体pytestは **362 passed / 2 skipped / 48.32秒、exit 0** でした。実行前後の37 PythonファイルのSHA256は一致しました。新規回帰テストは4ファイル、計53件（R1:15、R2:17、R3:7、R4:14）です。skipはCaptum未導入による `test_ig.py:169` と `test_ig.py:206` の2件で、成功した①-A/Bを失敗へ変更しません。

検証成果物は `outputs/input_attribution_failure_validation_20260909/` に分離しています。`validation_final_evidence.json` に最終コマンド、ソースSHA、件数、report再生成、未検証範囲と保護監査を保存しました。`validation_evidence.json` は最後のreport境界修正前の360件検証スナップショットとして保持しています。既存run、既存モデル配置、train/evaluate/env_factoryなど保護対象9,489ファイルのハッシュは作業前と一致しています。MetaDrive/SB3本体・学習済みモデル・既存実験設定を変更せず、配布パッケージも再生成していません。

| ID | 再現・確認結果 | 修正箇所 | 追加テスト／検証 | 残る制限 |
| --- | --- | --- | --- | --- |
| R1 | 実CLIでpolicy例外を起こし、P00・後続正常patternと失敗途中結果を保存、終了コード1・一部失敗表示を確認。現在failed stageのみを表示し、旧成功値を混ぜない | `cli.py`、`reporting.py`。既存artifact state列挙値を維持しcountsを追加 | `test_revision_failure_reporting.py`:15件。最終追加2件も成功。CLI/reuse20件は追加境界修正前のfocused結果 | 初期policy例外ケースの先行redはfixture誤りで製品再現に含めない。追加mixed episodeケースは実producerでred/green確認 |
| R2 | 先行13件で10 failed / 3 passed。metadata absent/false/trueの不整合組合せを拒否し、確認済み正しい組合せは受入 | `interventions.py`。全関連入力の無効値・出典とclip/dtype後の値を照合 | `test_revision_invalid_joint.py`:17件。neutral/fixed_level、reflectionの離散フラグ拒否、reference、legacy fixedを実APIで確認 | 実262の無効時表現・入力順は未確認。未知値を推測して補完しない |
| R3 | 先行3件が失敗。2回目step return後のtelemetry/decode例外でも取得済み値を保存。主要集計へfailed/budget値が混入する追加2件もredからgreenへ修正 | `closed_loop.py`。call/return/phaseを分離、partial recordと配列を整合、主要評価と診断集計を分離 | `test_revision_failure_runtime.py`:7件。関連runtime40件も成功 | env.step自身が例外の場合、内部物理更新の有無はunknownのまま保存 |
| R4 | 欠測状態をイベントなし・無効状態へ補完せず、全体unknown/partialと既知値・分母を分離。未return試行は物理step数から除外 | `trajectory_metrics.py`、`reporting.py`。既知RMS保持、欠測差分はNone、主表に一部未計測を表示 | `test_revision_missing_metrics.py`:14件。関連27件と実CLI report回帰が成功 | 先行テストの順序は未達。後述の未修正版module再実行を先行redとは扱わない |
| R5 | 新主経路と標準B対象、追加full_episode選択例を確認 | README、`docs/usage.md`、本書 | 文書の必須項目・差分形式を確認 | 全259入力の全区間B評価とは主張しない |

テスト追加順序には制限があります。R2/R3は製品修正前の失敗を確認しました。R1の初期policy例外ケースで記録された先行1 failedは合成fixtureの識別条件の誤りで、製品の不具合を再現した証拠ではありません。その後のレポート境界追加では、実 `ClosedLoopResult.as_dict()` とepisode別trajectoryを保存したmixed状態ケースの修正前失敗を確認しました。最終の境界2件は2 passed / 13 deselectedです。R4は修正後に `git show HEAD:input_attribution/trajectory_metrics.py` の未修正版ファイル全文を動的importし、最終14テストを適用して13 failed / 1 passedを確認しました。関数・式の抜粋ではありませんが、「修正前に回帰テストを追加して実行する」という順序を満たしていません。これらを先行redの成功として報告しません。ブランチ・HEADを戻す操作は行っていません。

既存の①-A（保存した同じ観測のコピーを用い環境を進めない）、①-B（各介入走行の最新観測のコピーへ同じルールを適用し、その行動を実行する）、型付き置換、自然な衝突・逸脱終了のpaired適格条件は維持しました。`RunArtifacts.update_status` は既存の `pending/running/success/failed/skipped` と任意detailsを使えるため変更していません。A/B主表はそれぞれ8列です。

## 実行コマンドと最終テスト

作業場所は `/home/kajiseigo/workspace/metadrive_rl/metadrive_rl-attribution`、既存Pythonは隣接作業ツリーの仮想環境を使用しました。

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/input-attribution-failure-mpl \
XDG_CACHE_HOME=/tmp/input-attribution-failure-cache \
TORCH_HOME=/tmp/input-attribution-failure-torch \
/home/kajiseigo/workspace/metadrive_rl/metadrive_rl-main/.venv/bin/python3 \
-m pytest tests input_attribution/tests -q -rs -p no:cacheprovider
```

最終ログ: `outputs/input_attribution_failure_validation_20260909/full_pytest_accepted.log`（362 passed、2 skipped、48.32秒、exit 0）。最後に追加した2件は、旧形式error/ERROR stageのpointer欠落時にstale successを読まないことと、実producerのmixed episode summaryが正常episodeの状態・RMSを失敗側へ伝播しないことを検証しています。独立read-onlyレビューでもこの2件の解消を確認しました。

先行の `full_pytest_final.log` は360 passed、2 skipped、48.60秒で、その時点のソースは固定されていましたが、最後のレポート境界2件を含みません。それ以前の `full_pytest.log` は360 passed、2 skipped、49.56秒で、実行中のreporting変更を含むため最終結果には採用していません。上のfocused件数は重複する範囲を含むので、全体件数へ足しません。

## 実行ロジック修正後の小規模な実259確認

モデルは `../metadrive_rl-main/models/official_baseline.zip`、SHA256は `254b19aea772480133e19eb5db68b0fb5e1bdadfa221890a68494c8eb5d513e6` です。過去履歴のモデルSHAとは異なるため、結果を同一モデルの追試として混ぜません。

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/input-attribution-failure-mpl \
XDG_CACHE_HOME=/tmp/input-attribution-failure-cache \
TORCH_HOME=/tmp/input-attribution-failure-torch \
/home/kajiseigo/workspace/metadrive_rl/metadrive_rl-main/.venv/bin/python3 \
-m input_attribution run \
--config outputs/input_attribution_failure_validation_20260909/real_normal_config.json
```

最終runは `outputs/input_attribution_failure_validation_20260909/real_runs/failure_revision_259_natural/official_baseline/20260909T064131Z-12b98dc7`、ログは `real_final.log`、終了コードは0でした。対象はP00、P02_heading_neutral、P03_speed_fixed_level、P03_history_group_neutralの4本、各1 episode、上限500 step、CPU、動画/IGなしです。CLIは `completed=4, failed=0, aborted=0` と表示しています。全15 patternの再実行や再学習はしていません。最終成果物の監査は `real_final_evidence.json` に保存済みです。4本ともarrive_destで自然終了し、P00・heading・historyは127 records、speedは128 records、各配列は `(records, 259)`、正常return数と保存件数は一致しました。3つの介入はP00 pair verified、全行の加工観測argmaxとforwarded actionが一致し、非対象入力の最大差は0でした。policy/model SHA、12件のadapter source hashは不変です。全patternの `metric_summary_scope` は `natural_completion` で、診断集計も保存されています。実走行後の追加変更は `reporting.py` とその回帰テストのみで、実行ロジックは変更していません。最終reportコードで別出力先 `accepted_report` へ再生成しexit 0、Markdown/HTMLのA/B主表は各8列、再生成前後の元run全54ファイルはハッシュ一致でした。先行の `newfinal_report` と `real_final_evidence.json` は最後のreport境界修正前の監査として保持し、最新のreport監査は `validation_final_evidence.json` に分離しています。

最終report再生成コマンド（実走行は追加せず、保存済みrunを読みます）:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/input-attribution-failure-mpl \
XDG_CACHE_HOME=/tmp/input-attribution-failure-cache \
TORCH_HOME=/tmp/input-attribution-failure-torch \
/home/kajiseigo/workspace/metadrive_rl/metadrive_rl-main/.venv/bin/python3 \
-m input_attribution report \
--run-dir outputs/input_attribution_failure_validation_20260909/real_runs/failure_revision_259_natural/official_baseline/20260909T064131Z-12b98dc7 \
--output-dir outputs/input_attribution_failure_validation_20260909/accepted_report
```

途中検証は次のように別runへ保存しています。

- `real_smoke_config.json` の24-step cap: run `20260909T061843Z-25752002`、exit 1。collect 127 recordsに対してB側24 recordsとなり、P00参照trace長が不一致でした。P00はpairing failure、他3本はcompletedで、診断が保存されました。`real_smoke_evidence.json` / `real_smoke.log` を参照してください。解析設定のscenario.horizonだけで実環境のhorizonが短くなるとは仮定せず、Bだけを途中で打ち切ったこの条件を自然終了の検証として扱いません。
- 中間の自然終了run `20260909T062000Z-20d96ab3`: exit 0、4 completed / 0 failed / 0 aborted、3 pair verified。P00、heading neutral、history group neutralは127 step、speed fixed levelは128 stepでした。非対象入力不変、加工観測のargmaxと行動の一致、方策・モデルSHA不変を監査しました。`real_normal_evidence.json` / `real_normal.log` はsummaryの最終修正前の記録として保持します。
- `real_check.log` は修正前のCLI check（exit 0、errors/warningsなし）であり、最終コードの実走行証拠とは分けています。

## 判定と未検証範囲

共通のモデル・adapter・環境生成、reset、seed/readbackなどの初期化失敗はrun全体の失敗です。初期化後のpattern内のpolicy、env.step、telemetry、decodeなどのruntime例外は `failed`、予期しないfull_episode介入不適用は `aborted` として理由・途中記録を保存し、他patternを継続します。必須実験にこれらがあればstageはfailed、CLIは非0となり、処理が最後まで進んだことを全実験成功へ変換しません。

自然な到達、衝突、道路逸脱、環境仕様上の終了、no-op、明示条件外skipは、それ自体でruntime failureにしません。主要metricはnatural completionだけで集約し、failed/aborted/budget-censoredの値は `diagnostic_metric_summary` と `execution_groups` に残します。env.stepの呼出しと正常returnを区別し、return後のtelemetry欠測は物理stepの期待母集団に含めたうえで計測率を下げます。

レポートは現在の明示stage pointerとその診断を読みます。failed/error stageに安全なpointerがなければ旧成功stageへフォールバックしません。pattern全体のmixed status配列を各episodeの状態へ流用せず、episode別trajectoryのscalar metadataを優先します。対応を特定できないsummary-onlyの混在状態はunknownとします。`report` の再生成は `--output-dir` の有無にかかわらず既存runのstatus・manifest・条件・生データを変更しません。欠測を0や完全評価にせず、execution/assessmentとmeasurement completenessを分離し、既知部分の値と分母を表示します。

実262モデル・観測生成器、Windows、Captumを用いたIG、動画frameは未確認です。合成262テストは移植先の262次元仕様の証明ではありません。単一モデル・1シナリオ・重点4patternの結果を全入力の重要度、一般的性能、統計的有意性へ拡張しません。

## 過去履歴: 実行対象と固定モデル

- 対象ブランチ: `input_attribution`
- 実走行用 Python: `/home/seigo/workspace/metadrive_rl/metadrive-rl/.venv/bin/python3`
- 報告対象モデル: `/home/seigo/workspace/metadrive_rl/metadrive-rl-input-attribution/models/official_baseline.zip`
- 報告対象モデル SHA256: `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`
- 新 259 設定: `input_attribution/configs/input_attribution_official_259_variants.toml`
- 旧条件を凍結した設定: `input_attribution/configs/input_attribution_official_259_legacy_freeze.toml`
- 新 full run: `outputs/input_attribution/official_baseline_259_typed_variants/official_baseline/20260908T220014Z-e7425ea0`
- 実走行 evidence: `outputs/input_attribution_revision_validation/new_real_evidence.json`
- 実走行 manifest model SHA256: `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`
- 実走行 manifest schema semantics SHA256: `b32e08e797f7c7d5a7d4a3fc989e955e820936130c69e6b6e732f270a30a172b`
- 実走行 manifest preprocess SHA256: `5486d5d3c281145521c358bf42b439fc3fe29997b6bb2a95a333242469b27dff`
- A reuse child: `outputs/input_attribution/official_baseline_259_typed_variants/official_baseline/20260908T224736Z-75bdb5bb`
- A reuse analysis: `01_offline/20260908T224736Z-374ab9b9`（success / 286 patterns）
- A reuse child fresh B stage: `02_closed_loop/20260908T225123Z-1de748bb`（success / 15 patterns）
- A reuse log: `outputs/input_attribution_revision_validation/saved_reference_reuse.log`
- 新 report output: `outputs/input_attribution_revision_validation/reused_run_report`（exit 0、成果物監査 PASS）
- 旧 report output: `outputs/input_attribution_revision_validation/legacy_report_final`（exit 0、成果物・保護監査 PASS）
- report audit: `outputs/input_attribution_revision_validation/artifact_checks.json` / `artifact_checks.log`

モデル、保存済み通常観測、前処理統計、adapter source の hash は run の manifest に保存し、再利用時に照合します。report の `--output-dir` は旧 run の status、manifest、生データを変更せず、指定先にだけ report を書き出します。`offline --run-dir PARENT --config NEW` は参照元 run の sealed reference をコピーした新しい A 専用子 run を作り、参照元の条件と入力意味・順序・前処理が一致しない場合は子 run を作らず拒否します。

## 過去履歴: 実走行で確認した項目

過去の実行記録にある新 full run は以下の状態で完了しています。

```text
run: 20260908T220014Z-e7425ea0
collect: success / 1 episode / 127 records
offline: success / 286 patterns / saved probabilities reproduced / policy unchanged
closed-loop: success / 15 patterns
IG: skipped（明示的な ig コマンド未実行）
3-step probe: old/new とも status OK（`outputs/input_attribution_revision_validation/initial_real_probe.log`、`new_real_probe.log`）
full run command: exit 0（`outputs/input_attribution_revision_validation/new_full_run.log`）
A reuse child: exit 0 / 286 patterns / saved probabilities reproduced / policy unchanged
fresh B on A child: exit 0 / 15 patterns / `summary.policy_unchanged=true`（stage `02_closed_loop/20260908T225123Z-1de748bb`）
```

通常観測は旧正常観測の `observations.npy` と bytes 一致しています。A は25個の明示 pattern（P00 + 非control 24）に259個の individual patternと2個の group展開を加え、合計286 patternを保存しました。A の非control明示 pattern数は24、Bの非control P00 pair verified 数は14です。A/B の `target = applied + skipped` と `applied = changed + noop` は evidence の全行で成立しています。

`closed_loop.max_steps=500` は planned 上限であり、自然終了した episode の実 step 数・実測 duration とは別です。今回の B は pattern ごとに実測 step 数が異なり、P00 は127、heading neutral は128、heading reflection は114、heading low は85でした。実測可能な時間がない場合の duration は明示的な `None`、既知 interval だけを合計できる場合は known partial sum と partial status とし、欠測を0や planned 500 step の時間で補完しません。pair の状態が unknown の episode は P00 差の集約から除外します。

A reuse child は参照元 run の sealed reference を再利用して environment を作成せずに A を実行しました。child の `observations.npy` SHA256 は参照元と同じ `b6ab07bfcb712a1d7d09f70ff3cf3b11e15379b07a71f32bbbd90863bd4fdc57` です。child manifest には参照元 model SHA256 `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`、参照元 input semantics SHA256 `b32e08e797f7c7d5a7d4a3fc989e955e820936130c69e6b6e732f270a30a172b`、source `input_schema.json`、preprocess SHA256 `5486d5d3c281145521c358bf42b439fc3fe29997b6bb2a95a333242469b27dff`、parent collection provenance SHA256 `57db2ebd5fa1b53556452d0393ef1e58e5911cb057219d7716a5846cf60f38a0`、parent data ID `cda9f497f3c78f1374e279b4fd9f9bedea3e1e73a941aed2e5453448f54fd10c` を保存しています。collect は saved reference reused として skipped でした。child の fresh B は `02_closed_loop/20260908T225123Z-1de748bb` で exit 0、全15 pattern が success、`summary.policy_unchanged=true` です。

child の保存配列監査は `outputs/input_attribution_revision_validation/final_child.json` に保存されています。参照元との対応は同一で、A は286 patternについて float32配列、非対象 index、exact mask/count、各不変条件がすべて一致し、individual LiDAR 240 pattern は全件 no-op でした。B は15 episodeで float32配列の非対象 indexが不変、`action == forwarded == argmax(modified probabilities)` の不一致0・欠測0でした。child fingerprint の介入前後は `64bd78c1c476b2c0ab44b139cdf2867abfcdd4d290f1bad1cbb5a57645145dfd` で一致し、verified P00 pair は14、missingは0、telemetryは15件すべて complete（lane/clock/progress/speed の欠測0）でした。この配列監査は fresh B child の証拠であり、参照元 full run の fresh collect 証拠とは分けて記録します。

新 report の生成は exit 0 で完了し、`report.md`/`report.html` の主表は最大8列、主表から参照するローカル href/src は全件解決しました。主表には P00 の物理基準を A 表より先に置き、個別 LiDAR 240入力の全 no-op 集約、動画無効15件の理由、重複件数の説明、IG未実行1行を保存しています。A は286 pattern、B は15 run、検証済み B pair は14、P00 は127 applied/127 noopです。A 主表（P00除外）の分類は changed 29、noop 250、partial 17、all-skip 6（重複分類を含む）、B は changed 11、noop 3、partial 4、all-skip 0です。A CSV は source records 72,644 行ですが `(pattern, episode, step)` の unique 行は36,322で、2つの source representation を `source`/`source_priority` とともに保存した結果であり、観測数の水増しではありません。B CSV は1,852行、通常観測は127 unique step・1 episodeです。変更なしの主表列は N/A とし、raw JS は詳細に0を保存しています。

旧 report の生成も exit 0 で完了し、`legacy_report_final` は同じ成果物監査に合格しました。旧条件では A は272 pattern、B は5 run、検証済み B pair は4、P00 は127 applied/127 noopでした。旧主表（P00除外）の分類は A が changed 11、noop 254、partial 20、all-skip 6（重複分類を含む）、B が changed 2、noop 2、partial 3です。旧 `P02` A の JS は `9.638e-16`、平均絶対確率差は `1.144e-6 pp` で、微小値として意味のある依存とは判定していません。旧 report も P00 を A 表より前に置き、raw JS 0 と主表 N/A、video の実際の理由を保存しています。

新旧 report の保護監査では、変更なし42件、旧 run 588件、旧 inventory 全件が一致しました。HEAD は `6584bd9f5a5786c1374cbf7d3536a5dbd5ea50f11`、branch は `input_attribution` のままで、tracked の変更・追加31件は `input_attribution/` 配下だけです。両 report の `audit_artifacts.py` は exit 0 でした。

主な A 実測値は次のとおりです。確率差は実変更時の平均絶対 pp で、母数を併記しています。

| pattern | A 実測 |
| --- | --- |
| `P02_heading_neutral` | 127対象、120 exact、7 no-op、meaningful 73（tolerance `1e-7`）、行動変更 5/120、平均絶対選択確率差 2.48307654 pp |
| `P02_heading_reflection` | 120 exact、行動変更 12/120、平均絶対選択確率差 5.0375491 pp |
| `P04_lateral_neutral` | 71 exact、56 no-op |
| `P06_lidar_no_detection` | 127適用、127 no-op、assessment は `no_exact_input_change` |
| `P06_lidar_virtual_detection` | 127 exact、0 no-op |

主な B 実測値は次のとおりです。全15 pattern が完了し、P00以外の14 patternは `matched=true` でした。P00 の reference trace は127 records、mismatch 0です。unexpected な full-episode abort はありませんでした。

| pattern | B 実測 |
| --- | --- |
| `P00` | arrival=true、target-lane RMS `2.762390679 m`、逸脱1回・6 s、progress `155.682517 m`、duration `12.7 s`、crash=false |
| `P02_heading_neutral` | 128 steps、121 exact、7 no-op、arrival=true、road-out=false |
| `P02_heading_reflection` | 114 steps、107 exact、road-out=true、arrival=false |
| `P02_heading_low` | 85 steps、85 exact、road-out=true、arrival=false |
| `P04_lateral_neutral` | target-lane RMS `2.346670696 m`、逸脱1回・6 s、arrival=true。ただし逸脱が残るため改善またはレーン維持成功とは判定しない |
| `P06_lidar_no_detection` | 127 steps、127 no-op、P00 と同じ control-baseline 指標 |
| `P06_lidar_virtual_detection` | 127 steps、127 exact、arrival=true、road-out=false |

全 B pattern で lane、speed、progress、clock の coverage は100%でした。`wrong_lane_arrival` と `start_lane_departure` は `None` で、取得できていないため成功・失敗の0へ置換していません。動画は `video.enabled=false` のため全 patternで disabled です。

report と保護の確認済み工程:

- 旧 run に対する `report --run-dir OLD --output-dir NEW_REPORT_DIR` は exit 0、出力先は `legacy_report_final` です。
- A 子 run の実コマンド `python3 -m input_attribution closed-loop --run-dir CHILD`（config の全15 patternを実行）は stage `02_closed_loop/20260908T225123Z-1de748bb`、exit 0 です。
- 新旧 report の CSV / HTML / SVG / media link、P00 基準、N/A 表示、重複説明、参照元全ファイル hash を監査済みです。

新 report は `outputs/input_attribution_revision_validation/reused_run_report`、旧 report は `outputs/input_attribution_revision_validation/legacy_report_final` に生成され、主表・詳細表・ローカルリンク・media reason・P00 基準・N/A 表示の監査に合格しました。A reuse child 上の fresh B 全15 pattern も完了しています。

## 過去履歴: 合成・保存 package で確認した項目

ここには全体 pytest、pack 検査、保存結果の再解析の実行結果を記録します。これらは MetaDrive 車両の挙動を検証しません。

```text
統合 full pytest コマンド:
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/input-attribution-revision-mpl XDG_CACHE_HOME=/tmp/input-attribution-revision-cache TORCH_HOME=/tmp/input-attribution-revision-torch python3 -m pytest tests input_attribution/tests -q -p no:cacheprovider
結果: 309 passed, 2 skipped in 70.92s
ログ: outputs/input_attribution_revision_validation/final_pytest.log
補足: Captum 未導入による skip 2件。主工程の失敗ではない。

portable package の所定コマンドと出力先:
python3 -m input_attribution.pack --source-root . --output outputs/input_attribution_revision_validation/input_attribution_portable.zip
package verification: outputs/input_attribution_revision_validation/portable_package_checks.json に記録
```

新 report 成果物監査:

```text
report command: exit 0
output: outputs/input_attribution_revision_validation/reused_run_report
main report: 最大8列、ローカル href/src 全件解決、IG未実行1行、video_disabled 15件の理由表示
保存件数: A 72,644 source rows / 36,322 unique (pattern, episode, step)、B 1,852 rows
分類: A(P00除外) changed 29 / noop 250 / partial 17 / all-skip 6、B changed 11 / noop 3 / partial 4 / all-skip 0
P00: 127 applied / 127 noop、verified B pair 14、通常観測 127 unique step / 1 episode
監査: final_child.json PASS、raw JS 0 保存、変更なしの主表値は N/A
```

確認対象:

- variant の追加・推奨 replacement の変更は意味 hash の例外として許可し、入力順、encoding、normalization、coupling、validity、invalid sentinel の変更は拒否する。
- legacy manifest に意味 hash がなくても、参照元の `input_schema.json` snapshot を正規化して hash を復元し、根拠 path と hash を子 manifest に保存する。
- 参照元の `resolved_config.json`、snapshot files、reference seal、前処理統計、adapter source、model hash の改変を子 run 作成前に拒否する。
- offline reuse は environment を作成せず、参照元の observation/action mapping を継承し、A のみを成功、B を未実行として明示する。
- `01_offline/patterns.json` / `patterns.csv` と子 run の pattern snapshot は、型付き variant の `variant_id`、evidence、classification、resolved value、reflection expression を保存し、report の主表・詳細表へ引き継ぐ。
- `video.patterns` の config 検証と closed-loop の pattern 別動画選択は合成テストで確認済み。実 MetaDrive の frame、frame 数、動画 metadata は実走行欄へ記録する。
- `report --output-dir` は旧 run のファイルを変更せず、pack は `input_attribution/` 追加 package だけを収録して外部 model、MetaDrive、262 観測生成器を上書きしない。

## 過去履歴（現 revision の証拠ではない）

旧文書にあった 2026-09-08 の 259/262 実行記録、旧 run ID、旧テスト件数は revision 前の履歴です。参照元として使用する旧 run の manifest は確認済みで、`20260908T191831Z-400a7dd2` は今回の報告対象モデルと同じ SHA256（`0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`）を記録しています。ただし revision 前の run なので、現 revision の実走行証拠と混ぜず、参照元 integrity と provenance を確認したうえで report 再生成または A reuse の入力として扱います。旧文書に残るさらに古いモデル SHA256 や存在しない環境パスは別の過去記録であり、今回の旧 run と同一視しません。

旧 run 例: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T191831Z-400a7dd2`。この run は変更せず、report 再生成または reuse の参照元として使う場合も全 hash と provenance を先に検証します。

## 過去履歴: 実環境で未確認の項目

- 262 check は `unresolved_262_check.log` で expected exit 2（`SchemaError: input index must be an integer`）でした。未解決の262 schemaを成功扱いにせず、実 262 モデル・観測生成器も本 repository には提供されていません。
- Windows 実行、実 262 adapter/モデル、Captum を導入した IG、動画 frame と media metadata は今回の証拠では未確認です。Captum 未導入の skip は主工程の成功へ置き換えず、status と理由を保存します。
- 動画対象は `[video].patterns = ["P00", "<重点pattern>"]` で選択します。`closed-loop --patterns P00,<重点pattern>` は走行対象の選択で、動画選択とは別です。未指定は走行対象全件です。全件無効は `video.enabled = false` で指定します。今回の full run は video disabled なので frame 成果物はありません。
