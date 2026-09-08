# 検証レポート（revision 実走行証拠）

この文書は、revision 後の合成検証と実 MetaDrive 259 走行の証拠を分けて記録します。A reuse 子 run、子 run の fresh B 再走行、新旧 report の生成と成果物・保護監査は完了しました。移植用 zip は所定の出力先へ生成するコマンドを記録し、実 262、Windows、Captum、動画 frame の実環境確認は未確認として残します。1 episode の値を全場面の一般的な性能や入力重要度へ拡張しません。

実行の成立境界も分けます。モデル・adapter・環境の作成、run 全体の reset、要求 seed の適用と readback など、全 pattern に共通する初期化が失敗した場合は run 全体を無効とします。共通初期化が成立した後の pattern 内で intervention、policy inference、Action decode、`env.step()`、または pattern 固有 telemetry に失敗した場合は、その pattern に `aborted` と理由・途中件数を保存し、他 pattern の実行と run の診断を継続します。pattern の abort を run 全体の success や失敗隠しに変換しません。

## 実行対象と固定モデル

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

モデル、保存済み通常観測、前処理統計、adapter source の hash は run の manifest に保存し、再利用時に照合します。report の `--output-dir` は旧 run の status、manifest、生データを変更せず、指定先にだけ report を書き出します。`offline --run-dir PARENT --config NEW` は親の sealed reference をコピーした新しい A 専用子 run を作り、親の条件と入力意味・順序・前処理が一致しない場合は子 run を作らず拒否します。

## 実走行で確認した項目

親担当が実行した新 full run は以下の状態で完了しています。

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

A reuse child は親の sealed reference を再利用して environment を作成せずに A を実行しました。child の `observations.npy` SHA256 は親と同じ `b6ab07bfcb712a1d7d09f70ff3cf3b11e15379b07a71f32bbbd90863bd4fdc57` です。child manifest には親 model SHA256 `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`、親 input semantics SHA256 `b32e08e797f7c7d5a7d4a3fc989e955e820936130c69e6b6e732f270a30a172b`、source `input_schema.json`、preprocess SHA256 `5486d5d3c281145521c358bf42b439fc3fe29997b6bb2a95a333242469b27dff`、parent collection provenance SHA256 `57db2ebd5fa1b53556452d0393ef1e58e5911cb057219d7716a5846cf60f38a0`、parent data ID `cda9f497f3c78f1374e279b4fd9f9bedea3e1e73a941aed2e5453448f54fd10c` を保存しています。collect は saved reference reused として skipped でした。child の fresh B は `02_closed_loop/20260908T225123Z-1de748bb` で exit 0、全15 pattern が success、`summary.policy_unchanged=true` です。

child の保存配列監査は `outputs/input_attribution_revision_validation/final_child.json` に保存されています。親参照は同一で、A は286 patternについて float32配列、非対象 index、exact mask/count、各不変条件がすべて一致し、individual LiDAR 240 pattern は全件 no-op でした。B は15 episodeで float32配列の非対象 indexが不変、`action == forwarded == argmax(modified probabilities)` の不一致0・欠測0でした。child fingerprint の介入前後は `64bd78c1c476b2c0ab44b139cdf2867abfcdd4d290f1bad1cbb5a57645145dfd` で一致し、verified P00 pair は14、missingは0、telemetryは15件すべて complete（lane/clock/progress/speed の欠測0）でした。この配列監査は fresh B child の証拠であり、親 full run の fresh collect 証拠とは分けて記録します。

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
- 新旧 report の CSV / HTML / SVG / media link、P00 基準、N/A 表示、重複説明、親全ファイル hash を監査済みです。

新 report は `outputs/input_attribution_revision_validation/reused_run_report`、旧 report は `outputs/input_attribution_revision_validation/legacy_report_final` に生成され、主表・詳細表・ローカルリンク・media reason・P00 基準・N/A 表示の監査に合格しました。A reuse child 上の fresh B 全15 pattern も完了しています。

## 合成・保存 package で確認した項目

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
- legacy manifest に意味 hash がなくても、親の `input_schema.json` snapshot を正規化して hash を復元し、根拠 path と hash を子 manifest に保存する。
- 親の `resolved_config.json`、snapshot files、reference seal、前処理統計、adapter source、model hash の改変を子 run 作成前に拒否する。
- offline reuse は environment を作成せず、親の observation/action mapping を継承し、A のみを成功、B を未実行として明示する。
- `01_offline/patterns.json` / `patterns.csv` と子 run の pattern snapshot は、型付き variant の `variant_id`、evidence、classification、resolved value、reflection expression を保存し、report の主表・詳細表へ引き継ぐ。
- `video.patterns` の config 検証と closed-loop の pattern 別動画選択は合成テストで確認済み。実 MetaDrive の frame、frame 数、動画 metadata は実走行欄へ記録する。
- `report --output-dir` は旧 run のファイルを変更せず、pack は `input_attribution/` 追加 package だけを収録して外部 model、MetaDrive、262 観測生成器を上書きしない。

## 過去履歴（現 revision の証拠ではない）

旧文書にあった 2026-09-08 の 259/262 実行記録、旧 run ID、旧テスト件数は revision 前の履歴です。親として使用する旧 run の manifest は確認済みで、`20260908T191831Z-400a7dd2` は今回の報告対象モデルと同じ SHA256（`0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`）を記録しています。ただし revision 前の run なので、現 revision の実走行証拠と混ぜず、親 integrity と provenance を確認したうえで report 再生成または A reuse の入力として扱います。旧文書に残るさらに古いモデル SHA256 や存在しない環境パスは別の過去記録であり、今回の旧 run と同一視しません。

旧 run 例: `outputs/input_attribution/official_baseline_259/official_baseline/20260908T191831Z-400a7dd2`。この run は変更せず、report 再生成または reuse の親として使う場合も全 hash と provenance を先に検証します。

## 実環境で未確認の項目

- 262 check は `unresolved_262_check.log` で expected exit 2（`SchemaError: input index must be an integer`）でした。未解決の262 schemaを成功扱いにせず、実 262 モデル・観測生成器も本 repository には提供されていません。
- Windows 実行、実 262 adapter/モデル、Captum を導入した IG、動画 frame と media metadata は今回の証拠では未確認です。Captum 未導入の skip は主工程の成功へ置き換えず、status と理由を保存します。
- 動画対象は `[video].patterns = ["P00", "<重点pattern>"]` で選択します。`closed-loop --patterns P00,<重点pattern>` は走行対象の選択で、動画選択とは別です。未指定は走行対象全件です。全件無効は `video.enabled = false` で指定します。今回の full run は video disabled なので frame 成果物はありません。
