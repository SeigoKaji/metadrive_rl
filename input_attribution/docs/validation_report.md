# 実装・検証記録

検証日: 2026-09-10。対象は1モデル・map=C・scenario seed=5の1 episode条件です。結果は範囲外入力の固定値ストレス試験であり、一般的な因果的重要性や統計的有意性を示すものではありません。

## Gitと変更範囲

- `git fetch origin main`で取得して固定したBASE_MAIN_SHA: `7849aad80ac353fd616c1a1398c11dd3497eed05`。
- 新規branch: `input_attribution_visual_main`。作業worktree: `metadrive_rl-input-visual`。元の`metadrive_rl-attribution` worktreeはbranch `input-attribution`、元HEADのままclean。
- 走行実装SHA: `983e913a6454a66fc5ba0611404b5799274621e3`。最終の実N=10／合成N=2のmanifestはともに`dirty=false`。manifestには基点SHA、実装SHA、package内容SHA-256と対象ファイル、model hash、設定、schema、seed、前処理、依存version／import元を保存。
- 派生レポート再生成SHA: `7a3e1ce3b2f9bc6336d5b056cd356d6e26823924`。走行後に固定値ラベルを短縮し、失敗した再生成で旧画像が残る問題を修正したもの。保存`data/`を変更せず再生成した。人工262 demoもこのSHAで実行。
- 変更は追加`input_attribution/`配下のみ。既存のtrain/evaluate/env_factory/start_lane_env、既存config、MetaDrive／SB3／torch本体、モデル重みは変更していない。学習は実行していない。
- 再利用: `configs.experiment_config.select_experiment` → `env_factory.make_evaluation_env`、`evaluation_visualization.derive_timing`、MetaDriveのtopdown render API。

## 実行環境と入力境界

既存仮想環境`../metadrive_rl-main/.venv`を使用。Python 3.12.3、MetaDrive 0.4.3、SB3 2.9.0、torch 2.13.0、Gymnasium 1.3.0、numpy 2.5.2、Pillow 12.3.0。実際のimport絶対pathは各manifestの`adapter.runtime.libraries`に保存。

- モデル: `models/official_baseline.zip`（既存モデルへのローカルsymlink）。SHA-256: `254b19aea772480133e19eb5db68b0fb5e1bdadfa221890a68494c8eb5d513e6`。実行後も一致。
- MetaDrive source: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`、実行後もclean。259 schemaの根拠となる実source file hashも照合。
- D=259、K=9、1次元float32 Box、Discrete、PPO MlpPolicy。canonical `configs/official.toml`、map=C、scenario/policy seed=5、horizon=500、deterministic=True。
- raw観測からモデル入力まではidentity。SB3の非画像Box／FlattenExtractor経路で、`policy.mlp_extractor.policy_net`の一時forward pre-hookがMLP入口を捕捉。対象値と非対象値を実際の入口で検証し、hookは必ず解除する。
- 最終checkはbaseline/P01の各2 stepとフレーム取得に成功。通常predictとの一致、-1到達、実D/K、schema、描画、報酬接続状態を確認。-100は有限固定値として受入テストで確認し、実10groupの既定実行には追加していない。
- 全11走行で実seedと初期観測hashが一致し、取得可能な初期状態、model／前処理identity、action_dt、返却報酬の段階もpairingで照合。走行が分岐した後の乱数列の完全一致は主張しない。

## テスト

リポジトリルートで既存仮想環境を有効にして実行。共有CPU上では`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1`を使用した。

```bash
python -m pytest -q -p no:cacheprovider input_attribution/tests
# 80 passed in 8.23s
python -m pytest -q -p no:cacheprovider tests lookahead_learning
# 251 passed in 6.11s
python train.py --help
python evaluate.py --help
# ともにexit 0
```

既存251件の後で既存コードは変更していない。描画再生成の最終修正後は追加80件を再実行し、全件通過。独立レビューで検出したargmax比較、失敗baseline、返却済みstepの記録保持、provider結果の再検証、wrapper/schema拒否、ID衝突、stale mediaも回帰テストへ含めた。

| 受入ID | 主な証拠 |
|---|---|
| T1 | `test_core.py`: 道路区間変更を含む127step介入、対象外と保存baseline不変 |
| T2 | `test_adapter.py`: 実SB3小型MLPのpre-hook、-1→0 clip／二重正規化検出、通常predict一致 |
| T3 | `test_core.py`: N=2、T×N入力、環境生成/reset/step増加0、逐次／batch一致、同時刻p/q argmax |
| T4 | 同一の合成環境設定でbaseline127／介入50step、変更行動で次観測が変化。実P07はbaseline127より長い128step |
| T5 | 同分布0、非重複分布ln(2)、対称性、0確率、不正分布／欠測の扱い |
| T6 | `test_acceptance.py`とcore: 正負項、終了上書き、上書き後加算、別provider、strictと非有限、不一致残差保持 |
| T7 | step例外、返却後の不正tuple／reward／flag、provider/info/snapshot/GIF失敗、部分結果・実行済みreward保持 |
| T8 | 127/50の図の終端、部分／比較不能のN/A。実11行の累積和・差をrawと独立照合 |
| T9 | 合成N=2実CLI: GIF3、reward PNG3、JS PNG2、6列summaryとHTML |
| T10 | 別フォルダへの実N=10コピー、report CLI再生成、raw全983ファイルhash不変、相対媒体リンク。旧派生媒体を失敗時に残さない回帰テスト |
| T11 | 全GIFを再オープンしてstep数／100ms duration照合。複数時点のGIF、reward／JS PNG、HTMLを目視確認 |
| T12 | `test_schema.py`: LiDAR各60／全240重複・漏れ0、259全行schema、mismatch／不正index／非有限拒否 |
| T13 | `test_synthetic_262_high_index_and_alternate_provider_connect_a_b_js_report`: D262/index261と別providerを共通A/B・JS・描画へ接続。人工262 CLIも成功 |
| T14 | torch/MetaDrive/SB3/captum import禁止でPNG・HTML生成。実N=10コピーでも同じ禁止条件でreport CLI成功 |

## 最終の実走行

[実N=10 HTML](../../outputs/input_attribution/official_fixed_stress/validated-n10/report.html)

`outputs/input_attribution/official_fixed_stress/validated-n10/`。baselineを含め全11走行が`complete`／比較可能。自然な到達・逸脱は実験結果として保存し、コード失敗と分離した。合計960実行step、①-Aはbaseline全127step×10＝1,270比較。主成果物はGIF11、reward PNG11、JS PNG10、6列summary.csv、report.html。

| ID・対象 | 実行step数 | 累積返却報酬 | baselineとの差 | 終了理由 |
|---|---:|---:|---:|---|
| P00 baseline | 127 | 169.502750 | 0.000000 | arrive_dest |
| P01 road_edges | 115 | 131.060594 | -38.442156 | out_of_road |
| P02 lane_heading | 68 | 62.986104 | -106.516646 | out_of_road |
| P03 speed | 70 | 66.088049 | -103.414701 | out_of_road |
| P04 controls_history_yaw | 103 | 108.490695 | -61.012056 | out_of_road |
| P05 lane_lateral | 114 | 127.191415 | -42.311335 | out_of_road |
| P06 navigation | 68 | 62.986104 | -106.516646 | out_of_road |
| P07 lidar_front | 128 | 169.682379 | 0.179629 | arrive_dest |
| P08 lidar_left | 14 | -2.954067 | -172.456817 | out_of_road |
| P09 lidar_rear | 126 | 168.226958 | -1.275793 | arrive_dest |
| P10 lidar_right | 27 | 4.106330 | -165.396420 | out_of_road |

①-Bの対象値は全実行stepで-1に到達し、対象外は同stepの通常入力と完全一致。保存した次観測は次stepのraw観測と一致。①-Aは同じ保存pを各patternで使用し、変更後の報酬は生成していない。表示表の累積値と差は保存rawの合計と表示精度内で一致。

報酬内訳は実960stepすべて`unavailable`。実装根拠がない成分を作らず、各stepの最終返却報酬を合計として採用した。baseline終端は返却reward=10.0に対し`info.step_reward=2.1581358012919676`であり、後者を終端の合計と誤認していない。

## 合成デモと再生成

- [合成259 N=2](../../outputs/input_attribution/demo/synthetic/validated-n2/report.html): baseline127step／各介入50step。全227返却報酬に検証済み内訳。合成出力を実MetaDrive出力と分離。
- [人工262 N=2](../../outputs/input_attribution/demo/synthetic_262/validated-262/report.html): 明示した人工schemaでA/B・PNG・HTML成功。configどおりGIFは無効。実262環境の確認を示すものではない。
- コピー先: `.runtime/validation/copied-final-report/report.html`。torch／MetaDrive／SB3／captumをimport時に例外にするガードの下で統合report CLIがexit 0。生成前後・コピー元を比較して`data/`の983ファイル（全960フレームを含む）すべてhash不変。
- `report_metadata.json`の`raw_data_sha256`も983ファイルを含む。原データは再描画のために更新していない。
- 初期N=2実スモークで録画あり／なしの同じ3走行を比較し、全入力・変更入力・確率・action・rewardが全step完全一致した。描画API呼出し前後の乱数状態もadapterで検証。

## 画像・HTMLの確認

全11GIFをPillowで開き、フレーム数が各実行step数、各durationが100ms、総時間が`step数×0.1秒`であることを確認。baselineのstep0/63/126、P07到達のstep127、P08逸脱のstep13、P10逸脱のstep26等を画像として開き、車線・車両・対象・時刻・最終報酬・終了理由を確認した。例: baseline12.7秒、P07 12.8秒、P08 1.4秒。

P07のJS図で共通0〜ln(2)軸と0付近の同時刻argmax印、P08の報酬図でstep13の-5とbaseline step126の+10、それぞれの終端で線が止まることを確認。合成図の正負の内訳、baseline破線／介入太線／内訳細線も確認した。

Chromiumで1440px幅の最終HTMLを開き、条件→6列表→baseline→各①-B→各①-Aの縦配置を確認。CDPでP07／①-A／コピー先baselineへスクロールして実描画を確認し、390px幅ではGIFとrewardが縦配置、ページの横方向はみ出しなしを検証した。全32画像の読み込み完了と自然幅を確認し、コピー先GIFリンクを実際にクリックして、同フォルダ配下の600×600 image/gifへ遷移することを確認。スクリーンショット時だけDOMのloadingをeagerにし、decodeと実時間1秒を待った。HTMLファイルは変更していない。390px幅はCSS配置の確認で、実端末UA／タッチ操作の検証ではない。

検証用記録は`.runtime/validation/validated-n10-audit.json`、`copied-regeneration-audit.json`、`final-frames/`、`final-desktop.png`、`final-pattern.png`、`final-js.png`、`final-mobile.png`、`final-copied.png`。ブラウザ検証スクリプトは`browser_verify.py`。実行ログは`final-check.log`、`final-official-n10.log`、`final-report-regen.log`等。実験出力と検証用画像はローカル生成物としてGit対象外。

## 未確認事項と移植

実262次元hostの入力順序、追加位置、正規化、参照レーン、有効フラグ、独自報酬内訳は未確認。全262行unknownのtemplateを確認済みと扱わず、実adapterは未検証schemaを拒否する。VecNormalize／独自前処理、画像・連続・MultiDiscrete・再帰方策は今回のidentity MLP経路の対象外として明示的に拒否する。

移植時は追加フォルダをコピーし、[PORTING_QUICKSTART.md](PORTING_QUICKSTART.md)に示すadapterの接続関数、`reward_adapter.py`、schema／configを確認する。[COPILOT_PORT_PROMPT.md](COPILOT_PORT_PROMPT.md)は1185文字。主解析・JS・描画を移植先の報酬意味に合わせて変更する必要はない。必要なinfoが得られない場合は内訳をunavailableのまま使い、報酬関数の再実行や差額項を作らない。
