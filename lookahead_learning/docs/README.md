# 前方注視の学習拡張

既存hostのraw観測Dに注視点3値を追加する `LookaheadEnv` です。
今回、時間Tと平面速度で参照点を選ぶ機能1と、post状態からの等速・一定曲率予測位置の誤差を加える機能2を追加しました。
従来の距離指定、PP、必要横加速度報酬の接続を維持しています。加速度推定・加速度付き予測は含みません。

[設計仕様・数値例・ログの読み方](time_prediction.md) は実装に対応しています。
別PCへ渡すときは [Copilot用の最初の1本](copilot_porting_prompt.md) と [必要時の接続手順](porting.md) を使います。

## 通常train/evaluateで比較・併用する

比較用の最初の3つのTOMLは `configs/official_start_lane_return_lookahead.toml` と同じ学習・環境条件を持つ独立した設定です。
PP重み0、既存必要横加速度報酬Offでそろえ、元設定を上書きしていません。条件ごとに異なる実験名でモデル・出力を分けます。
この3条件で横加速度報酬をOffにしているのは機能1・2を分けて比較するためで、移植先の既存報酬をOffにする指示ではありません。
時間を使う設定例は同じT=1sです。weight=0.1、scale=1mとともに検証用初期値であり最適値ではありません。

| 条件 | 設定ファイル（rootから） | 実験名 |
|---|---|---|
| 従来の距離指定6m | lookahead_learning/examples/distance.toml | lookahead_distance |
| 機能1のみ | lookahead_learning/examples/time_only.toml | lookahead_time_only |
| 機能1＋2 | lookahead_learning/examples/time_prediction.toml | lookahead_time_prediction |
| 既存横加速度報酬＋機能1＋2 | lookahead_learning/examples/time_prediction_lateral.toml | lookahead_time_prediction_lateral |

hostの既存Python環境で、同じ設定を学習と評価へ渡します。

```bash
python train.py --config lookahead_learning/examples/distance.toml
python evaluate.py --config lookahead_learning/examples/distance.toml
python train.py --config lookahead_learning/examples/time_only.toml
python evaluate.py --config lookahead_learning/examples/time_only.toml
python train.py --config lookahead_learning/examples/time_prediction.toml
python evaluate.py --config lookahead_learning/examples/time_prediction.toml
python train.py --config lookahead_learning/examples/time_prediction_lateral.toml
python evaluate.py --config lookahead_learning/examples/time_prediction_lateral.toml
```

上記trainは比較実験を実行するコマンドで、軽量テストではありません（設定の300,000stepを学習します）。
この変更の検証では長時間学習を実行しません。別PCでは固有の学習・環境条件を保ち、比較用に別名TOMLを作ってください。
通常出力は `models/<name>.zip`、`outputs/<name>/training/` と `outputs/<name>/evaluation/` です。
同じ実験名での再実行は既存成果物を更新するため、保持する実験には別名を使います。

## 設定と優先順位

```toml
[lookahead]
lookahead_m = 6.0
lookahead_time_s = 1.0
pp_weight = 0.0
# 既存の横加速度報酬も使う場合。使用中の上限・重みはその値を保持する。
lateral_accel_reward_enabled = true
max_lateral_accel = 0.8
lateral_accel_weight = 0.1
prediction_reward_enabled = true
prediction_reward_weight = 0.1
prediction_error_scale_m = 1.0
```

Tがあれば **時間指定優先** でlookahead_mは未使用です。T省略・予測Offは従来の距離指定です。
[lookahead]自体を省略すればbaseline D、ありならD+3です。
予測の実効Onはenabledかつweight>0。Off/重み0は新項を計算・加算しません。
新項は `-weight * dt * 未クリップ位置誤差[m] / scale[m]`。
詳細な型・マスク・近似の限界は仕様書にまとめています。

モデルはZIP内schema3で実効設定を照合します。旧schema1/2は距離指定・予測Offとして読み取り互換を保ちます。
同じ観測次元でもTや実効報酬設定が違うモデルは評価できません。時間指定中の未使用lookahead_mは照合対象外です。

## 既存の横加速度報酬へ予測項を追加する

既存の横加速度項は、参照区間の最大絶対曲率から `v_post² * K_max` を求め、許容値 `max_lateral_accel` の超過を罰する項です。
[既存横加速度の仕様](lateral_acceleration_reward.md) と実装は維持し、予測位置誤差の項を独立して追加します。

```text
r_total = r_base + r_pp + r_lateral_accel + r_prediction
```

併用例は既存 `official_start_lane_return_lookahead_lateral_accel.toml` の横加速度設定を保持して、新4キーを加えた独立TOMLです。
移植先では例の上限・重みを上書き適用せず、使用中の `lateral_accel_reward_enabled`、`max_lateral_accel`、`lateral_accel_weight`、PP設定を保ちます。
横加速度項が独自hostのreward_functionに既に含まれる場合は、それをr_baseの一部として残し、wrapper側にも同じ項を追加しません。
時間指定への切替えでは共有previewの参照距離がL=vTになるため、横加速度項の参照区間も変わります。式・上限・重み・pre区間/post速度の時刻契約は維持します。

## 配布するファイル

[PORTABLE_FILES.txt](../PORTABLE_FILES.txt) に、移植元rootからの相対パスで26ファイルを列挙しています。
実行コード7本、標準ライブラリだけの配布ツール、必要テスト、4設定例、最小文書です。
モデル・動画・Excel・画像・assets・無関係なレポート・移植元rootの実装は含みません。
既存横加速度の仕様も配布します。追加の既存解説（methods.md、route_definition.md、pp_derivation.md）は元repoに残し、今回の最小配布には含めません。

配布ZIPを生成するコマンドです（出力先は未作成のファイルを指定します）。

```bash
python -B -m lookahead_learning.pack --output /tmp/lookahead_learning_update.zip
```

ZIPの最上位は **lookahead_learning_update/** だけです。展開先で既存のlookahead_learning/と並べて配置します。
元フォルダを直接コピーする操作や、配布フォルダ全体のリネーム・置換は行いません。
すでに同名の更新フォルダがあれば別の空の場所へ展開し、Copilotへ実際の配置パスを伝えてください。

```text
移植先root/
├── lookahead_learning/          ← 稼働中。比較・バックアップ後に必要差分だけ適用
├── lookahead_learning_update/   ← 今回の配布物。配置しただけでは実行コードは切り替わらない
│   ├── PORTABLE_FILES.txt
│   ├── docs/copilot_porting_prompt.md
│   └── ...
├── train.py
└── evaluate.py
```

manifestのlookahead_learning/は適用先の論理パスです。pack.pyはその先頭だけをlookahead_learning_update/へ置き換えます。
配布フォルダをPYTHONPATHへ追加したり、そこから学習を実行したりしません。適用後は通常のlookahead_learningを使います。
テストで、ZIPの展開が既存adapter・報酬コード・未コミットファイルを変更しないことと、元root・MetaDrive/SB3なしの依存閉包を確認します。

導入・更新時に最初に貼る文面:

```text
lookahead_learning_update/docs/copilot_porting_prompt.md を最初に読み、単一エージェントで導入/更新してください。
実接続から未導入・既存版・独自改変・適用済みを判定し、変更予定ファイルと未コミット差分を先にバックアップしてください。
既存の横加速度報酬の式・有効設定・上限・重み、観測・開始車線・host adapter・モデルを保持し、lookahead_learning_updateとの差分から不足分だけ適用してください。
既存の報酬へ今回の予測項を1回だけ追記してください。同梱実装を再実装せず、通常train/evaluate/factory/workerと既存metadata helperを再利用し、二重wrapper・D+6・二重加算を防いでください。
契約不明な接続だけ保留し、適用済みなら検証のみ。軽量テスト・CLI設定読込・可能なら数step smokeを実行し、根拠、バックアップ、変更、未検証事項、今回差分だけの戻し方を報告してください。
```

今回差分の撤去・復元時に貼る文面:

```text
今回の時間指定/予測報酬の導入差分だけを撤去してください。まずbranch・HEAD・git statusと適用前バックアップ・適用差分・現在の状態を比較してください。
元からあったlookahead、独自adapter、既存host接続、既存設定・モデル、利用者の後続変更を残し、今回追加した行/ファイルのうち後続変更と競合しない部分だけ戻してください。
稼働フォルダ全体の削除、reset --hard、バックアップの無条件復元はしないでください。競合箇所は保留して根拠を報告してください。
対応する旧設定・旧モデルのmetadata互換とD+3、通常CLI、関連軽量テストを確認し、戻した差分と残った項目を報告してください。
```

## 軽量検証

```bash
python -B -m unittest discover -s lookahead_learning -t . -p 'test_*.py'
python train.py --help
python evaluate.py --help
```

純粋な設定・幾何・予測テストは標準ライブラリのみです。fake hostはNumPy/Gymnasium、Monitor/VecEnv/ZIP確認は既存SB3を使います。
依存関係の追加・更新は不要です。テスト時にSB3をimport禁止にした隔離ケースではSB3部分だけskipします。
hostの通常入口への引数保持はrootの関連テストで別に確認しています。

## 検証記録

別フォルダ配布・横加速度併用の追加確認（2026-09-29）:

- 関連pytest: **24 passed、17 subtests passed**。配布ZIPの安全な展開、出力の上書き拒否、範囲外manifestの拒否、26ファイルの隔離コピー、4設定例の通常CLI読込を含みます。
- ZIPを模擬hostへ展開し、既存lookahead_learning/のadapter・報酬コード・独自ファイルの内容とファイル一覧が不変であることを確認しました。
- 公開factoryと実providerを使うfake hostで、使用中の横加速度上限1.2・重み0.07を保持し、非ゼロの横加速度項が予測Off/Onで同一、返却報酬の差がr_predictionだけになることを確認しました。
- この追加確認では実MetaDriveを再実行していません。実環境の記録は下記の機能1・2実装時のものです。別PCへの実適用は未実施です。

```bash
python -B -m pytest -q -p no:cacheprovider lookahead_learning/test_portability.py lookahead_learning/test_prediction_env.py tests/test_lookahead_integration.py
```

機能1・2実装時（コミットd23d3e7、2026-09-29）、既存 `.venv` (Python 3.12.3) で確認した記録です。

- 関連pytest: **237 passed、191 subtests passed、1 deselected**。数値・設定・旧schema・fake host・通常CLI設定読込・factory/worker・モデル属性・既存評価ログを含みます。
- 実SB3の小さな未学習PPOを一時ZIPへ保存/読込みし、schema3と旧schema1属性の互換、旧ZIP非変更を確認しました。利用者のモデルは使用していません。
- 実SubprocVecEnvの2workerでT/weight/scaleの保持と返却報酬を確認しました。AIへの委譲はありません。
- PORTABLE_FILESの23ファイルだけを一時コピーし、元root・MetaDrive・SB3禁止でfake hostが成立しました。純粋テストではさらにNumPy/Gymnasiumも禁止しています。SB3固有テストだけ隔離中にskipし、通常pytestでは実行しています。
- `train.py --help` と `evaluate.py --help` は成功。3設定例の通常CLI読込と学習/環境条件一致もテストしています。

実MetaDriveでは既存assetsの整合を先に確認し、ダウンロードを禁止して、各条件を順番に単一環境・描画なし・seed=5・action=7固定で12stepずつ実行しました。
実測のraw幅は259、wrapper後は262（D+3）、dt=0.1sでした。時間指定2条件と重み0は同じT=1sです。

| 条件 | 実行step | 予測有効step | 新項合計 |
|---|---:|---:|---:|
| 距離6m | 12 | 対象外（Off） | 0 |
| 時間指定のみ | 12 | 対象外（Off） | 0 |
| 時間指定＋予測報酬 | 12 | 12 | -2.0798709945e-7 |
| 時間指定＋enabled=true/weight=0 | 12 | 対象外（重み0） | 0 |

時間指定のみと重み0で、全stepの観測・報酬・終了フラグが一致しました。
予測Onではpostのqと時刻、未クリップ座標からの報酬再計算、返却reward=r_totalを確認しました。
この短い直進区間の結果から走行性能・報酬の有効性は評価していません。

機能1・2実装時に実行したpytestの範囲:

```bash
python -B -m pytest -q -p no:cacheprovider lookahead_learning tests/test_lookahead_integration.py tests/test_experiment_config.py tests/test_training_ppo_config.py tests/test_evaluation_visualization.py -k 'not one_step_connects_to_metadrive'
```

描画を伴う既存MetaDrive/GIFテスト1本はこの選択から外し、実環境は上記headless smokeで別に確認しました。
実描画・GIF生成、長時間学習、学習改善、別PCでの実移植は未検証です。
