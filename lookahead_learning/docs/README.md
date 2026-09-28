# 前方注視の学習拡張

既存hostのraw観測Dに注視点3値を追加する `LookaheadEnv` です。
今回、時間Tと平面速度で参照点を選ぶ機能1と、post状態からの等速・一定曲率予測位置の誤差を加える機能2を追加しました。
従来の距離指定、PP、必要横加速度報酬の接続を維持しています。加速度推定・加速度付き予測は含みません。

[設計仕様・数値例・ログの読み方](time_prediction.md) は実装に対応しています。
別PCへ渡すときは [Copilot用の最初の1本](copilot_porting_prompt.md) と [必要時の接続手順](porting.md) を使います。

## 通常train/evaluateで3条件を実行する

3つのTOMLは `configs/official_start_lane_return_lookahead.toml` と同じ学習・環境条件を持つ独立した設定です。
PP重み0、既存必要横加速度報酬Offでそろえ、元設定を上書きしていません。条件ごとに異なる実験名でモデル・出力を分けます。
時間を使う2条件は同じT=1sです。weight=0.1、scale=1mとともに検証用初期値であり最適値ではありません。

| 条件 | 設定ファイル（rootから） | 実験名 |
|---|---|---|
| 従来の距離指定6m | lookahead_learning/examples/distance.toml | lookahead_distance |
| 機能1のみ | lookahead_learning/examples/time_only.toml | lookahead_time_only |
| 機能1＋2 | lookahead_learning/examples/time_prediction.toml | lookahead_time_prediction |

hostの既存Python環境で、同じ設定を学習と評価へ渡します。

```bash
python train.py --config lookahead_learning/examples/distance.toml
python evaluate.py --config lookahead_learning/examples/distance.toml
python train.py --config lookahead_learning/examples/time_only.toml
python evaluate.py --config lookahead_learning/examples/time_only.toml
python train.py --config lookahead_learning/examples/time_prediction.toml
python evaluate.py --config lookahead_learning/examples/time_prediction.toml
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
lateral_accel_reward_enabled = false
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

## 配布するファイル

[PORTABLE_FILES.txt](../PORTABLE_FILES.txt) に、移植元rootからの相対パスで23ファイルを列挙しています。
パッケージ内の実行コード7本、必要テスト、3設定例、最小文書です。
モデル・動画・Excel・画像・assets・無関係なレポート・移植元rootの実装は含みません。
追加の既存解説（methods.md、route_definition.md、pp_derivation.md、lateral_acceleration_reward.md）は元repoに残し、今回の最小配布には含めません。

配布物から移植先の **incoming/lookahead_learning/** へ配置してください。
実際の稼働フォルダを上書きしてからCopilotへ依頼しないでください。
同梱テストはmanifestだけを一時コピーし、元rootとMetaDrive/SB3なしで依存閉包を確認します。

導入・更新時に最初に貼る文面:

```text
incoming/lookahead_learning/docs/copilot_porting_prompt.md を最初に読み、単一エージェントで導入/更新してください。
実接続から未導入・既存版・独自改変・適用済みを判定し、変更予定ファイルと未コミット差分を先にバックアップしてください。
既存の観測・報酬・開始車線・host adapter・設定・モデルを保持し、incomingとの差分から不足分だけ適用してください。
同梱実装を再実装せず、通常train/evaluate/factory/workerと既存metadata helperを再利用し、二重wrapper・D+6・二重加算を防いでください。
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

2026-09-29、既存 `.venv` (Python 3.12.3) で確認しました。

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

再実行したpytestの範囲:

```bash
python -B -m pytest -q -p no:cacheprovider lookahead_learning tests/test_lookahead_integration.py tests/test_experiment_config.py tests/test_training_ppo_config.py tests/test_evaluation_visualization.py -k 'not one_step_connects_to_metadrive'
```

描画を伴う既存MetaDrive/GIFテスト1本はこの選択から外し、実環境は上記headless smokeで別に確認しました。
実描画・GIF生成、長時間学習、学習改善、別PCでの実移植は未検証です。
