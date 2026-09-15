移植先rootに更新済み lookahead_learning/ を配置しました。以下を読み、単一エージェントで実装・軽量検証まで進めてください。新規移植と既存移植の更新の両方を、この依頼だけで扱ってください。

## 1. まず実際のhost接続を確認する

同梱 docs/porting.md、docs/lateral_acceleration_reward.md を必要箇所から読みます。元repository、このチャット、外部サイト、旧版フォルダは不要です。
branch・HEAD・git status を確認し、rg で次の関数と呼出し元だけを探してください。

- config loader の [lookahead] 許可キー、resolve_lookahead_config 呼出しと戻り値の保持
- 通常 train/evaluate → 共通env factory → worker への lookahead_config 受け渡し
- LookaheadEnv / wrap_lookahead_env の実接続、既存の追加3次元、Monitorとの順序
- PPOの保存／読込み時の checkpoint metadata helper 呼出し
- host固有adapterと既存の独自修正、設定を旧2キーだけに絞る処理

hostの接続から「新規」「既存版からの更新」「部分移植／独自改変」「既に今回仕様を満たす」のどれかを、根拠のファイル・関数付きで判定してください。
**フォルダの存在や同梱schema=2だけで移植済み・更新完了と判定しないでください。**

## 2. 保護する

変更予定のhostファイルと、その未コミット差分を変更前に別のバックアップ先へ保存してください。利用者の変更、既存TOML、モデルを保持します。認証・共有設定・AGENTS等は対象外です。
フォルダ差し替え前の独自変更が既に消えていた場合、この依頼からは復元できないので明示してください。新しいhost固有adapterは可能ならコピー対象パッケージの外へ置きます。大規模な再配置は行いません。

## 3. 判定に応じて接続する

### 既存版からの更新／部分移植・独自改変

既存接続を再利用し、新configキー・型・configの保持と受け渡し・metadata互換・ログなどの不足だけを直してください。lookahead_config を渡す既存設計に新hookや別runnerを足しません。
**既存wrapperをさらに包まない、D+6にしない、PP項・新項を二重加算しない** ことを確認してください。
共通resolverが更新されたことだけでhostも対応済みと決めず、hostの閉じた許可キーや2キーへの再構築を実際に確認します。host独自adapterを丸ごと上書きしません。

### 新規

既存の入力生成・reward_function・終了条件・Action・開始車線特徴・シナリオ・並列数を基盤として、その外側にlookahead wrapperを1回だけ接続します。既存のraw観測幅Dは observation_space から読み、259/262/265を固定値にしません。
同じ意味のhost接続点を探してください。StartLane系クラス名・ファイル配置の違いを許容し、移植元rootの start_lane_env.py 等をコピーしたり、その名前を必須importにしたりしません。

### 既に今回仕様を満たす

必要な検証だけ行い、重複接続や不要な差分を作らないでください。同じプロンプトの再実行でもこの分岐で保てる状態にします。

## 4. 共通の接続契約

- [lookahead] は同梱 resolve_lookahead_config(raw.get("lookahead")) で正規化します。5キーの型・値・未知キー検証を流用し、host側で旧2キーに落とさずboolを含むmapping全体を通常train/evaluate/factory/workerへ渡してください。
- tableなしはbaseline D、tableありはD+3です。新項は lateral_accel_reward_enabled=false が省略時の既定値、max_lateral_accel=0.8、lateral_accel_weight=0.1。既存Off設定を維持し、Onの比較用設定例を別に作ります。PPとは独立です。
- 共通factoryは既存Env生成後、設定がある場合だけ wrap_lookahead_env(raw_env, **lookahead_config) を呼びます。既にその接続があればそのまま使います。raw Env → LookaheadEnv → 既存Monitor → VecEnv の順序を保ち、評価にMonitorを新設しません。
- reset/step は既存wrapperチェーンを通し、unwrapped.step/reset を直呼びしません。hostの観測・reward_function・Action・終了条件を再実装しません。
- PPO.save前に set_lookahead_model_metadata、load直後（保存後再読込を含む）に validate_lookahead_model_metadata を使います。schema v2/v1 Off互換、実効設定の照合は同梱helperに任せます。未知schemaやOn不一致を迂回せず、旧ZIPを書き換えず、sidecarやハッシュ管理を増やしません。
- adapterの座標、開始車線参照、速度の意味・単位、decision dt、車線中心線半径をhostソースで確認します。PP OffならMetaDrivePPProviderは不要です。新項Off/ゼロ重みは新しい半径APIを要求しません。
- 新項Onでは既存preview/snapshotから同じ区間の K_t を保存し、postの平面速度 [m/s] で採点します。host固有providerは LateralReference、state_readerは speed_m_s の契約を満たしてください。半径APIが異なれば確認済み radius_reader を渡します。数値から単位や形状を推測しません。
- 既存の評価出力へ info["lookahead_learning"] とepisode合計・lateral_accel_episodeを渡します。root側に報酬式や集計式を複製しません。詳細な時刻、境界、マスク、数式、出力定義は同梱 lateral_acceleration_reward.md にあります。

## 5. 最小検証と完了報告

まず既存の関連テストを実行し、変更前の失敗を分けて記録してください。その後、通常の依存環境で次を実行します。

~~~bash
python -B -m unittest discover -s lookahead_learning -t . -p 'test_*.py'
python train.py --help
python evaluate.py --help
~~~

test_portability はフォルダだけを一時コピーし、元rootとMetaDriveのimportを禁止して設定・純粋関数・fake hostを検証します。
host自身でも旧schema v1＋Off、新規接続、既存版更新後、既に更新済みの再確認、PP On/Off × 新項 On/Off、ゼロ重み、D/D+3、wrapper回数、報酬の単一加算・Monitor合計を確認してください。
Simulatorが利用可能なら1環境をresetして数stepだけ確認します。長時間学習、無断の依存更新、assets download、commit/push、無関係な全repo改修は行いません。
必須契約が不明なら推測で成功扱いにせず、確認済み部分・保留部分・不足情報を報告します。

最後に、接続状態の判定根拠、変更ファイル、差分だけで済んだ接続／新規接続、バックアップ先、実行コマンド、テスト結果、未検証事項、戻し方を短く報告してください。実際の移植先やCopilotを実行していない確認を「移植済み」とは呼びません。
