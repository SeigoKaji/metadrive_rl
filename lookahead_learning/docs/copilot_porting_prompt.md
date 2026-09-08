# GitHub Copilot向け移植プロンプト

移植先プロジェクトのルートに `lookahead_learning/` を配置済みであることを前提に、以下のプロンプト全体をGitHub Copilotへ貼り付けてください。

```text
このリポジトリの既存環境に、配置済みの lookahead_learning/ を移植してください。

【目的】
start_lane_env.py にある MetaDriveEnv 派生クラスを使い、既存の262次元観測と独自報酬を土台に、以下のモードを動作可能にする。
- baseline：既存262次元＋既存報酬
- lookahead_obs：既存262次元＋前方注視3値＝265次元、既存報酬
- lookahead_obs_pp_reward：265次元、既存報酬＋PP操舵不一致ペナルティ

【作業方針・トークン節約】
- リポジトリの指示に従い、まず読み取り、その後に最小限の修正を行う。
- rgで関連シンボルを探し、必要な関数周辺だけ読む。全ファイル・全文書・全テストの一括読み取りや広範なリファクタリングは避ける。
- 長い計画説明やコード全文の再掲は不要。
- 関連ファイルの所在やクラス名は自分で確認する。実装から判断できない重要事項だけ質問する。

【確認と実装】
1. 移植先の環境生成経路と、実際に呼ばれる観測生成処理を確認する。
   MetaDriveでは環境のobserveではなく、観測オブジェクトのobserveが呼ばれる場合があるため、get_single_observation / agent_observation の接続も追う。
   observation_spaceとreset/stepの出力が(262,)・float32であること、末尾3値の意味・順序・符号・正規化を確認する。

2. 既存の自作Envの外側にLookaheadEnvを付ける。
   順序は「自作Env → LookaheadEnv → Monitor → VecEnv」。
   既存262値の順序・意味・正規化、独自報酬、終了条件を維持する。
   前方注視3値の追加はwrapperに任せる。

3. 接続箇所を必要に応じて修正する。
   - adapter.py：HostAdapter、環境factory、開始レーン情報の読み取り
   - runner.py：環境生成、設定読み込み、観測仕様の登録
   現行CLIはenv_factory.make_env、project_paths.py、configs/experiment_config.py等に依存する。移植先の既存構成に接続する。
   開始レーンは_target_lane_ordinals等、resolve_target_lane_state、Navigation経路との対応を確認する。

4. 現行adapterが期待する末尾3値は次のとおり。
   259: start_lane_lateral_offset
   260: start_lane_heading_error
   261: start_lane_reference_valid

   実装が一致する場合は、意味・encoding/正規化・ソース位置をPrefixFeatureEvidenceに記述し、runner.pyの_VERIFIED_HOST_PREFIX_EVIDENCEへ実際の環境クラスの完全修飾名で登録する。
   262次元という理由だけで登録しない。不一致なら内容を報告し、既存観測を維持して対応できる最小修正を判断する。padding、切り捨て、名称だけの置換で検証を通さない。

5. PPモードでは、既存報酬との二重加算、実際の適用操舵の取得、車両・Action設定の対応を確認する。
   265入力のモデルは新規学習する前提とし、既存262入力checkpointの重み変換は追加しない。

【最小検証】
編集後に行う最小の確認内容を一言示してから実施する。
- まず関連する軽量テスト。
- 実行環境が利用可能なら、1環境のresetと数回のstepで確認する。
- rawは262、拡張後は265、先頭262値は一致、値は有限で観測空間内。
- baseline / lookahead_obsの報酬はrawと一致する。
- PPモードは定義どおり追加項が入り、終了stepの追加項は0になる。
比較には同じstepで得たrawの値を使うか、乱数条件を揃える。

学習、長時間評価、依存関係のインストール、assetsのダウンロードは実行しない。
実環境で確認できない場合は、ソース確認・テスト結果と未検証事項を区別する。

【最終報告】
「変更ファイルと要点」「実行した検証と結果」「残る条件」「移植先での最小起動コマンド」を簡潔に報告する。
```
