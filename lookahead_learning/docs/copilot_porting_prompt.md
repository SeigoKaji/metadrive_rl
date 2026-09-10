移植先プロジェクトのrootに lookahead_learning/ は配置済みです。次の方針で既存hostへ接続してください。

【目的】
既存の start_lane_env.py にあるStartLane系Subclassの入力生成、reward_function、終了条件、Action適用を基盤として保持し、その外側へ前方注視3値と任意のPure Pursuit（PP）追加報酬を接続する。
raw観測はflat Boxの1次元float32、幅Dを使い、注視ありのpolicy入力はD+3とする。既存prefixは変更しない。

【設定と入口】
- 専用CLIを追加せず、通常の train.py と evaluate.py に同じTOMLを渡す。
- 既存TOMLへ次を追記する。[lookahead]を省略すればbaseline。
  [lookahead]
  lookahead_m = 6.0
  pp_weight = 0.0
- lookahead_mは経路弧長[m]、pp_weightはPP不一致ペナルティ係数。
  pp_weight=0.0なら注視点だけ、正値なら追加報酬も有効にする。

【読む範囲】
- 最初に rg で env factory、StartLane系Subclass、config loader、train/evaluateの呼出しを探す。
- 関連する関数とその呼出し元だけを読み、全repository・docs・geometry・全testの網羅調査は避ける。
- 既存の入力生成、reward_function、終了条件を再実装・差替え・再呼出ししない。

【接続する4つのhook】
1. 既存schemaの許可キーへ[lookahead]を追加し、unknown keyを拒否してから次を一度呼び、解決値をprofileへ保持する。
   from lookahead_learning.checkpoint import (resolve_lookahead_config, set_lookahead_model_metadata, validate_lookahead_model_metadata)
   lookahead_config = resolve_lookahead_config(raw.get("lookahead"))
2. train.py/evaluate.pyから同じlookahead_configをcommon env factoryへ渡す。
   train.pyはPPO.save()前にset_lookahead_model_metadata(model, config)、evaluate.pyはPPO.load()直後にvalidate_lookahead_model_metadata(model, config)を呼ぶ。
   ZIP属性はlookahead_configとlookahead_schema_version=1だけを使い、追加ファイルやハッシュ照合を追加しない。
3. common factoryでraw Envを作り、設定がある場合だけ次を実行する。既存の学習用Monitorはwrapperの外側へ置き、評価側にMonitorを新設しない。
   from lookahead_learning.adapter import wrap_lookahead_env
   if lookahead_config is not None:
       raw_env = wrap_lookahead_env(raw_env, **lookahead_config)
4. adapterでhostのobservation_space、Navigation、vehicle、Actionの実際の意味と単位を確認する。
   MetaDrivePreviewProviderは全てのactive lookaheadで使い、MetaDrivePPProviderはpp_weight>0だけで使う。

既存Subclass/raw Env -> LookaheadEnv -> Monitor -> VecEnvの順序を保つ。評価のreset/stepはwrapperを通し、MetaDrive固有property/renderはenv.unwrappedから読む。
LookaheadEnvは基底の返却値とr_baseを使い、raw報酬を再計算しない。pp_weight>0の追加項はpre-action pp_validかつnot terminatedかつnot truncatedでマスクし、r=r_base+r_ppとする。

【最小検証と報告】
- 同梱のtest_checkpoint.py、test_geometry.py、test_env.pyを実行する（test_checkpoint.pyだけは標準ライブラリで実行可能）。
- simulatorが利用可能なら1環境をresetし数stepでDとD+3、有限値、pp_weight=0の報酬一致を確認する。
- PPOの長時間学習、assets download、無関係な全repo/docs調査は行わず、変更ファイル、call path、コマンド、未検証host依存条件を短く報告する。
