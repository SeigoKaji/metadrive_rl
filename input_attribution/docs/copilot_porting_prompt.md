# 移植先 Copilot へ渡す依頼

この依頼は、既存 MetaDrive/PPO 学習環境へ input attribution の追加 package を接続するためのものです。

1. まず `input_attribution/docs/porting.md` と `check --config ... --probe` の人間向け出力・JSONを読み、既存 Python、SB3、PyTorch、MetaDrive の版と import 元を記録してください。既存公式環境の確認例は `/home/seigo/workspace/metadrive_rl/metadrive-rl/.venv/bin/python3`、モデルはこの repository の `/home/seigo/workspace/metadrive_rl/metadrive-rl-input-attribution/models/official_baseline.zip`（SHA256: `0af58466f690f97c8e140261a5e1c8aa69a888eb0ec69c20425e551341c235db`）です。環境・モデルを別物へ置き換えないでください。
2. 既存の262次元観測を実際に作る少数のソースだけを確認し、3入力（目標レーン横ずれ、向き誤差、有効フラグ）の正確な index、符号、正規化、無効値、道路区間ごとの更新規則、関連する coupled index を根拠付きで schema に反映してください。
3. 小さな adapter のみを実装してください。`load_policy(config)` は環境を起動せず `env=None` でモデルを読み、`make_env`/`reset_env(seed)`、モデル入力前処理、Action decode、未加工の target-lane telemetry、`close_env` を接続します。`assert_contract`、`verify_schema_contract`、`source_paths` で flat Box/Discrete、全 index の根拠、参照ソースを記録します。意味不明な値を0/0.5へ置いたり、次元数から末尾 index を推測したりしないでください。
4. `train.py`、`evaluate.py`、MetaDrive本体、公式 API、既存学習設定、リポジトリ全体の再設計は行わないでください。monkey patch、観測の自動 padding/truncation、二重 normalization も禁止です。
5. 接続確認 → 少数 step probe → P00整合確認 → 本実験の順で実行し、実機で未確認の項目を成功扱いにしないでください。P00と重点 pattern の結果を保存し、report で no-op、N/A、P00との差、step/episode母数を確認してください。通常の `run` は IG を実行せず、`[ig].enabled` は依存確認だけなので、IG は対象 step と baseline を指定した `ig` サブコマンドで別途実行してください。

6. 既存runを扱うときは用途を分けてください。表示だけの再生成は `python -m input_attribution report --run-dir <old-run> --output-dir <new-report-dir>`、保存済み通常観測で新しい①-A variantを試す場合は `python -m input_attribution offline --run-dir <old-run> --config <new-analysis.toml>`、新しい①-B条件は返された子runへ `closed-loop --patterns P00,<重点pattern>` です。動画を少数だけ保存する場合は解析設定の `[video] enabled = true` と `patterns = ["P00", "<重点pattern>"]` を指定します。`closed-loop --patterns` は走行対象、`video.patterns` は動画保存対象を選びます。①-Aの子run作成で旧runのmanifest、参照観測、statusを書き換えないでください。

不明点が残る場合は推測で実装せず、check JSON に不足する source/key/値を記録し、移植先の担当者へ返してください。
