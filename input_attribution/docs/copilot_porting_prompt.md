# 移植先 Copilot へ渡す依頼

この依頼は、既存 MetaDrive/PPO 学習環境へ input attribution の追加 package を接続するためのものです。

1. まず `input_attribution/docs/porting.md` と `check --config ... --probe` の人間向け出力・JSONを読み、既存 Python、SB3、PyTorch、MetaDrive の版と import 元を記録してください。
2. 既存の262次元観測を実際に作る少数のソースだけを確認し、3入力（目標レーン横ずれ、向き誤差、有効フラグ）の正確な index、符号、正規化、無効値、道路区間ごとの更新規則、関連する coupled index を根拠付きで schema に反映してください。
3. 小さな adapter のみを実装してください。`load_policy(config)` は環境を起動せず `env=None` でモデルを読み、`make_env`/`reset_env(seed)`、モデル入力前処理、Action decode、未加工の target-lane telemetry、`close_env` を接続します。`assert_contract`、`verify_schema_contract`、`source_paths` で flat Box/Discrete、全 index の根拠、参照ソースを記録します。意味不明な値を0/0.5へ置いたり、次元数から末尾 index を推測したりしないでください。
4. `train.py`、`evaluate.py`、MetaDrive本体、公式 API、既存学習設定、リポジトリ全体の再設計は行わないでください。monkey patch、観測の自動 padding/truncation、二重 normalization も禁止です。
5. 接続確認 → 少数 step probe → P00整合確認 → 本実験の順で実行し、実機で未確認の項目を成功扱いにしないでください。P00と重点 pattern の結果を保存し、report で no-op、N/A、P00との差、step/episode母数を確認してください。通常の `run` は IG を実行せず、`[ig].enabled` は依存確認だけなので、IG は対象 step と baseline を指定した `ig` サブコマンドで別途実行してください。

不明点が残る場合は推測で実装せず、check JSON に不足する source/key/値を記録し、移植先の担当者へ返してください。
