# このMDを読んで、シーンカタログと任意scenario seedの学習・評価対応を移植してください

この文書が移植先の実装担当への依頼です。追加のチャット指示や移植元PCを参照せず、同梱ソースと移植先の既存コードを比較し、バックアップ、不足差分の適用、検証、結果報告まで進めてください。対象は、このプロジェクトから派生した既存のMetaDrive RL環境です。

完成条件は、既存の学習・評価・報酬・観測・lookahead接続とカタログ機能を保持したまま、nested configを含む `inspect_env.py` の表示、任意の `scenario_seeds` による通常の学習・評価、その短い実動作確認まで行うことです。カタログ未導入の環境では、同梱のシーンカタログ、遅い先行車・停止車、2車線での連続追い越し、統合HTML、比較再生成シェルも不足差分から導入してください。

**カタログを移植済みのPCでは、今回の差分適用、3a節のinspect・短い学習評価・metadata確認を中心に進めます。** 既存カタログのコード・設定・出力が変わらなければ、4〜7節のslow 22シーンsmoke、slow 88シーン、alternating 4例の再生成は不要です。これらはカタログ初回導入、関連処理の変更、既存結果の不一致がある場合の確認手順として残しています。長時間学習や全256シーンの自動再生成は行いません。

シーンを見るときに最初に開く入口は **`outputs/scene_catalog/index.html`** です。生成済みの正式カタログを1つにまとめ、ブラウザでfilter・検索・ページ送り・GIF自動再生・原寸表示を使います。既存の生成済みシーンから統合HTMLだけを更新でき、表示のためにシミュレーションをやり直す必要はありません。

## 0. 配布物を確認する

別PCへ運ぶものは `scene_catalog_update.zip` です。移植先リポジトリの**外側の新しい空ディレクトリ**へ展開し、中の `scene_catalog_update/START_HERE.md` を読んで実施してください。同名の既存フォルダに重ねて展開しません。

このMDだけがあるフォルダから開始した場合は、親ディレクトリの `scene_catalog_update.zip` を探し、新しい外部作業ディレクトリへ展開して同梱ソースを取得してください。ZIPもソースもない場合は、不足する配布物を報告します。元GitHub、過去チャット、別の移植プロンプトからコードを推測して作りません。

配布フォルダの内容は次の23ファイルです。コード・設定・テスト・説明の21ファイル、本MD、`SHA256SUMS`です。

| 移植先rootからの相対パス | 適用内容 |
|---|---|
| `tools/scene_catalog.py` | TOMLのパターン×seedを走行し、HTML・PNG・GIF・環境TOMLを生成 |
| `tools/scene_catalog_gallery.py` | 既存の正式5カタログの表示行を読み、統合HTMLを構築。標準ライブラリのみ、CSS/JSを内包 |
| `lead_vehicle_env.py` | private RNGによる先行車抽選、経路に沿う一定速の先行車managerと環境subclass |
| `env_factory.py` | 先行車環境選択、MetaDriveへ渡すseed包含範囲、学習workerの候補抽出wrapper。独自factoryへ必要差分をmerge |
| `configs/experiment_config.py` | `scenario_seeds` の検証・順序保持と包含範囲の導出。既存の完全実験TOML loaderを維持 |
| `inspect_env.py` | nested configの表示・指定値照合、候補指定時の先頭seed固定検査 |
| `train.py` | 通常PPO学習と候補一覧・実reset回数のmetadata保存 |
| `evaluate.py` | 候補一覧順の評価と実episode seedの記録 |
| `run_scene_catalogs.sh` | 比較カタログの再生成。実行権限を保持 |
| `configs/scene_catalog/README.md` | カタログの操作、設定、出力の説明 |
| `configs/scene_catalog/curve.toml` | カーブ5パターン×12seed |
| `configs/scene_catalog/intersection.toml` | 交差点6パターン×12seed |
| `configs/scene_catalog/overtake.toml` | 通常交通の追い越し8パターン×8seed |
| `configs/scene_catalog/overtake_slow_lead.toml` | 遅い先行車11パターン×8seed。うち車線randomは5パターン×8seed |
| `configs/scene_catalog/overtake_alternating.toml` | 移動車・停止車を0→1→0で順に抜く4パターン×seed 0 |
| `tests/test_lead_vehicles.py` | RNG分離、配置・運動・衝突・cleanup、既存lookaheadとの接続テスト |
| `tests/test_scene_catalog_gallery.py` | シミュレーションを起動せず、統合HTMLと既存カタログの取込契約を検証 |
| `tests/test_experiment_config.py` | 完全TOMLと任意seed一覧の検証・継承・stage上書き |
| `tests/test_configuration_refactor.py` | inspectのnested表示・指定値照合・seed固定と既存入口の検証 |
| `tests/test_training_ppo_config.py` | factoryの候補抽出、RNG分離、通常学習とmetadataの検証 |
| `tests/test_evaluation_visualization.py` | 一覧順の評価とepisode/既存出力の互換検証 |

展開先で `sha256sum -c SHA256SUMS`、またはPython標準ライブラリのSHA-256で、21ファイルと本MDの計22ファイルを照合してください。manifest自身は照合対象に含めません。欠落・不一致・symlink・想定外ファイルがあれば適用前に報告します。

配布フォルダは比較専用です。PYTHONPATHに追加せず、そこから `tools/scene_catalog.py` やシェルを実行しません。適用後は**移植先rootの通常モジュール**を使います。配布用 `tests/` をリポジトリ内へ残すとpytestが二重収集することがあるため、展開先はリポジトリ外に保ちます。pytest設定の変更で回避しません。

以降のコマンドは移植先rootで実行します。`PYTHON_BIN`は移植先で使用中のPython実行ファイル、`PORT_WORK`は新しい外部作業ディレクトリとして担当者が設定してください。既存 `.venv/bin/python` が使える場合はそれを選びます。

## 1. 既存状態・依存を調べ、適用前の記録を残す

最初は読み取り専用で調べます。branch、HEAD、`git status --short --untracked-files=all`、staged/unstagedの差分を保存し、同名ファイルが「未導入／旧版／部分適用／独自変更あり／適用済み」のどれかをシンボルと呼出し元から判定してください。`git diff`だけでは未追跡ファイルを含まないため、ファイルの実在と内容も確認します。

今回の21ファイルだけでは独立したシミュレーション環境になりません。移植先に次の既存接続があるか、ファイル名だけでなくシンボルと契約を確認してください。統合HTMLだけを構築する `tools/scene_catalog_gallery.py` は標準ライブラリのみで動き、MetaDriveを起動しません。

| 既存ファイル・依存 | 確認する接続 |
|---|---|
| `start_lane_env.py` | `StartLaneMetaDriveEnv`。先行車環境はこのクラスを継承し、既存開始車線タスク・reward/doneを保持する。先行車だけの設定でもimportが必要 |
| `env_factory.py` | `make_env`、`make_training_env`、`make_evaluation_env`。既存独自環境の分岐、cleanup、lookahead、Monitorの接続。今回同梱版との比較でもこれらを保持する |
| `evaluation_visualization.py` | `derive_timing(config)`の戻り値に `gif_frame_duration_ms` があり、実効 `physics_world_step_size × decision_repeat` に対応する |
| `evaluation_results.py` | `evaluate.py` がimportする `_prepare_evaluation_output_directory`、`_write_json`、録画・telemetryの既存helper。既存の評価出力契約を保持する |
| `project_paths.py` | `PROJECT_ROOT`と`OUTPUT_DIR`が移植先を指す。配布フォルダや移植元PCを指さない |
| `configs/experiment_config.py` | 同梱のloader修正を既存へmergeする。`load_experiment_config(path).profile`、train/evaluationの環境mappingと `lookahead_config`を保持する |
| `configs/official_start_lane_return_lookahead.toml` | 組合せテストが読む既存の完全な実験TOML。既存内容を保護する |
| `lookahead_learning/adapter.py`など既存lookahead | `wrap_lookahead_env`、既存設定・観測・報酬・checkpoint互換を維持する |
| `lookahead_learning/checkpoint.py` | `set_lookahead_model_metadata`、`validate_lookahead_model_metadata`。通常train/evaluateのimportとモデル再読込に必要 |
| 使用中のPython環境 | `tomllib`、NumPy、Pillow、Gymnasium、Stable-Baselines3、PyTorch、pytest、MetaDriveを利用できる。カタログでもfactoryのtop-level importにSB3が必要 |
| 使用中のMetaDrive | `TrafficDefaultVehicle`、`BaseManager`、single-agent manager、経路・lane API、`set_static`、`set_velocity`、top-down renderの互換性 |

不足や契約不明を見つけたら、その接続と不足根拠を報告してください。欠けた既存モジュールの推測実装、独自報酬の置換、別の学習runnerの追加はしません。確認できる作業は進め、未検証箇所を成功扱いしません。

移植元ではPython 3.12.3、MetaDrive 0.4.3、MetaDrive source commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`を使用しました。`tomllib`はPython 3.11以降の標準機能です。この情報は比較の基準であり、移植先のPythonやMetaDriveを自動更新する指示ではありません。

既存rootの `README.md` と `requirements.txt` を確認してください。移植元は隣接 `../metadrive` のeditable installと既存assetsを使っています。移植先ではimport先の実パス、package版、MetaDriveのHEAD、assetsの所在・版の整合を記録します。assets不足時に自動downloadを起こさないよう、環境生成の前に確認してください。仮想環境の新設、pip更新、依存の追加・更新、MetaDriveの更新、asset downloadはこの依頼に含めません。

カタログは `env.render(mode="topdown", window=False, ...)` を使いますが、移植先でMetaDrive初期化と描画backendが動くかは別途確認が必要です。既存のheadless実行方法があればそれに従い、DISPLAY値やXvfbの必要性を推測で決めません。

変更予定ファイルの適用前原本、権限、HEAD、設定、未コミット差分を `PORT_WORK` 配下の新しいバックアップへ保存し、元からなかったファイルも一覧化します。検証で上書きする既存 `outputs/scene_catalog/overtake_slow_lead/`、`outputs/scene_catalog/overtake_alternating/`、統合 `outputs/scene_catalog/index.html` も事前に退避してください。適用後diffとハッシュを残し、今回の変更だけを戻せる状態にします。

依存とassetsが利用できたら、変更前に既存rootテストの基線を記録します。既存の失敗・skipと今回の追加失敗を区別してください。

```bash
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests
```

## 2. 不足差分だけを通常入口へmergeする

同梱ファイルを同じ相対パスで比較し、新規なら追加、既存なら必要差分だけmergeします。root全体・フォルダ全体の置換、同梱 `env_factory.py` による無条件上書きはしません。適用済みの箇所は再追加しません。

今回、カタログのPython本体2本は `tools/` に配置します。移植先に旧root配置がある場合は、次の順で移行してください。

1. 旧 `./scene_catalog.py`、`./scene_catalog_gallery.py` と、新 `tools/scene_catalog.py`、`tools/scene_catalog_gallery.py` の存在・内容・未コミットの独自変更を調べ、移動元も適用前バックアップへ保存します。
2. 独自変更を保持して新配置へ不足差分をmergeし、シェル・テスト・独自の呼出し元を新パスへ切り替えます。importは `tools.scene_catalog_gallery` を使います。CLIは `python tools/scene_catalog.py ...` / `python tools/scene_catalog_gallery.py ...`、module実行は `python -m tools.scene_catalog ...` / `python -m tools.scene_catalog_gallery ...` です。既存の `tools/` フォルダ全体を置換しません。
3. 新配置に必要な変更が揃ったことを確認したら、旧root版を記録した外部バックアップへ退避し、移植先rootに二重配置や互換stubを残さないようにします。旧版に独自変更があるまま無条件削除せず、未解決の競合は根拠とともに報告します。旧root版が元からなければ作りません。

`tools/` はnamespace packageとして扱い、同梱版では `tools/__init__.py` や別bootstrapファイルを追加しません。Pythonの配置変更後も、シェルは移植先root、出力はその `outputs/scene_catalog/` を使います。

factoryの接続例は `lead_vehicles` のキーがある場合に `LeadVehicleMetaDriveEnv(config)` を選び、その後は既存lookahead接続へ進むものです。同梱版ではこの分岐を `start_lane_objective` の分岐より先に置きます。キーのない設定の既存経路、移植先独自のraw Env、reward/done、観測、開始車線タスク、lookahead mapping、Monitor/VecEnvの順序、例外時closeを保持してください。独自base classがある場合は、その契約を確認して最小の接続を行い、独自処理を脱落させません。

今回の追加対応は、同梱の `configs/experiment_config.py`、`env_factory.py`、`train.py`、`evaluate.py`、`inspect_env.py` の差分を既存ファイルへmergeします。別の学習runnerや補助runtime moduleを新設せず、既存の入口・出力契約を使います。`scenario_seeds` はloaderで検証・保持し、MetaDriveへ渡す直前に取り除きます。学習時の候補抽出wrapperは既存lookaheadとMonitorの順序を保つ位置に入れ、指定がなければ従来経路を保持してください。inspectは表示用にnestedなMappingや配列をJSON化できる形へ変換し、環境config本体の値を表示目的で書き換えません。

`lead_vehicles`は次の5キーを持ちます。台数は整数の包含範囲、距離・間隔は車両中心間の経路距離[m]、速度はkm/hです。

```toml
lead_vehicles = { count = [1, 2], distance_m = [30, 60], gap_m = [20, 40], speed_kmh = [12, 12], lane = "ego" }
```

- `count`、`distance_m`、`gap_m`、`speed_kmh`はそれぞれ `[最小, 最大]`。`lane`は `"ego"` または各車で抽選する `"random"`。
- 抽選はscenario seedを受けるprivate RNGを使い、既存の地図・交通・Python/NumPy/PyTorchのRNG列を消費しないこと。単一自車を対象とすること。
- 先行車はナビゲーション経路の指定車線へ配置し、physics tickごとに車線中心を一定速で移動して経路終端で停止すること。衝突判定を持ち、追従・回避・車線変更は行わないこと。
- 距離が最初の道路区間を超えれば次の区間へ配置すること。経路が短すぎる、または指定車線が途中で消える設定は明示的な `ValueError` とすること。
- slow比較用TOMLの速度は通常 **12 km/h**、`lane2_very_slow`だけ **5 km/h**。alternatingの移動車は **12 km/h**、停止車は **0 km/h**。同じ車線の後方車を速くすると接近・重なりが生じうるため、比較設定を勝手に速度範囲へ戻さないこと。
- 「追い越した台数」は終了時の経路進捗で自車中心より後方にいる先行車の台数。各区間の第0レーンを共通基準とし、途中の追い越しイベント累積や車体全体の通過判定へ意味を変えないこと。

slow比較用TOMLの車線randomは次の5パターンです。既存の `lane2_random_lane` を残し、4パターンを追加します。いずれも `map_config.config = "SSS"`、初期距離 `[30, 60]` m、速度 `[12, 12]` km/h、`lane = "random"`です。

| variant | 車線数 | 台数範囲 | 車間範囲[m] |
|---|---|---|---|
| `lane2_random_lane`（既存） | 2 | `[2, 2]` | `[10, 40]` |
| `lane2_random_one`（追加） | 2 | `[1, 1]` | `[20, 40]` |
| `lane2_random_one_or_two`（追加） | 2 | `[1, 2]` | `[20, 40]` |
| `lane3_random_two`（追加） | 3 | `[2, 2]` | `[20, 40]` |
| `lane3_random_three`（追加） | 3 | `[3, 3]` | `[20, 40]` |

車線は各車で独立に抽選します。3車線に3台を置いても全車線が埋まる保証はありません。既存7パターンとseed 0〜7も維持し、設定追加のためにruntimeや乱数処理を変更しません。

HTMLのGIF自動再生、画像クリックによる原寸表示、列のソート、縮尺と表示倍率を保持してください。GIFの時間は実効dtと描画間隔から計算し、最終の短い区間も実step数に対応させます。環境TOMLの出力は既存の通常train/evaluateで利用できる形を保ちます。

`tools/scene_catalog.py` の `main` で個別カタログHTMLを保存した後に、`tools.scene_catalog_gallery.build_gallery(output_dir=OUTPUT_DIR / "scene_catalog")` を呼ぶ同梱の接続もmergeしてください。これにより通常のシーン生成後に統合入口も更新されます。`build_gallery(root=None, *, output_dir=None)` はMetaDriveを起動せず、既存の正式indexの表示行を取り込みます。CSS/JS用の追加ファイルや依存パッケージは不要です。

既存の実験TOML・モデル・ログ・認証・共有設定・AGENTS・pytest設定は変更しません。検証用の完全TOMLとログは `outputs/` 配下の新しい一意な検証用ディレクトリを使います。実験名も一意にし、標準CLIが使う `models/<実験名>.zip`、`logs/tensorboard/<実験名>/`、`outputs/<実験名>/` で既存成果物を上書きしないこと。シェルの実行権限を保持します。この依頼ではcommit/pushを行いません。

## 3. 小さい検証から、対象テスト・既存テストへ進む

まず今回適用したPythonを `compile(source, filename, "exec")` で構文確認し、5つのカタログTOMLと検証用完全TOMLを `tomllib` でparseしてください。`compileall`や`py_compile`でcacheを作る必要はありません。TOMLのvariant名、seed範囲、速度、描画設定が同梱版と対応することを確認します。

```bash
bash -n run_scene_catalogs.sh
bash run_scene_catalogs.sh --help
"$PYTHON_BIN" -B tools/scene_catalog.py --help
"$PYTHON_BIN" -B tools/scene_catalog_gallery.py --help
"$PYTHON_BIN" -B train.py --help
"$PYTHON_BIN" -B evaluate.py --help
"$PYTHON_BIN" -B inspect_env.py --help
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests/test_experiment_config.py tests/test_configuration_refactor.py tests/test_training_ppo_config.py tests/test_evaluation_visualization.py
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests/test_scene_catalog_gallery.py
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests/test_lead_vehicles.py -k 'sampling or invalid_lead'
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests/test_lead_vehicles.py
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests
```

galleryテストは標準ライブラリとpytestだけで取込を検証し、MetaDriveや環境を起動しません。leadの `-k` 選択は抽選と入力検証、対象ファイル全体は実MetaDriveの配置・運動・衝突・cleanup・既存lookaheadとの接続を含みます。MetaDriveの共有engineを使うため、テストや環境生成を並列実行しません。

先行車導入時の移植元rootテストは変更前 **212件成功**、変更後 **222件成功**でした。この記録はgallery追加前のもので、移植先の合格件数を固定する値ではありません。移植先ではgalleryテストも含めて実行したコマンド・対象コード状態・結果を記録し、未実行、失敗、skipを明記してください。同じコード状態で成功した検証は、新しい変更や懸念がなければ繰り返しません。

今回のinspect・任意seed対応時の移植元テストは、変更前 **437件＋249 subtests成功**、変更後 **467件＋249 subtests成功**でした。これも移植先の固定合格件数ではなく、今回の変更に対する移植元の実測記録です。

## 3a. inspectと任意scenario seedを、短い実学習・評価で確認する

完全な実験TOMLの `[environment.train]` と `[environment.evaluation]` に、たとえば `scenario_seeds = [2, 5, 11]` を指定できるようにします。配列は非空・非負の整数・重複なしで、boolは受け付けません。順序を保持し、`start_seed` / `num_scenarios` は省略可能です。併記されていても一覧を優先し、最小から最大を含む範囲へ導出します。この例では `start_seed = 2, num_scenarios = 10`（2以上12未満）ですが、実際の候補は2・5・11だけです。MetaDriveへ `scenario_seeds` キーを渡しません。

学習はworkerごとの `random.Random(RL seed + rank)` で候補から復元抽出し、同じ候補の反復を許します。Python・NumPy・MetaDriveの既存乱数列を消費しません。評価は一覧の入力順に各候補を1回ずつ実行します。一覧なしの場合の従来の連続範囲・学習・評価・metadataの互換性も保ってください。乱数器の再現性を、異なる環境や学習条件で報酬・軌跡まで完全一致する保証と混同しません。

短い実動作確認は、次の条件で新しい完全実験TOMLを `outputs/` 内へ作ります。既存 `configs/official.toml` などの学習・評価tableをコピーし、`name` を未使用の実験名へ変更してください。同梱の完全runnerを使い、新しい検証runnerは作りません。

- 環境は同梱 `overtake_slow_lead.toml` の `[base]` に `lane2_two` の `env` を重ねたもの。生成済み `lane2_two/seed_0000.toml` があれば同じ環境部分を利用できますが、そのファイルを得るためのカタログ再生成は不要です。
- 自車はPPOです。カタログ表示用の `IDMPolicy` を学習設定へ追加せず、元の `discrete_action = true`、`discrete_throttle_dim = 3`、`discrete_steering_dim = 3` を残します。`map_config` と競合する `map`、各stageの古い環境overrideを確認します。元ファイルは変更しません。
- 2車線SSS・先行車2台・速度12 km/h・`horizon = 1000` を保持し、両stageに `scenario_seeds = [0, 3, 5]` を指定します。これは短い実測比較用の候補です。前述の `[2, 5, 11]` も設定検証・対象テストで受理されることを確認します。
- CPU、RL seed 0、2環境、rollout長256、要求学習step数3072を使います。元のPPO scalar設定とaction次元を保持し、既存モデルを更新しません。

`VERIFY_CONFIG` はその完全TOMLの実パス、`VERIFY_NAME` はその `name` に担当者が設定してください。`VERIFY_DIR` は新しい検証用ディレクトリです。次の処理は共有MetaDrive engineと学習資源を使うため直列で実行し、exit codeとログを保存します。

```bash
"$PYTHON_BIN" -B inspect_env.py --config "$VERIFY_CONFIG" --stage train
"$PYTHON_BIN" -B inspect_env.py --config "$VERIFY_CONFIG" --stage evaluation
"$PYTHON_BIN" -B train.py --config "$VERIFY_CONFIG" --device cpu --seed 0 --num-envs 2 --n-steps 256 --timesteps 3072 --log-file "$VERIFY_DIR/train.log"
"$PYTHON_BIN" -B evaluate.py --config "$VERIFY_CONFIG" --model "models/$VERIFY_NAME.zip" --device cpu --seed 0 --no-record-gif --output-prefix "$VERIFY_NAME" --log-file "$VERIFY_DIR/evaluate.log"
```

inspectではnestedな `map_config`・`lead_vehicles` を含む実効configの表示、要求した値の照合、reset/step、`check_env`、random action検査、closeまで到達することを確認します。MetaDriveが補った未指定defaultは許容し、指定したnested値が保たれていることを検査します。`scenario_seeds` 指定時は最初のresetからchecker・random走行までを一覧の先頭seedへ固定し、checkerも検査専用adapterを通します。候補全件の走査はevaluateの役割です。未指定時は従来どおりraw checkerを使い、SB3の `seed=0` と対象範囲の既知衝突がある場合だけadapterへfallbackします。adapter使用をraw検査成功と混同せず記録します。

`outputs/<実験名>/training/training_metadata.json` で `training.scenario_seeds` と `training.scenario_reset_counts` を確認します。後者は各workerが実際にresetした候補別回数の合計で、JSONのキーはseedの文字列です。終了直後の自動resetも含み、完了episode数とは異なります。候補外seedを使っていないこと、短い今回の実行でも0・3・5すべての回数が正であること、実学習step数・CPU・RL seed・解決済みPPO設定・モデル保存と再読込成功を記録してください。

`outputs/<実験名>/evaluation/` の結果JSONでは、`scenario_seeds` が `[0, 3, 5]`、`episode_count` が3、各episodeの実seedが順に0→3→5であることを確認します。包含rangeの6シーンを評価していないこと、学習時と同じ完全TOMLのhash・モデルhashが記録されていること、録画なしで評価が完了したことも確認します。短い実学習・評価が通ることと、十分な学習による追い越し・到着性能を区別して報告してください。

今回の移植元実測では、上記条件で通常CLIの学習・評価がともにexit 0となりました。学習は実3072 step・約7.49秒、実reset回数の合計は `{"0": 4, "3": 4, "5": 1}`、モデル546,242 bytesを保存し再読込に成功しました。評価は約2.77秒、0→3→5の3 episodeが各19 step、計57行のstep記録で、同じモデルhashを確認しました。結果は全3件 `out_of_road`・到着0/3です。これは短い学習とseed選択・接続の確認成功であり、運転性能の改善を示す結果ではありません。inspectは両stageと順序を変えた `[5, 0, 3]` の検査も終了し、nested表示・checker・random 50 step・closeを確認しました。移植先でも実行結果を測定し、元PCの実測をそのまま成功記録に転記しないでください。

## 4. 先に11パターン×seed 0・1の22シーンを確認する

この節から7節までは、カタログ初回導入や関連する配置・走行処理の変更時に行う確認です。カタログ移植済みで今回の差分だけを適用した場合は、既存成果物を保ち、再実行を省略します。

`configs/scene_catalog/overtake_slow_lead.toml` を読み、**seedだけを0・1の2個**へ絞った一時TOMLを `PORT_WORK` へ作ります。全11パターン、max_steps、速度、距離、間隔、描画間隔、GIF、trail設定は保持してください。元TOMLを書き換えません。

一時TOMLのstemは、既存出力と重ならない新しい名前にします。出力先はTOMLの置き場所ではなく、移植先の `OUTPUT_DIR/scene_catalog/<一時TOMLのstem>/` になるためです。

```bash
"$PYTHON_BIN" -B tools/scene_catalog.py "$PORT_WORK/<一時TOML名>.toml"
```

担当者が実際の一時ファイル名へ置き換えて実行し、22行のHTML、各行のPNG/GIF/環境TOML、先行車の台数・距離・速度・車線、経路、終了理由、追い越した台数を確認してください。

全11パターンのseed 0・1、計22本のGIFを時系列で確認します。`lane2_one`、`lane2_two`、`lane2_one_or_two`を中心に、遅い先行車へ追いつくこと、自車が車線変更して抜けることをGIFと原寸画像で確認します。2台以上の設定では車間、`lane2_curve`では車線に沿った配置・旋回、車線randomの5パターンでは台数・間隔・車線数の違いと、抜けずに追従する場合も確認します。数値列の成功だけで目視確認を代替しません。

ブラウザが使えない場合は、Pillowなど既存の手段でGIFをデコードし、前半・接近時・追い越し中・終盤の画像を時系列で確認してください。この確認とブラウザでの自動再生・原寸リンクの確認は区別して報告します。画像を見る手段自体がなければ、その部分を未検証として報告します。

## 5. 環境TOMLの再現性と既存機能との組合せを確認する

生成された `seed_XXXX.toml` は環境部分だけです。そのまま `train.py --config` や `evaluate.py --config` へ渡せる完全な実験TOMLではありません。

移植先で使用中の完全な実験TOMLを別名にコピーし、生成ファイルの内容を同名の `[environment.common]`、`[environment.train]`、`[environment.evaluation]` へ反映してください。既存の学習・評価条件、離散action設定、lookahead、独自報酬は保持します。`map_config`を採用したら競合する既存`map`を、`map`を採用したら競合する`map_config`を除き、commonだけでなく各stageのoverrideも含めて実効設定を確認します。

通常の `configs.experiment_config.load_experiment_config` でこの完全TOMLを読み、train/evaluationの両方に地図・seed・`lead_vehicles`が届くことを確認してください。同じfactoryから順番に環境を作り、resetと数stepまで確認します。長時間学習や保存済みモデルの更新は不要です。

今回同梱の `inspect_env.py` は、nestedなMetaDrive `Config` のJSON表示に対応します。旧版の表示失敗を未修正の既知制約として残さず、3a節のtrain/evaluation両stageで実効configの表示、reset、step、検査終了まで確認してください。tracebackがあれば原因を区別して報告し、表示だけの成功や、別のfactory検証だけでinspect全体を成功扱いにしません。

同seed・同実効設定・同action列で2回実行し、先行車の抽選records、地図・初期配置、位置・heading・速度の軌跡、step数・終了理由・追い越した台数を比較します。数値比較の許容差と差異を記録してください。カタログ走行は `IDMPolicy` をruntimeで設定しており、生成TOMLにはpolicy classを出力しません。カタログとの走行比較では同じpolicyを検証用の環境mappingだけへ設定し、学習用の実験設定や通常policyを変更しません。

slowの車線randomの既存パターンと追加4パターン、alternatingの移動車・停止車4例も再現確認に含めます。`lane = "random"`を固定車線へ置き換えず、同じseedで台数・距離・速度・各車線を再現できることを確認してください。alternatingはseed 0、`num_scenarios = 1`、速度0も含む設定を通常loader・factoryまで保持します。

**車両の色やGIFの画素・ファイルhashは再現判定に使いません。** 移植元では同seedで配置・軌跡・結果を再現してもGIF色は一致しないことがあります。色の違いを直すために既存乱数列やMetaDrive本体を変更しません。

既存の開始車線タスクとlookaheadを有効にした場合も、通常loader・factory経由で単一stepが進むこと、観測が既存Boxに収まること、既存lookahead観測の追加幅・報酬・終了判定が保たれること、先行車が1control step分だけ進むことを確認します。同梱 `test_config_loader_start_lane_and_lookahead_share_one_real_step` はこの接続例です。先行車キーなしの既存テストと、既存地図・交通・RNGを保持するテストも通し、wrapperの重複やreset/stepの二重呼出しを増やしません。

## 6. シェルで最終88シーンを再生成する

上記が確認できたら、移植先rootで同梱のシェルを使います。既存出力の退避を確認し、標準出力・エラーとexit codeも保存してください。`tee`を使う場合は `set -o pipefail` で失敗を見落とさないようにします。

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh slow
```

出力は `outputs/scene_catalog/overtake_slow_lead/index.html` と、各パターンの `seed_0000`〜`seed_0007` のPNG・GIF・環境TOMLです。各88ファイル、計264シーン成果物とHTMLの88行を確認してください。車線randomは5パターン×8seedの計40シーンです。別のパターンを `--variant` で実行すると同じカタログの `index.html` がその選択だけで上書きされるので、最終88行を消さないようにします。

移植元で今回の11パターン・88シーンをシェルから再生成した実測は次のとおりです。移植先でも全88シーンの結果を測定し、差があれば設定・版・実効dt・配置・軌跡の根拠から説明してください。単に元の数値へ合わせるための設定変更はしません。

- 全88シーン: **77 arrive_dest、11 timeout、衝突・逸脱0**。
- 車線randomの40シーン: **29 arrive_dest、11 timeout**。今回追加した4パターン・32シーンに限ると **27 arrive_dest、5 timeout**。
- 主要3パターン `lane2_one`、`lane2_two`、`lane2_one_or_two`: **それぞれ8/8で全先行車が終了時に自車後方となって到着**。`lane2_very_slow`、`lane3_one_or_two`、`lane2_curve`も各8/8で同じ結果でした。

| 車線randomのvariant | 到着 | timeout | 全先行車が終了時に自車後方＋到着 |
|---|---|---|---|
| `lane2_random_lane` | 2/8 | 6/8 | 2/8 |
| `lane2_random_one` | 8/8 | 0/8 | 8/8 |
| `lane2_random_one_or_two` | 3/8 | 5/8 | 3/8 |
| `lane3_random_two` | 8/8 | 0/8 | 6/8 |
| `lane3_random_three` | 8/8 | 0/8 | 5/8 |

random設定では抜けずに追従してtimeoutになる場合があり、到着しても全先行車が後方にいるとは限りません。全シーンの到着や全台追い越しを要求する仕様ではありません。

最終HTMLでもGIF自動再生と原寸リンク、複数seedの配置差、通常12/very_slow 5 km/h、2台の車間、曲線区間、random_laneの追従を確認します。

最低成功基準は、主要3パターンの**それぞれで、全先行車を追い越して到着したseedが4/8以上**であることです。どれか1パターンでも未達なら完了扱いにせず、実測結果と原因・残作業を報告してください。移植元の各8/8という記録と、この最低基準を区別します。

## 7. alternatingで連続追い越し4例を確認する

slowの88シーンとは別に `configs/scene_catalog/overtake_alternating.toml` を使います。2車線の直線 `SSS` に先行車を2台置き、自車が**車線0→1へ移って1台目を通過し、その後1→0へ戻って2台目を通過する**動きを、移動車と停止車で比較します。

| variant | 1台目の初期距離[m] | 先行車同士の間隔[m] | 各先行車の速度[km/h] |
|---|---|---|---|
| `moving_d30_g50` | 30 | 50 | 12 |
| `moving_d30_g70` | 30 | 70 | 12 |
| `stopped_d30_g50` | 30 | 50 | 0 |
| `stopped_d30_g70` | 30 | 70 | 0 |

全4例とも `count = [2, 2]`、`lane = "random"`、seed 0だけを使います。このseedでの実際の配置は1台目が車線0、2台目が車線1の `[0, 1]`です。2台目は自車から80mまたは100m先に配置されます。他のseedで同じ車線順や連続追い越しが再現する保証はありません。新しい車線指定APIを追加したり、seedを増やしても同じ動きになると説明したりしません。

停止車は、**速度0 km/hの衝突判定を持つ物理車両を故障車役にしたもの**です。故障フラグを付ける機能や、バリア・コーンなどの障害物生成ではありません。

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh alternating
```

出力は `outputs/scene_catalog/overtake_alternating/index.html` の4行と、各variantの `seed_0000.png`、`.gif`、`.toml`、計12シーン成果物です。全4本のGIFを時系列で確認し、通常loader・factory経由で生成TOMLの地図・seed・距離・間隔・速度・配置も再現してください。ブラウザが使えない場合のデコード画像による確認と、ブラウザ再生・原寸リンクの確認は前述と同様に区別します。

各例で、初期車線0、0→1、1台目の車体全体を通過、1→0、2台目の車体全体を通過、到着、の順序とstepを位置・車線の軌跡で確認します。終了時の「追い越した台数」は車両中心の比較なので、この数値2だけでは順序や車体全体の通過確認を代替できません。移動車は経路終端で停止する前に両台を通過し、停止車は元の位置で止まっていること、衝突・逸脱がないことも確認してください。

この4例は全例で0→1→0と順番の両台通過が確認できることを成功基準とします。移植先で再現しなければ、その例の実効設定・配置・軌跡・未達理由を報告し、成功扱いにしません。確認のためのruntime変更や学習は不要です。

## 8. 統合HTMLを更新し、最初の入口として確認する

カタログ移植済みで表示処理・出力に差分がなければ、既存入口の場所を記録すれば十分です。この節のgallery再構築やブラウザ操作の再検証は、初回導入、表示関連の差分適用、出力更新が必要な場合に行います。

適用後、移植先に個別カタログの生成済みindexと画像があれば、次のコマンドで**統合HTMLだけ**を作成・更新してください。環境生成やシーンの再実行は行いません。

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh gallery
```

通常のPythonからは `"$PYTHON_BIN" -B tools/scene_catalog_gallery.py` でも同じ更新ができます。CLIは必要に応じて `--root <移植先root>`、`--output-dir <カタログ出力root>` を受け付けます。既定は `tools/` の1階層上の移植先rootと、その `outputs/scene_catalog/` です。更新後は **`outputs/scene_catalog/index.html`をローカルのブラウザで開いてください**。`file://`でfilter・検索・ページ送り・GIF自動再生・原寸リンクを利用できることを確認します。

デスクトップでは左のサイドバーで絞り込み・検索・表示設定を操作し、右の比較領域が画面の高さを使って表示されることを確認してください。画像倍率25/50/75/100%と一覧の1列/2列の切替も確認します。サイドバーと比較一覧はそれぞれ独立してスクロールできます。PCではサイドバー全体を閉じると細い操作帯と展開ボタンだけになって右一覧が広がり、ボタンで元に戻せます。モバイルでは絞り込み欄が最初は折り畳まれ、開くと欄内でスクロールでき、一覧は残りの高さを使うことを確認します。狭い画面では列数の選択を保持したまま1列で表示し、広げると選択した列数へ戻ることも確認します。

統合対象は `curve`、`intersection`、`overtake`、`overtake_slow_lead`、`overtake_alternating` の正式5カタログです。設定TOMLと個別 `index.html` の両方があるものについて、そのindexに現在表示されている行だけを取り込みます。ディレクトリに残った未掲載画像、smoke/probe用や未知のカタログは追加しません。個別カタログのHTMLも引き続き使えます。

初期状態で未生成のカタログは静かに省かれます。画像・GIF・生成済みindexはこのソース配布ZIPには含まれないため、移植先の既存成果物か、この手順で生成した成果物を使います。slow 88とalternating 4を生成した後には、この92シーンが統合HTMLにも含まれることを確認してください。他のカタログが未生成でも、表示のために全256シーンの生成を強制しません。既存の生成済み5indexがあればgallery更新だけで統合できます。

移植元の既存5indexは、curve 60＋intersection 72＋overtake 32＋slow 88＋alternating 4＝256シーン、GIF136本の構成です。移植先の総件数は生成済みindexの表示行に応じて変わります。たとえば `overtake_all` の64行が表示されていれば、その64行を反映します。filter・検索・ページ送りで表示件数と選択内容が対応すること、別カタログへ切り替えてもGIFと原寸・設定リンクが正しいことを確認してください。

個別シーン生成後の自動更新接続も確認します。slowやalternatingを通常コマンドで生成した後、統合indexの対応カタログに最新行が反映され、既存の他カタログも保持されることを確認してください。ブラウザが利用できない場合は取込件数・リンクの構造検査とGIFデコード確認を記録し、ブラウザ操作や自動再生を確認したことにはしません。

## 9. その他のシェルtargetsは案内だけにする

カタログ移植済みで今回その処理を変えない場合は、統合HTMLだけを更新する `gallery` で十分です。`slow`、`alternating` は4〜7節の初回導入・関連変更の検証が必要な場合に使います。以下のその他のtargetは利用者が追加比較を希望したときのもので、移植確認のために全件を自動実行する必要はありません。

| target | 件数・呼出し先 |
|---|---|
| `curve` | `curve.toml`、5×12=60行 |
| `intersection` | `intersection.toml`、6×12=72行 |
| `overtake` | `overtake.toml`の `lane2_respawn lane3_respawn lane2_accident lane3_accident`、4×8=32行 |
| `overtake_all` | `overtake.toml`の全8×8=64行 |
| `slow` / `overtake_slow_lead` | `overtake_slow_lead.toml`の11×8=88行。車線randomは5×8=40行 |
| `alternating` | `overtake_alternating.toml`の4×seed 0=4行 |
| `gallery` | 既存の正式5indexから統合HTMLだけを更新。シミュレーションなし |
| `all` | curve → intersection → overtake → slow → alternating、計256行。`overtake_all`は含まない |

```bash
bash run_scene_catalogs.sh --help
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh overtake intersection
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh overtake_all
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh all
```

シェルは任意の作業ディレクトリから呼べ、内部で自身のある移植先rootへ移動します。Pythonの既定はそのrootの `.venv/bin/python`、変更時は `PYTHON_BIN` を渡します。`overtake`と`overtake_all`は同じ `outputs/scene_catalog/overtake/index.html` を32行/64行で上書きするため、両方の比較を残したい場合は実行前に成果物を別々に保管してください。

## 10. 日本語で結果を報告する

最後に、次をこのPCで確認した事実と未検証事項に分けて日本語で報告してください。

1. 移植先の既存状態の判定根拠、Python・MetaDrive・assets・描画環境、配布hashの照合結果。
2. 適用・merge・適用不要だったファイルと接続、保持した独自変更、旧root版の退避先と新パスへの切替状況、バックアップ先、適用後diffとhash。旧パスの呼出しが残っていないかの検索結果も記録。
3. 実行コマンド、作業場所、対象コード状態、基線と変更後のテスト結果、inspect両stage、短い実学習の実step数・モデル再読込、候補seedごとの実reset回数の合計、評価の一覧順seed・episode数、設定hash・metadataの場所。カタログ再生成を省略した場合は導入済みと判断した根拠を示す。
4. 統合HTMLと個別HTMLの場所、必要に応じて行った取込件数・GIF/原寸リンク・自動更新、slow 22/88・alternating 4例・車線random40の確認結果、通常loader/factory再現・配置/軌跡比較・開始車線/lookahead組合せ。inspectの失敗、ブラウザ未検証、skip、契約不明も明示し、今回の必須確認と省略した任意確認を区別する。
5. 今回の変更だけを撤去する手順。適用前原本・適用後差分・現在を比較し、後続変更を保持して今回の行だけを戻す。追加ファイルも後続利用を確認し、無条件削除、フォルダ置換、`reset --hard`は行わない。

必要作業が未実行なら、その理由・不足根拠・残作業を示し、移植完了と記載しません。元PCの検証記録や構文確認だけを、移植先の実動作成功・学習改善と扱わないでください。

最後に、作業のためだけに作った一時TOML・補助スクリプト・不要な展開コピーなどを確認して削除します。バックアップ、適用diff/hash、短い学習・評価に用いた完全TOML、metadata・モデル・検証ログ、最終成果物は記録した場所に残し、その保存先を報告してください。後続利用のあるものや元から存在したものを一時物として削除しません。
