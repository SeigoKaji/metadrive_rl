# このMDを読んで、混合シーン学習・評価とカタログの選択書き出しを移植してください

この文書が移植先の実装担当への依頼です。追加のチャット指示や移植元PCを参照せず、同梱ソースと移植先の既存コードを比較し、バックアップ、不足差分の適用、検証、結果報告まで進めてください。対象は、このプロジェクトから派生した既存のMetaDrive RL環境です。

この依頼は2部を順に完了します。第1部は、設定の異なる「グループ×seed」を `scene_groups` で選び、既存の学習・評価・報酬・観測・lookahead接続を保って1つのモデルへ混ぜて学習・評価できること。第2部は、統合カタログで学習・評価の選択を `scene_selection.toml` へ書き出し、任意の完全な実験設定をベースに、新しい正しい学習設定を作れることです。第1部のゲートを確認してから第2部へ進み、必要な不具合修正も今回の範囲で扱います。

**第1部まで移植済みのPCでは、3b節の現コード・差分・既存検証記録でゲートを確認し、足りない確認だけを行って3c節へ進みます。** カタログの配置・走行処理が変わらなければ、4〜7節のslow 22シーンsmoke、slow 88シーン、alternating 4例の再生成は不要です。3a節は従来の単一設定・任意seed経路の互換確認用です。長時間学習や全256シーンの自動再生成は行いません。

シーンを見るときに最初に開く入口は **`outputs/scene_catalog/index.html`** です。生成済みの正式カタログを1つにまとめ、ブラウザでfilter・検索・ページ送り・GIF自動再生・原寸表示を使います。既存の生成済みシーンから統合HTMLだけを更新でき、表示のためにシミュレーションをやり直す必要はありません。

## 0. 配布物を確認する

別PCへ運ぶものは `scene_catalog_update.zip` です。移植先リポジトリの**外側の新しい空ディレクトリ**へ展開し、中の `scene_catalog_update/START_HERE.md` を読んで実施してください。同名の既存フォルダに重ねて展開しません。

このMDだけがあるフォルダから開始した場合は、親ディレクトリの `scene_catalog_update.zip` を探し、新しい外部作業ディレクトリへ展開して同梱ソースを取得してください。ZIPもソースもない場合は、不足する配布物を報告します。元GitHub、過去チャット、別の移植プロンプトからコードを推測して作りません。

配布フォルダの内容は次の25ファイルです。コード・設定・テスト・説明の23ファイル、本MD、`SHA256SUMS`です。

| 移植先rootからの相対パス | 適用内容 |
|---|---|
| `tools/scene_catalog.py` | TOMLのパターン×seedを走行し、HTML・PNG・GIF・環境TOMLを生成 |
| `tools/scene_catalog_gallery.py` | 既存の正式5カタログを統合し、学習/評価チェック・保存・選択TOML書き出しを構築。標準ライブラリのみ、CSS/JSを内包 |
| `lead_vehicle_env.py` | private RNGによる先行車抽選、経路に沿う一定速の先行車managerと環境subclass |
| `env_factory.py` | 先行車環境選択、seed/scene_groupsの抽出・1環境内の切替・地図cache隔離とcleanup。独自factoryへ必要差分をmerge |
| `configs/experiment_config.py` | `scenario_seeds` / `scene_groups` の検証・順序保持と包含範囲の導出。既存の完全実験TOML loaderを維持 |
| `inspect_env.py` | nested configの表示・指定値照合、各グループの両stage検査 |
| `train.py` | 通常PPO学習とグループ×seed候補・実reset回数のmetadata保存 |
| `evaluate.py` | グループ順・seed順の評価とグループ別集計・実episode記録 |
| `evaluation_results.py` | グループ名を含むepisode出力先。グループなしの既存pathを保持 |
| `configs/example_scene_groups.toml` | 4グループ・10シーンの完全な混合学習・評価設定例 |
| `run_scene_catalogs.sh` | 比較カタログの再生成。実行権限を保持 |
| `configs/scene_catalog/README.md` | カタログの操作、設定、出力の説明 |
| `configs/scene_catalog/curve.toml` | カーブ5パターン×12seed |
| `configs/scene_catalog/intersection.toml` | 交差点6パターン×12seed |
| `configs/scene_catalog/overtake.toml` | 通常交通の追い越し8パターン×8seed |
| `configs/scene_catalog/overtake_slow_lead.toml` | 遅い先行車11パターン×8seed。うち車線randomは5パターン×8seed |
| `configs/scene_catalog/overtake_alternating.toml` | 移動車・停止車を0→1→0で順に抜く4パターン×seed 0 |
| `tests/test_lead_vehicles.py` | 先行車とRNG分離、scene_groupsの単独環境との一致・地図cleanup、既存lookaheadとの接続テスト |
| `tests/test_scene_catalog_gallery.py` | 統合HTML・選択データとTOML・Nodeによる書き出し純粋関数を検証。シミュレーションは起動しない |
| `tests/test_experiment_config.py` | 完全TOMLと任意seed一覧の検証・継承・stage上書き |
| `tests/test_configuration_refactor.py` | inspectのnested表示・指定値照合・seed固定と既存入口の検証 |
| `tests/test_training_ppo_config.py` | factoryの候補抽出、RNG分離、通常学習とmetadataの検証 |
| `tests/test_evaluation_visualization.py` | 一覧順の評価とepisode/既存出力の互換検証 |

展開先で `sha256sum -c SHA256SUMS`、またはPython標準ライブラリのSHA-256で、23ファイルと本MDの計24ファイルを照合してください。manifest自身は照合対象に含めません。欠落・不一致・symlink・想定外ファイルがあれば適用前に報告します。

配布フォルダは比較専用です。PYTHONPATHに追加せず、そこから `tools/scene_catalog.py` やシェルを実行しません。適用後は**移植先rootの通常モジュール**を使います。配布用 `tests/` をリポジトリ内へ残すとpytestが二重収集することがあるため、展開先はリポジトリ外に保ちます。pytest設定の変更で回避しません。

以降のコマンドは移植先rootで実行します。`PYTHON_BIN`は移植先で使用中のPython実行ファイル、`PORT_WORK`は新しい外部作業ディレクトリとして担当者が設定してください。既存 `.venv/bin/python` が使える場合はそれを選びます。

## 1. 既存状態・依存を調べ、適用前の記録を残す

最初は読み取り専用で調べます。branch、HEAD、`git status --short --untracked-files=all`、staged/unstagedの差分を保存し、同名ファイルが「未導入／旧版／部分適用／独自変更あり／適用済み」のどれかをシンボルと呼出し元から判定してください。`git diff`だけでは未追跡ファイルを含まないため、ファイルの実在と内容も確認します。

今回の23ファイルだけでは独立したシミュレーション環境になりません。移植先に次の既存接続があるか、ファイル名だけでなくシンボルと契約を確認してください。統合HTMLだけを構築する `tools/scene_catalog_gallery.py` は標準ライブラリのみで動き、MetaDriveを起動しません。

| 既存ファイル・依存 | 確認する接続 |
|---|---|
| `start_lane_env.py` | `StartLaneMetaDriveEnv`。先行車環境はこのクラスを継承し、既存開始車線タスク・reward/doneを保持する。先行車だけの設定でもimportが必要 |
| `env_factory.py` | `make_env`、`make_training_env`、`make_evaluation_env`。既存独自環境の分岐、cleanup、lookahead、Monitorの接続。今回同梱版との比較でもこれらを保持する |
| `evaluation_visualization.py` | `derive_timing(config)`の戻り値に `gif_frame_duration_ms` があり、実効 `physics_world_step_size × decision_repeat` に対応する |
| `evaluation_results.py` | 今回同梱版との比較でも `_prepare_evaluation_output_directory`、`_write_json`、録画・telemetryの既存helperと出力契約を保持する |
| `project_paths.py` | `PROJECT_ROOT`と`OUTPUT_DIR`が移植先を指す。配布フォルダや移植元PCを指さない |
| `configs/experiment_config.py` | 同梱のloader修正を既存へmergeする。`load_experiment_config(path).profile`、train/evaluationの環境mappingと `lookahead_config`を保持する |
| `configs/official_start_lane_return_lookahead.toml` | 組合せテストが読む既存の完全な実験TOML。既存内容を保護する |
| `configs/official.toml`、`configs/official_start_lane_return.toml` | 第2部のCopilot代替確認に使う既存の完全なベース。元ファイルの学習・評価・開始車線タスクなどを保護する |
| `lookahead_learning/adapter.py`など既存lookahead | `wrap_lookahead_env`、既存設定・観測・報酬・checkpoint互換を維持する |
| `lookahead_learning/checkpoint.py` | `set_lookahead_model_metadata`、`validate_lookahead_model_metadata`。通常train/evaluateのimportとモデル再読込に必要 |
| 使用中のPython環境 | `tomllib`、NumPy、Pillow、Gymnasium、Stable-Baselines3、PyTorch、pytest、MetaDriveを利用できる。カタログでもfactoryのtop-level importにSB3が必要 |
| 既存のNode・ブラウザ | 第2部の純粋関数試験にはNodeが必要。未導入でも勝手に追加せず不足を報告する。既存ブラウザが使えればfile://のチェック・保存・ダウンロードも確認する |
| 使用中のMetaDrive | `TrafficDefaultVehicle`、`BaseManager`、single-agent manager、経路・lane API、`set_static`、`set_velocity`、top-down renderの互換性 |

不足や契約不明を見つけたら、その接続と不足根拠を報告してください。欠けた既存モジュールの推測実装、独自報酬の置換、別の学習runnerの追加はしません。確認できる作業は進め、未検証箇所を成功扱いしません。

移植元ではPython 3.12.3、MetaDrive 0.4.3、MetaDrive source commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`を使用しました。`tomllib`はPython 3.11以降の標準機能です。この情報は比較の基準であり、移植先のPythonやMetaDriveを自動更新する指示ではありません。

既存rootの `README.md` と `requirements.txt` を確認してください。移植元は隣接 `../metadrive` のeditable installと既存assetsを使っています。移植先ではimport先の実パス、package版、MetaDriveのHEAD、assetsの所在・版の整合を記録します。assets不足時に自動downloadを起こさないよう、環境生成の前に確認してください。仮想環境の新設、pip更新、依存の追加・更新、MetaDriveの更新、asset downloadはこの依頼に含めません。

カタログは `env.render(mode="topdown", window=False, ...)` を使いますが、移植先でMetaDrive初期化と描画backendが動くかは別途確認が必要です。既存のheadless実行方法があればそれに従い、DISPLAY値やXvfbの必要性を推測で決めません。

変更予定ファイルの適用前原本、権限、HEAD、設定、未コミット差分を `PORT_WORK` 配下の新しいバックアップへ保存し、元からなかったファイルも一覧化します。検証で上書きする既存 `outputs/scene_catalog/overtake_slow_lead/`、`outputs/scene_catalog/overtake_alternating/`、統合 `outputs/scene_catalog/index.html` も事前に退避してください。適用後diffとハッシュを残し、今回の変更だけを戻せる状態にします。

依存とassetsが利用できたら、変更前に既存テストの基線を記録します。元PCの正本の対象は `tests`、`lookahead_learning`、`input_attribution/tests` の3 rootsです。移植先でも存在する既存packageのテストを含め、変更前後で同じ対象を使ってください。存在しないpackageをこのために導入せず、不在と対象外の理由を記録します。配布コピーや依存ライブラリ自身のテストは収集しません。既存の失敗・skipと今回の追加失敗を区別してください。

```bash
PORT_TEST_ROOTS=(tests)
for PORT_PACKAGE_TEST_ROOT in lookahead_learning input_attribution/tests; do
    if [ -d "$PORT_PACKAGE_TEST_ROOT" ]; then
        PORT_TEST_ROOTS+=("$PORT_PACKAGE_TEST_ROOT")
    fi
done
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider "${PORT_TEST_ROOTS[@]}"
```

## 2. 不足差分だけを通常入口へmergeする

同梱ファイルを同じ相対パスで比較し、新規なら追加、既存なら必要差分だけmergeします。root全体・フォルダ全体の置換、同梱 `env_factory.py` による無条件上書きはしません。適用済みの箇所は再追加しません。

今回、カタログのPython本体2本は `tools/` に配置します。移植先に旧root配置がある場合は、次の順で移行してください。

1. 旧 `./scene_catalog.py`、`./scene_catalog_gallery.py` と、新 `tools/scene_catalog.py`、`tools/scene_catalog_gallery.py` の存在・内容・未コミットの独自変更を調べ、移動元も適用前バックアップへ保存します。
2. 独自変更を保持して新配置へ不足差分をmergeし、シェル・テスト・独自の呼出し元を新パスへ切り替えます。importは `tools.scene_catalog_gallery` を使います。CLIは `python tools/scene_catalog.py ...` / `python tools/scene_catalog_gallery.py ...`、module実行は `python -m tools.scene_catalog ...` / `python -m tools.scene_catalog_gallery ...` です。既存の `tools/` フォルダ全体を置換しません。
3. 新配置に必要な変更が揃ったことを確認したら、旧root版を記録した外部バックアップへ退避し、移植先rootに二重配置や互換stubを残さないようにします。旧版に独自変更があるまま無条件削除せず、未解決の競合は根拠とともに報告します。旧root版が元からなければ作りません。

`tools/` はnamespace packageとして扱い、同梱版では `tools/__init__.py` や別bootstrapファイルを追加しません。Pythonの配置変更後も、シェルは移植先root、出力はその `outputs/scene_catalog/` を使います。

factoryの接続例は `lead_vehicles` のキーがある場合に `LeadVehicleMetaDriveEnv(config)` を選び、その後は既存lookahead接続へ進むものです。同梱版ではこの分岐を `start_lane_objective` の分岐より先に置きます。キーのない設定の既存経路、移植先独自のraw Env、reward/done、観測、開始車線タスク、lookahead mapping、Monitor/VecEnvの順序、例外時closeを保持してください。独自base classがある場合は、その契約を確認して最小の接続を行い、独自処理を脱落させません。

今回の追加対応は、同梱のloader・factory・train・evaluate・inspect・評価出力helperを、移植先の既存ファイルへ差分mergeします。別の学習機能やrunnerへ置き換えず、既存の入口を使います。`scene_groups` / `scenario_seeds` はプロジェクト側で解決し、MetaDriveへそのキーを渡しません。切替wrapperは既存lookaheadとMonitorの接続を保ち、未指定なら従来経路を保持します。inspectはnested configの表示だけでなく、グループごとの実効値・reset/step・checker・closeを確認してください。

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

既存の実験TOML・モデル・ログ・認証・共有設定・AGENTS・pytest設定は変更しません。標準CLIの既定は `models/<実験名>.zip`、`logs/tensorboard/<実験名>/`、`outputs/<実験名>/` ですが、今回の検証出力は3b節の方法で新しい一意な `outputs/` 配下へ集めます。一時config・一時runnerはリポジトリ外の `$TMPDIR` に置き、終了後に削除します。配布設定例は比較して導入し、移植先独自の実験設定を置換しません。シェルの実行権限を保持し、MetaDrive本体や依存の追加・更新、commit/pushは行いません。

## 3. 小さい検証から、対象テスト・既存テストへ進む

まず今回適用したPythonを `compile(source, filename, "exec")` で構文確認し、同梱TOMLと検証用完全TOMLを `tomllib` でparseしてください。`compileall`や`py_compile`でcacheを作る必要はありません。TOMLのvariant名、seed範囲、速度、描画設定が同梱版と対応することを確認します。

全体テストを別のshellで実行するときは、1節の `PORT_TEST_ROOTS` を同じ対象で再設定してから使います。

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
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider "${PORT_TEST_ROOTS[@]}"
```

galleryテストは標準ライブラリとpytestだけで取込を検証し、MetaDriveや環境を起動しません。leadの `-k` 選択は抽選と入力検証、対象ファイル全体は実MetaDriveの配置・運動・衝突・cleanup・既存lookaheadとの接続を含みます。MetaDriveの共有engineを使うため、テストや環境生成を並列実行しません。

先行車導入時の移植元rootテストは変更前 **212件成功**、変更後 **222件成功**でした。この記録はgallery追加前のもので、移植先の合格件数を固定する値ではありません。移植先ではgalleryテストも含めて実行したコマンド・対象コード状態・結果を記録し、未実行、失敗、skipを明記してください。同じコード状態で成功した検証は、新しい変更や懸念がなければ繰り返しません。

inspect・任意seed対応を導入した際の過去の移植元テストは、変更前 **437件＋249 subtests成功**、変更後 **467件＋249 subtests成功**でした。これは移植先の固定合格件数ではありません。今回のscene_groups対応は、この後の検証記録として区別してください。

第1部の元PC検証（2026-10-20）では、上記3 roots全体が変更前 **467件＋249 subtests成功**、変更後 **505件＋249 subtests成功**でした。移植先では同じ検査の記録を照合し、不足する確認を実際に行って対象と結果を記録してください。元PCの記録は移植先で実行済みという意味ではありません。第2部の結果とは分けて扱います。

## 3a. inspectと任意scenario seedを、短い実学習・評価で確認する

この節は従来の単一設定経路の確認手順です。既存導入済みなら対象テストで互換性を確認し、今回の主な実機検証は3b節で行います。以下の学習・評価を重ねて実行する必要はありません。

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

任意scenario_seeds対応を導入した際の過去の移植元実測では、上記条件で通常CLIの学習・評価がともにexit 0となりました。学習は実3072 step・約7.49秒、実reset回数の合計は `{"0": 4, "3": 4, "5": 1}`、モデル546,242 bytesを保存し再読込に成功しました。評価は約2.77秒、0→3→5の3 episodeが各19 step、計57行のstep記録で、同じモデルhashを確認しました。結果は全3件 `out_of_road`・到着0/3です。これは短い学習とseed選択・接続の確認成功であり、運転性能の改善を示す結果ではありません。inspectは両stageと順序を変えた `[5, 0, 3]` の検査も終了し、nested表示・checker・random 50 step・closeを確認しました。移植先でも実行結果を測定し、元PCの実測をそのまま成功記録に転記しないでください。

## 3b. scene_groupsを導入し、一致テスト・混合学習・全シーン評価を確認する

**ここが第1部のゲートです。** 先に導入済みかを比較し、loader・factory・train・evaluate・inspect・評価出力helperの6ファイル、関連テスト、MetaDrive/依存・実行条件と、移植先自身の既存検証記録のhash・対象コードを照合します。記録には8キーの単独環境との一致、store_map両設定、両stage inspect、短い4環境混合学習と1グループ比較、全10組の評価・GIF・cleanupを含めます。機能に関わる差分がなく有効な記録が揃っていれば再利用し、学習・評価を繰り返す必要はありません。

未導入、関連コード変更、証拠不足があれば、この節の不足部分を実行し、全条件を満たしてから3c節へ進んでください。元PCのhashや実測だけを移植先の合格根拠にはしません。これは依頼文の第1部「テストと確認」に相当し、後の旧カタログ再生成4〜7節を意味しません。

同梱 `configs/example_scene_groups.toml` は完全な実験設定です。4グループの候補は次の10組で、グループ間に同じseedを含みます。`evaluation_seeds` を省略したグループは `train_seeds` と同じ一覧を使います。

| グループ名 | train/evaluation seed | シーン |
|---|---|---|
| `curve__C` | `[0, 3, 5]` | 単一カーブC |
| `overtake_slow_lead__lane2_one` | `[0, 3, 5]` | 2車線SSS・遅い先行車1台・12 km/h |
| `overtake__lane2_respawn` | `[0, 3]` | 2車線SSS・通常交通・respawn |
| `intersection__cross_traffic` | `[0, 3]` | SXS交差点・通常交通・trigger |

各 `[[environment.scene_groups]]` に一意な `name`、非空で非負・bool不可・重複なしの `train_seeds`、任意の `evaluation_seeds`、`[environment.scene_groups.common]` を置きます。グループの `common` に許可するキーは `map`、`map_config`、`traffic_density`、`traffic_mode`、`accident_prob`、`lead_vehicles`、`horizon`、`random_spawn_lane_index` の8個だけです。`map` / `map_config` はどちらか1つが必須です。

この8キーと旧seed指定の `start_seed` / `num_scenarios` / `scenario_seeds` は、全体の `[environment.common]` やstageの `[environment.train]` / `[environment.evaluation]` に残しません。scene_groupsと混在したら明確なエラーにし、勝手に優先順位を決めません。固定する行動形式・報酬・観測・`store_map` は全体の `common` に保持します。`random_traffic` は全体で `false` 固定とし、グループ内や `true` は受け付けません。stage tableは省略できます。グループ間でのseed重複は許可します。

loaderが解決した各stageの `scene_groups` は `[{name, seeds, env_config}]` で、`env_config` は各グループの切替差分です。MetaDriveの `start_seed` / `num_scenarios` は全グループのtrain/evaluation seedを覆う包含範囲とし、独自キーはMetaDriveへ渡しません。workerは環境1つを保持し、resetの直前に設定・traffic modeを切り替えます。どこかのグループに先行車があれば先行車対応環境を使い、先行車なしのグループでは台数0に戻します。`store_map=true` はグループ別に地図cacheを分け、`false` は記憶しません。closeでは非選択グループを含む全地図を解放してください。MetaDrive本体を変更したり、episodeごとに環境を作り直したりしません。

**一致テストを先に通します。** カーブ、遅い先行車、trigger交通、respawn交通、事故ありの設定を、同じ環境内で往復して切り替えます。同seedの別グループ、同じ組への2回目の復帰、`store_map` true/falseの両方を含め、単独で作り直した環境と比較してください。reset直後と数十step後のブロック列・車線数・地図外接矩形・先行車台数と位置・通常交通の台数と全位置を照合します。8キー全部、特にhorizonと開始車線抽選の切替、グローバルRNG不変、同RL seed/rankの抽選再現、開始車線タスク/lookahead接続、close後の地図残留や `BaseMap.__del__` エラーがないことも対象です。不一致を隠すための設定変更や許容キー削減は行わず、原因と未対応を報告します。

次に同梱例を外部 `$TMPDIR` の `mixed.toml` へコピーし、未使用の実験名へ変更します。比較用 `single.toml` は最初の `curve__C` だけを残し、実験名以外のPPO設定・4環境・4096 step・CPU・RL seed 0・rollout長256・seed一覧を同じにしてください。混合側も短い接続確認であり、十分な学習を要求するものではありません。

今回の出力は `outputs/` 内の新しい一意なディレクトリだけへ保存します。`SCENE_PORT_REPO` は適用済みリポジトリroot、`SCENE_PORT_OUTPUT` はその新規出力ディレクトリ、`PORT_TMP` はリポジトリ外の新しい一時ディレクトリ、`VERIFY_MIX_NAME` / `VERIFY_SINGLE_NAME` は完全TOMLの各 `name` として担当者が設定します。前の配布フォルダをimport先に使いません。

一時 `PORT_TMP/run.py` は次の内容にします。これは出力先の定数だけを実行中に向け直して既存の `main` を呼ぶ検証用launcherです。学習・評価ロジックのコピーや恒久moduleではなく、MetaDriveやwrapperの動作を変えません。環境変数 `SCENE_PORT_REPO` / `SCENE_PORT_OUTPUT` をexportしてから使ってください。

```python
import importlib
import os
from pathlib import Path
import sys

def main():
    sys.path.insert(0, os.environ["SCENE_PORT_REPO"])
    module = importlib.import_module(sys.argv[1])
    artifacts = Path(os.environ["SCENE_PORT_OUTPUT"]) / "artifacts"
    destinations = {
        "OUTPUT_DIR": artifacts / "runs",
        "MODEL_DIR": artifacts / "models",
        "LOG_DIR": artifacts / "logs",
        "TENSORBOARD_LOG_DIR": artifacts / "tensorboard",
    }
    for key, value in destinations.items():
        if hasattr(module, key):
            setattr(module, key, value)
    return module.main(sys.argv[2:])

if __name__ == "__main__":
    raise SystemExit(main())
```

次を直列で実行し、各exit code・ログ・wall timeを保存してください。通常の引数解析・学習・評価・inspect入口を使います。既存のデータやモデルへ上書きしません。

```bash
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/mixed.toml" --stage train
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/mixed.toml" --stage evaluation
"$PYTHON_BIN" -B "$PORT_TMP/run.py" train --config "$PORT_TMP/mixed.toml" --device cpu --seed 0 --num-envs 4 --n-steps 256 --timesteps 4096 --log-file "$SCENE_PORT_OUTPUT/mixed_train.log"
"$PYTHON_BIN" -B "$PORT_TMP/run.py" train --config "$PORT_TMP/single.toml" --device cpu --seed 0 --num-envs 4 --n-steps 256 --timesteps 4096 --log-file "$SCENE_PORT_OUTPUT/single_train.log"
"$PYTHON_BIN" -B "$PORT_TMP/run.py" evaluate --config "$PORT_TMP/mixed.toml" --model "$SCENE_PORT_OUTPUT/artifacts/models/$VERIFY_MIX_NAME.zip" --device cpu --seed 0 --record-gif --output-prefix "$VERIFY_MIX_NAME" --log-file "$SCENE_PORT_OUTPUT/mixed_evaluate.log"
```

inspectは両stageで各4グループの少なくとも先頭seedを検査し、nested値、reset/step、checker、random走行、closeまで確認します。trainの `artifacts/runs/<実験名>/training/training_metadata.json` にある `training.scene_groups` と `training.scene_group_reset_counts` を照合してください。後者は `{グループ名: {seed文字列: 全workerの実reset回数}}` で、自動resetを含むため完了episode数ではありません。候補外の組がないこと、未使用の候補があれば0回として報告すること、実step数・モデル保存と再読込を記録します。混合と単一の両学習について同step数の実wall timeと差・比を報告し、episode長や地図生成回数の違いも含む比較であることを示してください。

評価は `curve__C` の0→3→5、`overtake_slow_lead__lane2_one` の0→3→5、`overtake__lane2_respawn` の0→3、`intersection__cross_traffic` の0→3の全10組が、この順で1回ずつになることを確認します。`evaluation.json` の `scene_groups`、`episodes[].scene_group`、実seed、`group_aggregates` のepisode数・既存集計項目・終了理由内訳を照合します。各episodeの出力は `episodes/episode_NNNN_<グループ名>_scenario_SSSSSS/` に分かれ、全10本のGIFが存在してデコード・再生できること、step記録と録画情報にもグループ名があることを確認してください。各GIFを時系列で確認し、ブラウザ未使用ならデコード画像の確認と区別して報告します。GIFの色・画素hashは再現基準にしません。

混合・単一の完全TOML、一時launcherの内容、source/config/modelのhash、実行コマンド・時刻・ログ・metadataを `SCENE_PORT_OUTPUT` 内の記録へ保存してから、`PORT_TMP` の今回作成した一時物を削除します。学習/評価結果が悪くても、今回の接続・候補選択・切替検証と性能を区別してください。元PCの実測は移植先で実行済みという意味ではありません。

第1部の元PC検証（2026-10-20）では、store_map両設定の一致テストが2 caseとも成功し、8キー全部を確認しました。60 step後の比較、horizon 75/90の切替、開始車線抽選の非0→0、事故確率0.01の境界、全グループの地図破棄も通っています。inspectはtrain/evaluationの各4グループでchecker・random走行・closeまで成功しました。開始車線return＋lookahead＋先行車の組合せも同じraw環境・engineのまま切り替わり、先行車0→1台、観測262次元、1回のstepでraw側が1回だけ進むことを確認しました。

CPU・4環境・rollout長256の混合/単一学習はいずれも実4096 stepで終了し、モデル保存・再読込に成功しました。混合の実reset使用回数と評価の内訳は次のとおりです。

| グループ | 混合学習の実reset回数 | 評価episode数 | 評価の終了理由 |
|---|---|---|---|
| `curve__C` | `{0: 1, 3: 1, 5: 1}` | 3 | out_of_road 3 |
| `overtake_slow_lead__lane2_one` | `{0: 1, 3: 0, 5: 0}` | 3 | out_of_road 3 |
| `overtake__lane2_respawn` | `{0: 2, 3: 0}` | 2 | out_of_road 1、crash_vehicle 1 |
| `intersection__cross_traffic` | `{0: 1, 3: 2}` | 2 | out_of_road 2 |

混合は計9 reset・10候補中7組を使用し、残り3組は有限回の抽選では選ばれず0回でした。未抽選を使用済みと説明せず、全候補を使うための追加学習も要求しません。wall timeは混合 **7.439秒**、curveだけの比較 **6.490秒**で、差は **約0.950秒・14.6%** でした。metadata内部のelapsedは5.797秒/4.887秒で、これも準備から保存・再読込までを含み、純粋なlearn時間ではありません。この1回の短い比較を一般的な速度保証にはしません。

評価は期待した全10組の順序で終了し、全グループの到着は0、合計でout_of_road 9件・crash_vehicle 1件でした。全10本のGIFは非空・複数frameで、全frameのデコードとstep時間との対応を確認し、step記録は計218行です。4グループの代表PNGを目視して道路種・車線・先行車・交通を確認しました。ブラウザでの再生確認はしていません。切替・記録機能の確認成功と、短期PPOの運転性能を区別してください。

元PCの記録は `outputs/scene_groups_review_20261020_175726/` の `baseline.json`、`pytest_after.json`、`pytest_scene_match.json`、`inspect_results.json`、`combination_probe.json`、`training_comparison.json`、`evaluation_audit.json`、`run_inputs.json` にあります。metadata・評価本体は同ディレクトリの `artifacts/runs/scene_groups_mix_20261020_175726/` 以下です。これらのoutputsはソース配布ZIPに含めず、移植先に存在する前提にも使いません。移植先では本節を実行して、そのPCの新しい出力先へ自分の記録を残してください。

元PCでは2026-10-21の第2部着手前に、現在の第1部ソースhashと上記の2026-10-20の検証記録を照合し、ゲートが引き続き有効であることを確認しました。第1部の学習・評価を本日再実行したという意味ではありません。移植先でも自分の対象コードと記録で判断します。

## 3c. 第2部：統合ページの選択書き出しとベース設定への適用を完了する

第1部のゲート確認後、同梱の `tools/scene_catalog_gallery.py`、`tests/test_scene_catalog_gallery.py`、READMEの不足差分をmergeします。学習・評価チェックは統合 `outputs/scene_catalog/index.html` だけに追加し、個別カタログのHTMLや `tools/scene_catalog.py` の出力形式は変えません。新しいruntimeファイルや追加依存は作らず、既存の倍率・列数・sidebar・filter・ページ送り・GIF/原寸リンクも保ってください。

書き出しは、カタログ・パターン・切替設定が完全一致するものをまとめ、`<カタログ>__<パターン>` をグループ名にします。同名で設定が違えば連番で区別します。各グループで評価を選ばなければ学習seedと同じ一覧を `evaluation_seeds` へ明示します。学習が0件の評価だけのグループ、`random_traffic=true`、壊れた設定TOMLの選択があれば、黙って除外せず選択全体の書き出しを止めて画面で警告してください。

切替キーは既存 `configs.experiment_config.SCENE_GROUP_SWITCH_KEYS` をギャラリー生成時にimportし、別の一覧をJavaScriptへ手書きしません。`seed_XXXX.toml` の環境commonからそのキーだけを取り込み、入れ子も含む値のTOML表記はPythonで生成してHTMLへ埋め込みます。JavaScriptの `buildSelectionExport(selectedScenes, metadata)` は選択の整理と用意済み表記の連結を行う純粋関数で、DOMやlocalStorageから独立してNodeで試験します。選択ファイルには冒頭のCopilot向け指示、`[selection]` の `schema = "scene_selection/1"`、`suggested_name = "scene_mix_YYYYMMDD_HHMM"`、UTCのISO日時 `created_at`、貼り付け可能な全 `[[environment.scene_groups]]` が必要です。

第1部の再実行を省略した場合も、検証用の新しい `SCENE_PORT_OUTPUT` と外部 `PORT_TMP` を用意します。必要な一時configと3b節の出力先だけを向け直すlauncherを作り、記録とダウンロード先は `outputs/` 配下へ集めてください。

まず既存Nodeの所在・versionを確認し、次を実行します。Nodeがなければ勝手にインストールせず不足を報告します。Node試験がskip・未実行なら、Python側だけの成功をNode試験成功とは記載しません。

```bash
node --version
"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider tests/test_scene_catalog_gallery.py
PYTHON_BIN="$PYTHON_BIN" bash run_scene_catalogs.sh gallery
```

対象テストでPythonの埋込データ/TOML表記と、Nodeで動かす純粋関数を確認します。複数カタログ・複数パターン、同seedの別グループ、同名異設定、評価未選択のfallback、評価のみ・random_traffic=trueの拒否を含め、成功した書き出しはPythonの `tomllib` でparseします。生成後の統合HTMLも実データで確認し、個別index・PNG/GIFを再生成するために旧88/256シーンを走らせることはしません。

ブラウザが使えれば、新しい統合ページを `file://` で開きます。各カードのタイトル下の「学習」「評価」、左の設定バー上部（狭い画面ではページ上部）の選択件数・「選択を書き出す」・「選択をすべて解除」を確認してください。カタログやパターンをまたいで選び、ページ送り・filter後も選択が残り、開き直してlocalStorageから復元されることを確認します。filter解除と選択解除は独立しており、設定内容が変わったシーンへ古い選択を復元しないこと、保存を利用できない場合は警告することも確認します。チェックを直して警告を解消できること、Blobとdownload属性による `scene_selection.toml` の保存も操作してください。禁止条件の入力はテスト用データで確認でき、個別カタログの成果物を書き換える必要はありません。ブラウザが使えなければその理由を記録し、NodeやHTML構造検査を画面操作確認の代わりに成功扱いしません。

**Copilotの代わりの確認を必ず行います。** 実際に書き出した `scene_selection.toml` の冒頭の指示を読み、その指示だけに従って、`configs/official.toml` と `configs/official_start_lane_return.toml` の各ベースから完全な実験TOMLを `PORT_TMP` へ作成します。ブラウザを使えない場合は同じ純粋関数の出力を使い、その取得方法を報告します。生成側の内部実装を頼りに推測で変換せず、迷う箇所があれば冒頭指示を今回修正して試験し直します。

- ベース原本は変更せず、実験名はベースごとに異なる未使用名にする。`[training]` / `[evaluation]` / `[lookahead]` など環境以外の設定と、環境内の行動形式・報酬・観測・開始車線タスクなど固定設定は保持する。
- 全体commonと各stageから切替8キー・旧seed指定を整理し、`random_traffic` はfalseにする。空になったstage tableは削除してよいが、残す独自設定があるtableは保護する。
- 旧 `scene_groups` があれば今回の全グループへ置き換え、重複追記しない。書き出しの `[[environment.scene_groups]]` とそのcommonはそのまま使い、メタ情報の `[selection]` は完全実験TOMLへ貼らない。

元ベースと完成設定を比較し、以上の保持と変換、完全TOML loaderでの受理を確認します。そのうえで、3b節の一時launcher経由でinspectを計4回実行してください。学習・評価ロジックやベース設定を変えるためのlauncherにはしません。

```bash
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/from_selection_official.toml" --stage train
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/from_selection_official.toml" --stage evaluation
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/from_selection_start_lane.toml" --stage train
"$PYTHON_BIN" -B "$PORT_TMP/run.py" inspect_env --config "$PORT_TMP/from_selection_start_lane.toml" --stage evaluation
```

両ベース・両stageで選択グループの実効値・checker・random走行・closeまで確認し、元ベースの固定設定が維持されていることを記録します。READMEにあるように、利用者はダウンロードした選択ファイルと任意ベースをCopilotへ渡し、冒頭指示に従う完全設定の作成とinspectを依頼できます。選択ファイル単独ではtrain/evaluateの完全設定ではありません。

最後に、変更前後で同じ `PORT_TEST_ROOTS` の既存テストを実行して結果を比較します。書き出しTOML、冒頭指示と変換内容、完成config、source/configのhash、Node・ブラウザのversion、コマンド・ログ・実測・未検証事項を `SCENE_PORT_OUTPUT` へ記録し、一時config・launcher・不要な展開コピーを削除します。第2部の必要作業を次の依頼へ残さず、解決できない外部制約だけは理由と利用者が取れる対処を報告してください。

第2部の元PC検証（2026-10-21）は、既存Node **v22.20.0** を使う試験を含む対象テスト **17件成功**、実 `file://` ブラウザ操作 **19項目成功**でした。正式5カタログ・256シーンの統合HTMLだけを再生成し、個別カタログの既存653ファイルは不変、従来のrandom検索も同じ56シーンを返しました。デスクトップ1280pxとモバイル390pxの画面を目視しています。

実ブラウザで保存した `scene_selection.toml` は、slowの `lane2_one` が学習 `[0, 3]`・評価 `[5]`、curveの `C` が学習 `[0, 3]`・評価fallback `[0, 3]` でした。その冒頭指示だけからofficialとofficial_start_lane_returnの完全設定を作り、name以外の非environment設定、固定するaction・報酬・開始車線設定の保持を確認しました。両ベースのtrain/evaluation inspect計4本は全exit 0で、各2グループのchecker・random走行・closeまで成功しました。wall timeはofficialのtrain/evaluationが4.33/2.02秒、start_lane_returnが2.12/2.02秒です。指示追従で追加の曖昧さや冒頭指示の修正要求はありませんでした。

正本3 rootsは、第1部完了時の **505件＋249 subtests成功** を基準として、第2部適用後は **517件＋249 subtests成功**（pytest 20.17秒、wall 21.15秒）でした。14ファイルのsource/test/base hashが実行前後で一致し、終了時の地図破棄例外もありません。これは第1部の467→505の実測とは別の、第2部505→517の記録です。

第2部の元PC記録は `outputs/scene_selection_review_20261021_114146/` の `gallery_generation.json`、`browser_checks.json`、実ダウンロードの `scene_selection.toml`、`guidance_following.json`、`inspect_results.json`、`pytest_final.json` と各ログ、完成設定の証拠コピー `configs/` 以下にあります。これらのoutputsは配布ZIPに含めません。**移植先は未検証です。** 本MDの受入確認をそのPCで行い、元PCの実測を移植先で実施済みと報告しないでください。

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

今回の第2部では選択UIを更新するため、3c節のgallery再構築を行います。統合HTMLだけを更新し、既存の個別カタログと画像を使います。すでにこの第2部も導入済みで表示処理・出力に差分がない場合は、その根拠と既存の確認記録を示してください。

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

1. 結論。第1部・第2部のそれぞれが動いたか、残った問題と未検証事項。Python・MetaDrive・assets・描画環境、配布hashの照合結果も示す。
2. 変更ファイルと行数、適用・merge・適用不要の根拠、保持した独自変更、バックアップ・適用diff/hashの保存先。旧root版がある場合は退避先とtoolsへの切替状況も記録。
3. 切替8キーの最終一覧と、それぞれの単独環境との一致テスト結果。同seedの別グループ、cache再利用、store_map両設定、開始車線/lookahead、cleanupの結果。
4. 第1部のゲートの根拠・再利用した記録・不足分の実行結果。inspect両stage、4環境混合学習の実step数・グループ×seed実reset回数・モデル再読込、同step速度比較、全10組の評価順・集計・GIF確認を報告する。元PCの記録を移植先の実行結果にしない。
5. 第2部のNode純粋関数試験、実gallery再生成、書き出したTOMLと冒頭指示に従って作ったofficial/start_laneの完全config、両stage inspectの結果。指示を直した箇所と、チェック・localStorage・ダウンロードの操作確認の有無を示す。
6. 既存テストの変更前後の対象・結果、失敗・skip・未実行とその理由。元PCの件数を移植先の固定基準にしない。
7. 移植資料の保存先、配布ZIPとMDのhash、別PCでの使い方。別PCではZIPを新しい外部ディレクトリへ展開し、担当者へその `START_HERE.md` を指定して「このMDを読んで実装してください」と依頼すればよいこと、元PCの検証結果と移植先でまだ未実施の確認を明確に分ける。
8. 解決できなかったものがある場合だけ、その理由と利用者が取れる対処を報告する。追加の依頼を前提とした提案は残さない。

必要なら今回の差分だけを撤去できるよう、適用前原本・適用後差分・現在の比較手順も記録します。後続変更を保持して今回の行だけを戻し、無条件削除、フォルダ置換、`reset --hard`は行いません。

必要作業が未実行なら、その理由・不足根拠・残作業を示し、移植完了と記載しません。元PCの検証記録や構文確認だけを、移植先の実動作成功・学習改善と扱わないでください。

最後に、作業のためだけに作った一時TOML・補助スクリプト・不要な展開コピーなどを確認して削除します。バックアップ、適用diff/hash、短い学習・評価に用いた完全TOML、metadata・モデル・検証ログ、最終成果物は記録した場所に残し、その保存先を報告してください。後続利用のあるものや元から存在したものを一時物として削除しません。
