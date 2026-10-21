# Codex依頼：設定の違うシーンを混ぜた学習・評価と、シーンカタログからの選択書き出し

対象：このリポジトリの現在の `main`（`cd13efb` 以降）。

この依頼は2部構成です。**第1部が「4. テストと確認」まで通ってから第2部に進んでください。**

- **第1部**：シーンカタログで選んだ **設定の違うシーン**（例：カーブA,B,C と 遅い先行車の追い越しD,E,F）を、**1回の学習で1つのモデルに学習させ、評価できる** ようにする（1〜5章）。
- **第2部**：シーンカタログの統合ページで、使いたいシーンにチェックを入れて **学習用・評価用の選択をファイルに書き出す** ボタンを付ける（6章）。ユーザーはそのファイルと学習用configのベースを GitHub Copilot に渡し、新しい学習用configを作らせる。

設計はユーザーと合意済みです。計画やコード案の提示だけで終わらず、実装・実機での確認まで行ってください。

**この依頼は第2部で完結させてください。** ユーザーは「第3部」以降の追加依頼を望んでいません。
- この文書の要件は、すべてこの依頼の中で実装・確認する。一部を「次の依頼」や「今後の課題」に回さない。
- 実装中に見つかった不具合や、要件を満たすのに必要な修正も、この依頼の範囲として対応する。
- どうしても解決できないもの（MetaDrive本体の変更が必要、など）だけを報告に残し、その理由と、ユーザーが自分で取れる対処を書く。

# 第1部：設定の違うシーンを混ぜた学習・評価

## 1. 背景と、確認済みの事実

### 現状の制約

学習用configの `[environment.common]` は1つしか持てず、`scenario_seeds` で選べるのは同じ設定（同じパターン）のseedだけです。設定の違うシーンを混ぜられません。

### 採用する方式（合意済み）

- 各ワーカーは MetaDrive の環境を **1つだけ** 持ち、作り直さない（同じプロセスにMetaDrive環境は1つしか置けない）。
- エピソード開始ごとに、選ばれた **全シーン（グループ×seed）から均等にランダムで1つ** 選び、そのシーンの設定に **環境の設定を書き換えてからリセット** する。

### 事前に実機・ソースで確認したこと（MetaDrive 0.4.3、`../metadrive`）

| 内容 | 根拠 |
|---|---|
| 地図設定（`map_config`）は、その seed の地図が未作成のときリセット時に読まれる | `metadrive/manager/pg_map_manager.py:52-65` |
| 作った地図は seed をキーに `map_manager.maps` に記憶される（`store_map=True` のとき）。同じ seed を別グループで使うと取り違える | `pg_map_manager.py:23, 56-65` |
| `traffic_density` はリセットのたびに読まれる | `metadrive/manager/traffic_manager.py:134` |
| `traffic_mode` / `random_traffic` は交通マネージャ生成時に1回だけ読まれ、`self.mode` / `self.random_traffic` に保持される | `traffic_manager.py:49-50` |
| 前エピソードの交通車両はモードに関係なく片付けられ、出現レーンは毎リセット再計算される | `traffic_manager.py:68, 128-136` |
| `accident_prob` はリセットのたびに読まれる | `metadrive/manager/object_manager.py:38` |
| `lead_vehicles` はリセット後に読まれる | `lead_vehicle_env.py:77` |
| `horizon` は毎stepで読まれる（**実機未確認**） | `metadrive/envs/metadrive_env.py:136` |
| 観測・行動の形はカタログの全種類で同じ（`(259,)` / `Discrete(9)`） | 実機：カーブ、遅い先行車、交差点＋交通、3車線＋交通 |

**試作で確認した切り替え方法**（1つの環境で、リセット直前に次を行う）：

1. `env.config.update(そのグループの設定)`
2. `env.engine.traffic_manager.mode = そのグループの traffic_mode`
3. `store_map=True` のときは、`env.engine.map_manager.maps` を **グループ専用の辞書に差し替える**（グループごとに seed→地図 を別々に記憶）

この方法で、カーブ／遅い先行車／交通あり（trigger・respawn）を、**同じ seed を含めて** 交互に切り替え、毎回作り直した環境と比べて次がすべて一致しました：ブロック列、車線数、地図の外接矩形、先行車台数、交通車両の台数と全位置（リセット直後と60step後）。

**実測時間**：

| 処理 | 時間 |
|---|---|
| 環境の作り直し（close＋生成＋reset） | 0.27 s |
| 切り替え後のreset（その地図を初めて作る） | 0.03〜0.08 s |
| 切り替え後のreset（記憶した地図を使う、`store_map=True`） | 0.005〜0.019 s |

注意：内部変数（`traffic_manager.mode`、`map_manager.maps`）を書き換える方法なので、**「作り直した環境と一致すること」を確かめるテストが必須** です（4章）。
試作では、環境を閉じたときに差し替え中でないグループの地図が片付けられず、終了時に `BaseMap.__del__` のエラーメッセージが出ました。`close()` で全グループの地図を確実に破棄してください。

## 2. 仕様

### 学習用configの書き方

```toml
[environment.common]              # 全グループ共通：行動形式、報酬・終了条件、store_map など
discrete_action = true
discrete_throttle_dim = 3
discrete_steering_dim = 3
log_level = 50
# store_map = true                # 省略時 true（MetaDriveの既定）

[[environment.scene_groups]]
name = "curve__C"                 # グループ名（一意）
train_seeds = [1, 5, 8]
evaluation_seeds = [1, 5, 8]      # 省略時は train_seeds と同じ
[environment.scene_groups.common] # このグループのシーン設定
map = "C"
horizon = 1000
traffic_density = 0
accident_prob = 0
random_spawn_lane_index = false

[[environment.scene_groups]]
name = "overtake_slow_lead__lane2_one"
train_seeds = [0, 3, 5]
[environment.scene_groups.common]
map_config = { type = "block_sequence", config = "SSS", lane_num = 2 }
traffic_density = 0
lead_vehicles = { count = [1, 1], distance_m = [30, 60], gap_m = [20, 40], speed_kmh = [12, 12], lane = "ego" }
...
```

### ルール

1. **グループの `common` に書けるのは「リセットごとに切り替えられるキー」だけ**。一覧は1か所の定数で持つ。初期案：`map`、`map_config`、`traffic_density`、`traffic_mode`、`accident_prob`、`lead_vehicles`、`horizon`、`random_spawn_lane_index`。
   - 各キーを「作り直した環境と一致するか」のテストで確認し、通らないキーは一覧から外して報告する（`horizon`・`random_spawn_lane_index` は未確認）。
   - 一覧外のキーがグループにあれば、`[environment.common]` へ移すよう促すエラーにする。
2. **切り替えられるキーは `[environment.common]` に書かせない**（エラー）。ベースconfigの `map = "C"` が残り、グループの `map_config` と衝突する事故を防ぐため。
3. グループ内で `map` と `map_config` を両方書いたらエラー。どちらも無ければエラー。
4. `random_traffic` は `false` 固定（`true` ならエラー）。カタログのシーンをseedで再現するため。
5. `scene_groups` を使うとき、`[environment.train]` / `[environment.evaluation]` に `start_seed`・`num_scenarios`・`scenario_seeds` があればエラー。
6. seed の検証は既存の `validate_scenario_seeds` と同じ（0以上の整数、グループ内で重複なし、空でない）。**グループ間で同じ seed は許可**。グループ名の重複はエラー。
7. MetaDrive に渡す `start_seed` / `num_scenarios` は、全グループの学習・評価seedを覆う範囲にする。
8. どれかのグループが `lead_vehicles` を持つ場合、環境は `LeadVehicleMetaDriveEnv` で作り、`lead_vehicles` の無いグループは台数0（`count = [0, 0]`）で切り替える。
9. `store_map=True`：グループごとに地図の記憶を分ける。`store_map=False`：記憶しない（毎回地図を作る）。
10. `scene_groups` が無いconfigは、今までとまったく同じに動くこと（`start_seed`/`num_scenarios`、`scenario_seeds` の既存経路）。

### 学習

- エピソード開始ごとに、全学習シーン（グループ×seed の組）から均等に1つ選ぶ。乱数は既存の `scenario_seeds` と同じく worker ごとの `random.Random(RL seed + rank)` で、グローバル乱数を変えない。
- 学習結果のメタデータに、グループ×seed ごとの使用回数を記録する。
- `start_lane_objective`（`StartLaneMetaDriveEnv`）と lookahead wrapper と組み合わせて動くこと。
- 注意：`train.py` は PPO に seed を渡していない（渡すと SB3 が `reset(seed=rank)` を呼び、一覧外のseedになる）。この前提を崩さない。

### 評価

- グループの順、各グループの `evaluation_seeds` の順に1回ずつ評価する（環境は1つのまま切り替える）。
- エピソードの出力ディレクトリ名と結果にグループ名を含める（グループ間で seed が重なるため）。
- `evaluation.json` に、全体の集計に加えて **グループごとの集計**（成功率、終了理由の内訳など、既存の集計項目に合わせる）を出す。

### `inspect_env.py`

`scene_groups` のconfigでも動くこと。各グループの少なくとも1シーンについて、学習・評価の両stageで既存のチェック（check_env、ランダム行動、close）を行う。

## 3. 制約

- **コード量を増やしすぎない**。ユーザーはコードが増えることを嫌っています。
  - 新しいPythonファイルは原則作らない。どうしても必要なら1本まで（理由を報告）。テストは既存のテストファイルへの追加を優先。
  - 目安：本体の追加は300行以内。機能追加・リファクタリング・抽象化は、この依頼に必要な分だけ。
- MetaDrive本体（`../metadrive`）は変更しない。
- 依存の追加・更新、commit・push、破壊的なGit操作はしない。
- 一時ファイルはリポジトリ外（`$TMPDIR`）に置き、最後に消す。出力は `outputs/` 配下のみ。
- シーンカタログのHTMLへの「書き出しボタン」は第2部（6章）で作る。
- 別PCへの移植資料・配布物に関係する場合は、これまでの移植の方針に沿って反映する。

## 4. テストと確認

1. **一致テスト（必須）**：1つの環境でグループを切り替えたときのシーンが、そのシーンだけで作り直した環境と一致すること。
   - 比較：ブロック列、車線数、地図の外接矩形、先行車の台数と位置、交通車両の台数と全位置（リセット直後と数十step後）。
   - 組み合わせ：カーブ、遅い先行車、交通あり trigger、交通あり respawn、`accident_prob` あり。**同じseedを複数グループで使う** ケースと、同じシーンに2回目に戻るケースを含める。
   - `store_map=True` と `False` の両方。
   - ルール1の各キー（特に `horizon`、`random_spawn_lane_index`）がリセットで切り替わること。
2. **設定の検証テスト**：ルール1〜7のエラーと、`scene_groups` が無い場合に従来と同じになること。
3. **学習の抽選テスト**：学習シーン以外が選ばれないこと、同じRL seed・rankで再現すること。
4. **後片付け**：`close()` 後に地図が残らず、終了時の `BaseMap.__del__` エラーが出ないこと。
5. **実機**：
   - カタログから次を選んだ config を `$TMPDIR` に作る：`curve` の `C`（seed 3つ）、`overtake_slow_lead` の `lane2_one`（seed 3つ）、`overtake` の `lane2_respawn`（seed 2つ、respawn）、`intersection` の `cross_traffic`（seed 2つ、trigger）。
   - `inspect_env.py` を両stageで実行して PASS。
   - `train.py` で短い学習（数千〜1万step、`--num-envs 4`）を実行し、メタデータのグループ×seed使用回数を確認する。
   - `evaluate.py` を実行し、各グループ・各seedが1回ずつ評価され、グループ別の集計とGIFが出ることを確認する。
   - 速度：同じ step 数を1グループだけで学習した場合と比べた、学習時間の差を報告する。
6. 既存テスト一式を変更の前後で実行して比較する。

## 5. ドキュメント

`configs/scene_catalog/README.md` の「選んだシーンを学習に使う」節に、`scene_groups` の書き方、ルール（特に1・2）、学習・評価での使われ方を追記する。

# 第2部：シーンカタログからの選択書き出し

## 6. 書き出しボタン

### 6.1 目的と使い方

ユーザーは統合ページ `outputs/scene_catalog/index.html`（`tools/scene_catalog_gallery.py` が生成）でシーンを見比べ、学習・評価に使うシーンにチェックを入れて **「選択を書き出す」** を押す。
書き出された `scene_selection.toml` と、学習用configのベース（例：`configs/official.toml`）を GitHub Copilot に渡し、新しい学習用configを作らせる。
**Copilot が迷わず正しいconfigを作れること** が最重要の要件。

### 6.2 画面

- 各シーンの行に **「学習」「評価」の2つのチェックボックス** を付ける。カタログ・パターンをまたいで何個でも選べる。
- ページ上部に「選択を書き出す」ボタンと、選択中の件数（学習・評価それぞれ）を表示する。「選択をすべて解除」も付ける。
- チェック状態はブラウザの localStorage に保存し、ページを開き直しても残す。
- ページは `file://` で開かれるため、書き出しは **ブラウザのダウンロード**（Blob と `download` 属性）で `scene_selection.toml` を保存する。

### 6.3 グループ分けと選択の扱い

- 第1部の **グループ = 1つの `[[environment.scene_groups]]`**。選んだシーンを、カタログ・パターン・**シーン設定（切り替えキーの値）が完全に一致するもの** ごとにまとめる。パターン名が同じでも設定が違えば別グループ（カタログの作り直しで設定が変わった場合など）。
- グループ名は `<カタログ名>__<パターン名>`。同名で設定が違う場合は連番で区別する。
- 評価にチェックがないグループは、**学習と同じseedで評価する**（ユーザー合意済み）。`evaluation_seeds` は省略せず、明示的に書き出す。
- 評価だけチェックして学習にチェックがないグループは、書き出さずに画面で警告する（第1部の仕様では `train_seeds` が必須のため）。
- `random_traffic = true` のシーンが選ばれた場合は、書き出さずに画面で警告する（第1部のルール4）。

### 6.4 書き出すファイルの形式

**それ自体が正しいTOML** で、Copilot が `[[environment.scene_groups]]` の部分をベースへ **そのまま貼れる** 形にする。

```toml
# scene_selection.toml（シーンカタログの選択。生成日時: ...）
#
# 【GitHub Copilot への指示】このファイルとベースの学習用configから、新しい学習用configを作る。
#  1. ベースをコピーし、name を新しい名前にする（指定がなければ suggested_name）。
#     [training]・[evaluation]・[lookahead] など、[environment.*] 以外はそのまま残す。
#  2. [environment.common] から、次の「シーンごとに切り替えるキー」を削除する（残すとエラー）:
#     map, map_config, traffic_density, ...（第1部の定数から自動で列挙）
#     random_traffic が true なら false にする。その他のキー（行動形式・報酬設定など）は残す。
#  3. [environment.train] / [environment.evaluation] から start_seed・num_scenarios・scenario_seeds を削除する
#     （中身が空になった table は削除する）。
#  4. このファイルの [[environment.scene_groups]] とその [environment.scene_groups.common] を、すべてそのまま末尾に貼る。
#  5. 作ったconfigで次の2つを実行し、どちらも PASS することを確認する:
#     .venv/bin/python inspect_env.py --config <新しいconfig>
#     .venv/bin/python inspect_env.py --config <新しいconfig> --stage evaluation

[selection]
schema = "scene_selection/1"
suggested_name = "scene_mix_YYYYMMDD_HHMM"
created_at = "..."

[[environment.scene_groups]]
name = "curve__C"
train_seeds = [1, 5, 8]
evaluation_seeds = [1, 5, 8]
[environment.scene_groups.common]
map = "C"
...
```

- 指示の手順2の「切り替えキー」の一覧は、**第1部で作った定数をギャラリー生成時にimportして埋め込む**（一覧を2か所に書かない）。
- グループの `common` には、各シーンの `seed_XXXX.toml` の `[environment.common]` のうち **切り替えキーだけ** を入れる。それ以外のキー（`log_level` など）はグループに入れない。
- 値のTOML表記（`map_config` のような入れ子を含む）は **Python側（ギャラリー生成時）で作ってページに埋め込み**、JavaScriptは並べるだけにする。JavaScriptでTOMLを組み立てない。
- 書き出しの組み立て処理は、JavaScriptの **純粋な関数**（選択されたシーンのデータ → ファイルの文字列）に分け、Node で単体テストできるようにする。

### 6.5 制約

- 変更は `tools/scene_catalog_gallery.py`、`tests/test_scene_catalog_gallery.py`、`configs/scene_catalog/README.md` が中心。新しいファイルは作らない。
- 各カタログ別の `index.html`（`tools/scene_catalog.py` の出力）は変えない。チェックは統合ページだけに付ける。

### 6.6 テストと確認

1. ギャラリー生成のテスト：各シーンのデータに切り替えキーの設定とTOML表記が埋め込まれること。
2. 書き出し関数のテスト（Node）：複数カタログ・複数パターン・同じseedが別グループにある・評価なし・評価だけ、の各ケース。出力を Python の `tomllib` で読めること。
3. **Copilot の代わりの確認（必須）**：書き出したファイルの冒頭の指示どおりに、**Codex自身が** `configs/official.toml` と `configs/official_start_lane_return.toml` をベースに新しいconfigを作り、`inspect_env.py` が両stageで PASS すること。指示が曖昧で迷った箇所があれば、ファイル冒頭の指示文を直す。
4. 実際の統合ページを再生成し、ブラウザで開けるならチェック・保存・書き出しを操作確認する。ブラウザが使えない場合はその旨を報告する。

### 6.7 README

`configs/scene_catalog/README.md` に、チェックと書き出しの使い方と、**Copilot に渡すプロンプトの例** を追記する。例：

```text
#file:scene_selection.toml と #file:configs/official.toml を使い、scene_selection.toml の冒頭の指示どおりに
configs/<新しい名前>.toml を作ってください。最後に指示にある inspect_env.py の2つのコマンドを実行して、結果を教えてください。
```

## 7. 報告（日本語、この順で）

1. 結論（第1部・第2部それぞれ、動いたか、残った問題）
2. 実装の概要と、変更したファイルごとの行数（新規ファイルを作った場合は理由）
3. ルール1のキー一覧の最終版と、各キーの一致テストの結果
4. 第1部の実機確認の結果（inspect_env、学習のグループ×seed使用回数、評価のグループ別集計、速度比較）
5. 第2部の確認結果（書き出し関数のテスト、Codexが指示どおりに作ったconfigと inspect_env の結果、ファイル冒頭の指示を直した箇所、画面の操作確認の有無）
6. 既存テストの変更前後の結果
7. 移植資料への反映の有無
8. 解決できなかったもの（ある場合のみ）：理由と、ユーザーが自分で取れる対処。追加の依頼を前提とした提案は書かない
