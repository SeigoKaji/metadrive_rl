# シーンカタログ（scene_catalog）

MetaDriveのシーン候補を俯瞰画像で描画し、パラメータと並べたHTMLの表で見比べるツールです。
学習に使うシーンのレパートリーを増やすとき、「どの設定でどんなシーンになるか」を事前に確認するために使います。

生成済みの5カタログは、[統合一覧](../../outputs/scene_catalog/index.html)（`outputs/scene_catalog/index.html`）からまとめて見比べられます。
カタログ別の表も `outputs/scene_catalog/<TOML名>/index.html` に残ります。

- 本体: `tools/scene_catalog.py`
- 統合一覧の更新: `tools/scene_catalog_gallery.py`（シーンの再実行なし）
- 設定: このディレクトリのTOML（`curve.toml` / `overtake.toml` / `overtake_slow_lead.toml` / `overtake_alternating.toml` / `intersection.toml`）

Python本体は `tools/`、再生成の入口はプロジェクト直下の `run_scene_catalogs.sh`、成果物は `outputs/scene_catalog/` に配置します。

## 前提：シーンの形はseedで決まる

`map = "C"` のような文字列が決めるのは **ブロックの並び** だけです。
カーブの半径・角度・向き、交差点でどの出口へ進むか、交通車両の配置は **seed から抽選** され、直接は指定できません。
そのため、このツールは「パターン（ブロック列・車線数・交通量など）× seed」の組み合わせを全部描画し、
気に入ったシーンを選んでそのseedを学習設定に入れる、という使い方をします。

## 実行方法

コマンドはすべてプロジェクトルートで実行します。

```bash
# TOMLの全パターンを描画する
.venv/bin/python tools/scene_catalog.py configs/scene_catalog/<TOML名>.toml

# 一部のパターンだけ描画する（複数指定可）
.venv/bin/python tools/scene_catalog.py configs/scene_catalog/<TOML名>.toml --variant <パターン名> [<パターン名> ...]
```

シーンごとに環境を作り直して自車を走らせるため、パターン数×seed数に比例して時間がかかります。
初めて使うときは1パターンだけで試してください。

```bash
.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant C
```

比較ダンプをまとめて再生成するには、プロジェクト直下のシェルスクリプトを使えます。

```bash
bash run_scene_catalogs.sh overtake intersection  # 追い越し比較32行と交差点72行
bash run_scene_catalogs.sh slow                   # 遅い先行車88行
bash run_scene_catalogs.sh alternating            # 連続追い越し4行
bash run_scene_catalogs.sh gallery                # 生成済みシーンから統合HTMLだけを更新
bash run_scene_catalogs.sh all                    # 5種類、計256シーンを順に生成
```

`overtake` は respawn・accident の2/3車線を各8seed、計32行で再生成します。
`bash run_scene_catalogs.sh overtake_all` は全8パターン64行を生成し、**同じ `outputs/scene_catalog/overtake/index.html` を上書き**します。
`all` は curve → intersection → overtake（32行）→ slow → alternating の順です。スクリプトは任意の作業ディレクトリから呼べ、既定はプロジェクトの `.venv/bin/python`、変更するときは `PYTHON_BIN` を指定します。対象一覧は `bash run_scene_catalogs.sh --help` で確認できます。

各カタログの生成が終わると、統合一覧も自動で更新されます。既存の画像・GIF・設定ファイルを再利用し、統合HTMLだけを作り直すには `gallery` を使います。直接実行することもできます。

```bash
.venv/bin/python tools/scene_catalog_gallery.py
# 別の出力先にあるカタログをまとめる場合
.venv/bin/python tools/scene_catalog_gallery.py --output-dir /path/to/scene_catalog
```

### カーブ（`curve.toml`）

seed 0〜11 の12シーンずつ。GIFは出力しません。

| パターン名 | 内容 | 実行コマンド |
|---|---|---|
| `C` | 単一カーブ。公式サンプルと同じ（seed 5 が現在の学習シーン） | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant C` |
| `C_lane2` | 単一カーブ、2車線 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant C_lane2` |
| `CC` | 連続カーブ（同じ向きなら大きな旋回、逆向きならS字） | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant CC` |
| `CSC` | 直線を挟んだ2つのカーブ | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant CSC` |
| `CCC` | 3連続カーブ | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml --variant CCC` |

### 追い越し（`overtake.toml`）

seed 0〜7 の8シーンずつ。全パターンでGIFを出力します。
直線の7パターンは画像が横長になりすぎないよう、ブロック列を `SSS` にしています。
表の「車線変更」は、同じ道路区間内でレーン番号が変わった回数です。区間境界と同時の変更は数えないため、0でも追い越しがなかったとは断定できません。
自車と交通車両のIDM目標速度はともに30 km/hなので、車線変更が起きにくい場合があります。

| パターン名 | 内容 | 実行コマンド |
|---|---|---|
| `lane2_trigger` | 2車線の直線。交通車両は自車が近づいてから動き出す | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane2_trigger` |
| `lane3_trigger` | 3車線の直線 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane3_trigger` |
| `lane3_dense` | 3車線で交通が密 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane3_dense` |
| `lane2_respawn` | 2車線で、交通車両が最初から走っている | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane2_respawn` |
| `lane3_respawn` | 3車線で、交通車両が最初から走っている | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane3_respawn` |
| `lane2_accident` | 2車線で故障車・コーン・バリアなどを確率配置。種類や配置先はseedで変わる | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane2_accident` |
| `lane3_accident` | 3車線で故障車・コーン・バリアなどを確率配置。種類や配置先はseedで変わる | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane3_accident` |
| `curve_lane2` | カーブを含む2車線での追い越し | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant curve_lane2` |

### 遅い先行車（`overtake_slow_lead.toml`）

seed 0〜7 の8シーンずつ、11パターン計88シーンです。うちrandom車線は5パターン計40シーンで、全パターンでGIFを出力します。
通常の `traffic_density` は台数を調整する設定で、交通車両の目標速度は変わりません。自車より遅く動く先行車を置くには `lead_vehicles` を使います。
この比較用TOMLは、通常12 km/h、`lane2_very_slow` だけ5 km/hに固定しています。速度を個別に抽選すると後方車が前方車に追いついて重なったため、走行中の車間を保って追い越す動きを比較しやすい設定にしています。

```bash
.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml
```

| パターン名 | 内容 | 実行コマンド |
|---|---|---|
| `lane2_one` | 2車線、自車と同じ車線に遅い車1台 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_one` |
| `lane2_two` | 2車線、同じ車線に遅い車2台 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_two` |
| `lane2_one_or_two` | 2車線、1〜2台をseedで抽選 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_one_or_two` |
| `lane2_very_slow` | 2車線、さらに遅い先行車1〜2台 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_very_slow` |
| `lane2_random_lane` | 2車線、2台の車線を各車ごとに抽選。抜けずに追従する場合もある | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_random_lane` |
| `lane2_random_one` | 2車線、先行車1台の車線を抽選 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_random_one` |
| `lane2_random_one_or_two` | 2車線、1〜2台の台数と各車の車線を抽選 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_random_one_or_two` |
| `lane3_one_or_two` | 3車線、自車と同じ車線に1〜2台 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane3_one_or_two` |
| `lane3_random_two` | 3車線、先行車2台の車線を各車ごとに抽選 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane3_random_two` |
| `lane3_random_three` | 3車線、先行車3台の車線を各車ごとに抽選 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane3_random_three` |
| `lane2_curve` | 2車線、カーブを含む経路に1〜2台 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_slow_lead.toml --variant lane2_curve` |

random車線は各車を独立に抽選するため、同じ車線に複数台が入る場合もあります。3車線3台でも、全車線が必ず1台ずつ埋まる指定ではありません。

`lead_vehicles` はカタログの `env` と、学習用TOMLの `[environment.common]` のどちらにも書けます。

```toml
lead_vehicles = { count = [1, 2], distance_m = [30, 60], gap_m = [20, 40], speed_kmh = [10, 20], lane = "ego" }
```

| キー | 意味 |
|---|---|
| `count` | 台数の `[最小, 最大]`。両端を含む整数範囲 |
| `distance_m` | 自車中心から1台目の中心まで、車線に沿った距離 [m] の範囲 |
| `gap_m` | 2台目以降の車両中心どうしの間隔 [m] の範囲 |
| `speed_kmh` | 各車ごとに抽選する一定速度 [km/h] の範囲 |
| `lane` | `"ego"` は全台を自車と同じ車線、`"random"` は各車を同方向の車線から抽選 |

同じシナリオseedなら台数・初期位置・速度を再現できます。最初の道路区間に収まらない距離はナビゲーション経路に沿って次の区間へ配置します。
先行車は車線を保って一定速で進み、経路終端で停止します。道路長が足りない、または選んだ車線が途中でなくなる設定は `ValueError` になります。
先行車どうしの追従・減速は行わないため、同じ車線で後方車が速いと接近して重なる場合があります。
「追い越した台数」は、**走行終了時に、経路上の進捗で自車中心より後ろにいる先行車の台数**です。進捗は各区間の第0レーンを共通基準にします。途中で抜いても再び前に出られた車は含まず、車体全体の通過判定は行いません。

### 別車線の2台を連続追い越し（`overtake_alternating.toml`）

2車線の直線 `SSS` で、自車がlane0からlane1へ移って1台目を抜き、lane0へ戻って2台目を抜く、0→1→0の比較です。
seed0だけを使う4パターン計4シーンで、すべてGIFを出力します。1台目は自車の30 m先のlane0、2台目はさらに50 mまたは70 m先のlane1に配置されます。
`lane="random"` の抽選結果からこの配置のseed0を選んでいます。seedの範囲を広げても、2台が別車線に並ぶことや同じ順序で追い越すことは保証しません。

```bash
bash run_scene_catalogs.sh alternating
# または
.venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake_alternating.toml
```

| パターン名 | 内容 |
|---|---|
| `moving_d30_g50` | 車間50 m、2台とも12 km/hで車線を保って走る |
| `moving_d30_g70` | 車間70 m、2台とも12 km/hで車線を保って走る |
| `stopped_d30_g50` | 車間50 m、故障車役として2台とも0 km/hで停止する |
| `stopped_d30_g70` | 車間70 m、故障車役として2台とも0 km/hで停止する |

停止例には衝突判定のある通常の車体を使います。故障フラグ、バリア、コーンは追加していません。
出力先は `outputs/scene_catalog/overtake_alternating/index.html` です。

### 交差点（`intersection.toml`）

seed 0〜11 の12シーンずつ。GIFは `cross_traffic` だけ出力します。
表の「正味の旋回角」は、画像との照合で **+90 前後が左折、-90 前後が右折**、0 前後が直進と確認しています。
複数の交差点では左右の折れが相殺されるため、0でも全区間が直進とは限りません。
各区間の角度を -180〜180 度に正規化して合計するので、ラウンドアバウトでは実際の総回転量と異なる場合があります。

| パターン名 | 内容 | 実行コマンド |
|---|---|---|
| `cross` | 十字路 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant cross` |
| `t_junction` | T字路 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant t_junction` |
| `roundabout` | ラウンドアバウト | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant roundabout` |
| `cross_twice` | 十字路が2つ続く。ゴールの出口もseedで変わる | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant cross_twice` |
| `cross_lane2` | 2車線の十字路 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant cross_lane2` |
| `cross_traffic` | 交通車両ありの十字路 | `.venv/bin/python tools/scene_catalog.py configs/scene_catalog/intersection.toml --variant cross_traffic` |

## 出力の見方

出力先は `outputs/scene_catalog/` です。

```
outputs/scene_catalog/
  index.html                    ← 5カタログの統合一覧
  <TOML名>/
    index.html                  ← カタログ別の一覧表
    <パターン名>/seed_0000.png   ← 最終フレームの俯瞰画像
    <パターン名>/seed_0000.gif   ← gif = true のパターンだけ
    <パターン名>/seed_0000.toml  ← このシーンを学習で使うための設定
```

`--variant` で絞って実行すると、カタログ別の `index.html` はそのパターンだけの表で上書きされます。統合一覧もその表に載っている行だけを取り込み、ディレクトリに残っている過去の画像は追加しません。

### 統合一覧（`outputs/scene_catalog/index.html`）

正式な5カタログの生成済みの行をまとめて表示します。デスクトップでは左のサイドバーでカタログ・パターン・終了理由の絞り込み、文字列や `seed 0` の検索、表示設定を操作します。
右の比較領域は画面の高さを使って表示し、サイドバーと一覧をそれぞれ独立してスクロールできます。
PCではサイドバー全体を閉じると細い操作帯と展開ボタンだけになって右一覧が広がり、ボタンで元に戻せます。
モバイルでは絞り込み欄が最初は折り畳まれ、開いた欄の中をスクロールできます。一覧は残りの高さに表示します。
GIFは一覧で自動再生され、クリックすると原寸GIFが開きます。PNGだけのシーンも同じ一覧で比較できます。
表示件数・並び順に加え、画像倍率（25/50/75/100%、既定75%）と一覧の1列/2列（既定1列）を選べます。狭い画面では選択を保持したまま1列で表示し、広げると選択した列数に戻ります。横長の画像は左右へスクロールできます。
各シーンから学習用の設定TOMLや元のカタログ別一覧へ進めます。

### カタログ別一覧（`<TOML名>/index.html`）

- 上部に全パターン共通の設定、その下に1シーン1行の一覧表があります。
- 画像は同じTOML内で縮尺（既定5 px/m）を揃え、一覧表では全行75%の大きさで表示します。約4.5 mの乗用車は原寸で約23 pxです。
- GIFがある行は一覧内でも縮小したGIFを自動再生します。画像をクリックすると原寸画像（GIFがあればGIF）が開きます。
- 見出しをクリックすると、その列で並べ替えます（もう一度で逆順）。

| 列 | 意味 |
|---|---|
| パターン固有の設定 | 共通設定に上書きしている値 |
| seed | このシーンのseed |
| 経路長 [m] | ナビゲーション経路の各区間の第0レーン長の合計（実走距離の近似） |
| 正味の旋回角 [deg] | 経路全体で曲がった角度の合計 |
| カーブ 半径[m]:角度[deg] | 経路上の各カーブ（例: `R38:+90`） |
| 先行車 | seedから抽選した台数と、各車の初期距離・速度・レーン |
| 追い越した台数 | 終了時に自車中心より経路上で後ろにいる先行車の台数 |
| step数 | 終了または `max_steps` による打ち切りまでのstep数 |
| 車線変更 | 同じ道路区間内でレーン番号が変わった回数（区間境界と同時の変更は含まない） |
| 終了理由 | `arrive_dest`（到着）/ `crash_vehicle` / `crash_object` / `crash_building` / `crash_human` / `out_of_road` / `timeout`（環境またはカタログのstep上限） |

### 画像の見方

- 自車と交通車両の通った跡は **薄い側が過去、濃い側が最近** です。`trail_frames` を指定した場合は直近の描画履歴だけが残り、薄い端も走行途中の位置になります。
- オレンジの線は対向車線との境界です。
- 自車はMetaDrive付属のルールベースの運転（IDMPolicy）で走らせています。理想的な走り方ではなく、「このシーンで何が起きうるか」の目安です。詳しくは次の節を参照してください。

## 自車の運転（IDMPolicy）について

カタログでは、自車を MetaDrive 付属の `IDMPolicy`（`metadrive/policy/idm_policy.py`）で走らせます（TOMLの `ego_policy = "idm"`）。

**ルールベースの運転です。強化学習ではありません。**
学習済みの重みはなく、決まった式とルールで毎step操作を決めます。
MetaDrive では、交通車両を走らせるのにも同じ `IDMPolicy` が使われています。

| 操作 | 決め方 |
|---|---|
| アクセル・ブレーキ | IDM（Intelligent Driver Model）という車間距離の式で決める。前に車がいなければ目標速度（30 km/h）まで加速し、前の車に近づくと、車間距離と相対速度に応じて減速して追従する |
| ハンドル | 走っている車線の中心線に沿うよう、向きと横ずれを小さくするフィードバック制御（PID制御）で決める |
| 車線変更 | 前の車が近く、隣の車線のほうが空いていて安全な距離があるときに、隣の車線へ移る簡単なルール |
| 進む道 | MetaDrive のナビゲーションが決めた経路（交差点でどの出口へ進むかなど）に従う。自分で経路を探したり、軌道を計画したりはしない |

つまり、先を見越して軌道を計画する「プランナー」ではありません。目の前の状況に反応して運転する、単純な運転モデルです。

**カタログを見るときの注意**

- 表の結果（追い越した台数、終了理由など）は「単純なルールの運転なら、このシーンでこう動く」という目安です。人間や学習済みモデルの最適な走り方ではありません。
- 学習（`train.py`）で使うのは PPO（強化学習）のモデルで、IDMPolicy とは別物です。生成される `seed_XXXX.toml` にも IDMPolicy の設定は含まれません。カタログ表示のためだけに使っています。
- 前の車が自分と同じ 30 km/h で走っていると、追いつかないため車線変更も起きません。通常の交通（`traffic_density`）で追い越しが起きにくいのはこのためです。追い越しを見たいときは、遅い先行車を置く `overtake_slow_lead.toml` を使ってください。
- `ego_policy = "idle"` にすると、自車は止まったままになり、地図と交通の動きだけを確認できます。

## 選んだシーンを学習に使う

`configs/official.toml` などの実験設定をコピーし、表の「設定ファイル」列の `seed_XXXX.toml` の内容を同名table（`[environment.common]` / `[environment.train]` / `[environment.evaluation]`）へ写します。
生成TOMLは環境部分だけなので、単独では `--config` に渡せません。元の `discrete_action = true`、`discrete_throttle_dim`、`discrete_steering_dim` は残してください。
`map_config` を写すときは元の `map` を、`map` を写すときは元の `map_config` を削除し、両方が残らないようにします。
写した学習用TOMLは、通常どおり `train.py` / `evaluate.py` に渡して使います。

```bash
.venv/bin/python train.py --config configs/<新しい学習用TOML>.toml
.venv/bin/python evaluate.py --config configs/<新しい学習用TOML>.toml
```

飛び飛びのseedを選ぶ場合は、完全な実験TOMLの学習・評価tableへ `scenario_seeds` を指定します。カタログの `[settings].seeds` とは別の設定です。

```toml
[environment.train]
scenario_seeds = [2, 5, 11]

[environment.evaluation]
scenario_seeds = [2, 5, 11]
```

`scenario_seeds` は非空の整数配列で、負数・bool・重複は指定できません。入力順を保持します。指定時は `start_seed` / `num_scenarios` を省略でき、既存の値があっても一覧を優先します。MetaDriveの有効なseed範囲は、最小から最大までを含む `start_seed = 2, num_scenarios = 10`（2以上12未満）へ導出しますが、実際に選ぶ候補は `[2, 5, 11]` だけです。`scenario_seeds` 自体はMetaDriveへ渡しません。

学習はworkerごとの `random.Random(RL seed + rank)` で候補から復元抽出するため、同じseedを続けて使うことがあります。この乱数器はPython・NumPy・MetaDriveの既存乱数列から独立しています。評価は指定した一覧順に各seedを1回ずつ使います。`scenario_seeds` を指定しない場合は、従来どおり `start_seed` / `num_scenarios` の連続範囲を使います。

### 設定の違うシーンを混ぜる（scene_groups）

カーブと遅い先行車など、設定自体が違うシーンを1つのモデルで学習する場合は `[[environment.scene_groups]]` を使います。完全な設定例は [example_scene_groups.toml](../example_scene_groups.toml) です。Cの3seed、遅い先行車の3seed、respawn交通の2seed、交差点交通の2seedを含む、4グループ・10シーンの短い学習用設定です。

```toml
[environment.common]
discrete_action = true
discrete_throttle_dim = 3
discrete_steering_dim = 3
random_traffic = false
store_map = true

[[environment.scene_groups]]
name = "curve__C"
train_seeds = [0, 3, 5]
evaluation_seeds = [0, 3, 5]  # 省略時はtrain_seedsと同じ
[environment.scene_groups.common]
map = "C"
horizon = 1000
traffic_density = 0

[[environment.scene_groups]]
name = "overtake_slow_lead__lane2_one"
train_seeds = [0, 3, 5]
[environment.scene_groups.common]
map_config = { type = "block_sequence", config = "SSS", lane_num = 2 }
horizon = 1000
traffic_density = 0
lead_vehicles = { count = [1, 1], distance_m = [30, 60], gap_m = [20, 40], speed_kmh = [12, 12], lane = "ego" }
```

これは環境部分の例です。実行時は、同梱の完全設定例か、既存の `[training]` / `[evaluation]` を保持した完全な実験TOMLへ反映してください。元の設定を保護するため別名へコピーし、`name` も未使用の値に変えます。統合ページからの選択書き出しは次の節を参照してください。

カタログの `seed_XXXX.toml` から選ぶ場合は、その環境設定のうち次の8キーをグループの `common` へ、行動形式や `log_level` など固定するキーを全体の `common` へ分け、使いたいseedをグループの一覧へ写します。

- グループの `common` で切り替えられるキーは、`map`、`map_config`、`traffic_density`、`traffic_mode`、`accident_prob`、`lead_vehicles`、`horizon`、`random_spawn_lane_index` の8個です。行動形式・報酬・観測・`store_map` など固定する設定は `[environment.common]` に置きます。
- この8キーは `[environment.common]` とstageの `[environment.train]` / `[environment.evaluation]` に置けません。各グループの `common` へ移します。各グループには `map` または `map_config` のどちらか1つが必要です。
- `random_traffic` は全体で `false` 固定です。グループ内には置かず、全体の `common` に指定します。`true` は使えません。
- `scene_groups` 使用時は、全体の `common` とstageに `start_seed`、`num_scenarios`、`scenario_seeds` を置けません。stageのtable自体は省略できます。
- グループ名は出力フォルダ名として使える一意な名前、seed配列は非空・非負の整数・bool不可・グループ内の重複なしです。別グループで同じseedを使えます。`evaluation_seeds` は省略すると `train_seeds` と同じになります。

学習は各episodeの開始時に、全学習シーンの「グループ×seed」の組から均等に1つを復元抽出します。上の完全設定例では10組が各1/10の確率です。workerごとの `random.Random(RL seed + rank)` を使い、環境はworkerにつき1つのまま切り替えます。`store_map = true` の地図cacheもグループ別に分け、同じseedの異なる地図を区別します。先行車なしのグループへ切り替えたときは、前の先行車を残しません。

評価はグループの記載順、そのグループの `evaluation_seeds` の順に各組を1回ずつ実行します。episodeの結果・出力フォルダ・GIFにはグループ名が入り、全体とグループ別の集計を保存します。学習metadataのグループ×seed使用回数は、終了直後の自動resetも含む実reset回数で、完了episode数とは異なります。`inspect_env.py --config <完全TOML> --stage train` / `--stage evaluation` は各グループを検査します。`scene_groups` を省略した場合は、既存の連続seed範囲と `scenario_seeds` の使い方を維持します。

### 統合ページで選択して書き出す

`outputs/scene_catalog/index.html` の各カードのタイトル下にある「学習」「評価」のチェックで、カタログ・パターンをまたいで使いたいシーンを選びます。左の設定バー上部（狭い画面ではページ上部）にそれぞれの件数が表示され、「選択を書き出す」で `scene_selection.toml` を保存できます。「選択をすべて解除」で両方の選択を消せます。filterの解除とシーン選択の解除は独立しています。

チェック状態はブラウザのlocalStorageへ保存し、同じページを開き直したときに復元します。設定TOMLの内容も照合するため、再生成で設定が変わったシーンへ古い選択を誤って戻しません。ブラウザが保存を許可しない場合は画面に警告し、そのページを閉じると選択は消えます。

- 同じカタログ・パターン・切替設定が完全に一致するシーンを1グループへまとめます。同じパターン名でも設定が違えば別グループになり、名前が重なる場合は連番で区別します。
- あるグループで評価を1つも選ばなければ、そのグループの学習seedと同じ一覧を `evaluation_seeds` に明示して書き出します。評価seedを学習seedとは別に選ぶこともできます。
- 学習が0件で評価だけ選ばれたグループ、`random_traffic = true`、読めない・壊れた設定TOMLの選択がある場合は、画面で警告して選択全体の書き出しを中止します。選択や設定を修正してから再実行してください。

`file://` で開いたまま使えます。サーバーへ送らず、ブラウザのダウンロードとして保存するため、保存先はブラウザ側で確認してください。選択ファイルは正しいTOMLですが、完全な学習設定ではなく、単独で `train.py --config` へ渡すものではありません。冒頭にGitHub Copilot向けの変換手順、`[selection]` に形式・推奨名・日時、続いて貼り付け用の `[[environment.scene_groups]]` が入ります。

保存したファイルと、使いたい完全な学習設定のベースをCopilotへ渡します。次の例の `configs/official.toml` は、開始車線タスクやlookaheadを含む手元のベースへ置き換えられます。

```text
#file:scene_selection.toml と #file:configs/official.toml を使い、
scene_selection.toml の冒頭の指示どおりに、別名の完全な学習用configを作ってください。
元ファイルは変更せず、nameを一意にし、training・evaluation・lookaheadと
行動形式・報酬・観測などの独自設定を保持してください。
最後にinspect_env.pyをtrain/evaluationの両stageで実行し、結果を教えてください。
```

作成する完全TOMLには `[selection]` を貼りません。ベースにある切替8キーと旧seed指定を全体・stageから整理し、固定する設定は保持します。既存の `scene_groups` があれば今回の選択へ置き換え、重複して追記しないでください。古い統合ページにチェックがない場合は、`PYTHON_BIN=.venv/bin/python bash run_scene_catalogs.sh gallery` で統合HTMLだけを再生成して開き直します。シミュレーションの再実行は不要です。

## パターンを追加・変更する

TOMLは次の3つの部分からなります。

```toml
[settings]                         # 描画と走行の設定（全パターン共通）
seeds = { start = 0, count = 12 }  # 描画するseedの範囲
scaling = 5.0                      # 縮尺 [px/m]。同じTOML内の全画像で共通
ego_policy = "idm"                 # "idm"（IDMPolicyで走る）| "idle"（停止したまま）
max_steps = 800                    # 1シーンで走らせる最大step数
render_every = 5                   # 何stepごとにフレームを描くか
trail_frames = 15                  # 省略可。通った跡を何フレーム分残すか（省略時は全フレーム）
gif = false                        # GIFを出力するか

[base]                             # 全パターン共通のMetaDrive環境設定
horizon = 1000
traffic_density = 0

[[variants]]                       # パターン。必要なだけ繰り返す
name = "C_lane2"                   # パターン名（--variant と出力ディレクトリ名に使う）
env = { map_config = { type = "block_sequence", config = "C", lane_num = 2 } }  # [base] に上書きする環境設定
settings = { gif = true }          # 省略可。[settings] に上書きする値
```

`env` に書けるのは、単一設定で学習するTOMLの `[environment.common]` に書けるMetaDriveの設定と同じです。
主なものは次のとおりです。

| キー | 意味 |
|---|---|
| `map` | ブロック列。`S` 直線、`C` カーブ、`X` 十字路、`T` T字路、`O` ラウンドアバウト など |
| `map_config` | 車線数（`lane_num`）や車線幅 [m]（`lane_width`）を変えるときに使う。下の注意を参照 |
| `traffic_density` | 交通車両の密度 |
| `traffic_mode` | `"trigger"`（自車が近づくと動き出す）/ `"respawn"`（最初から走る） |
| `random_traffic` | `false` で交通配置をseedで固定する |
| `accident_prob` | 対象ブロックに故障車・コーン・バリアなどを置く確率（自車レーンを塞ぐとは限らない） |
| `lead_vehicles` | 自車前方に遅い先行車を配置する。書式は「遅い先行車」を参照 |

**注意：`map` と `map_config` は同時に書けません。**
MetaDriveは両方が既定値から変更されていると `AssertionError` で停止します。
車線数などを変えるときは `map` を書かず、ブロック列も `map_config` の中に書きます。

```toml
# NG: env = { map = "SSS", map_config = { lane_num = 2 } }
env = { map_config = { type = "block_sequence", config = "SSS", lane_num = 2 } }
```
