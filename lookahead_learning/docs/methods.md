# 仕組みと数式

lookahead_learning は、ホストの既存環境を composition で包み、経路に沿った
前方の点を表す3値を観測の末尾へ追加します。pp_weight が正の場合だけ、同じ点から
計算したPure Pursuit（PP）の操舵参照と実際のActionの差を追加報酬にします。
方策が選んだAction、ホストの報酬、終了条件はwrapperが包む既存環境が持ちます。

## 観測と報酬

ホストのraw観測をflat Boxの1次元 float32 ベクトル o_host ∈ R^D とします。Dは移植先
hostが決める値で、固定値の前提はありません。設定と結果の対応は次のとおりです。

| TOML | 方策への入力 | 返す報酬 |
|---|---|---|
| [lookahead] なし | rawのD値 | ホストの r_base |
| [lookahead]あり、pp_weight=0 | rawのD値＋注視点3値＝D+3 | ホストの r_base |
| [lookahead]あり、pp_weight>0 | rawのD値＋注視点3値＝D+3 | r_base＋r_pp |

追加3値は末尾の順に x_norm、y_norm、preview_valid です。raw prefixの順序・
意味・encoding・正規化はそのまま保持します。padding、切り捨て、raw報酬の
再計算はしません。注視点3値は次の意味です。

| 値 | 意味 |
|---|---|
| x_norm | 車体前後方向の注視点座標 |
| y_norm | 車体左右方向の注視点座標 |
| preview_valid | 注視点が有効なら1.0、無効なら0.0 |

x_norm と y_norm は座標[m]を10m基準で次のように変換します。
n(v) = (clip(v / 10.0, -1, 1) + 1) / 2
無効時の値は (0.5, 0.5, 0.0) です。左右差0mも0.5になるため、
preview_validを同時に確認します。

pp_weight=0.0では r_pp=0 なので、注視点追加と既存報酬の動作を分離して確認
できます。正のpp_weightではRLのActionをPPへ置換せず、連続値の u_pp を報酬の
参照だけに使います。

## 経路に沿った注視点

reset後にhost adapterがNavigationの計画と開始レーンの参照から、有限区間の
固定経路 P(S) を一度構築します。Sは道路に沿った累積距離です。接続を検証できる
StraightLaneやCircularLaneの区間だけを採用し、未対応・不連続・未検証境界の先へ
外挿しません。

車体位置 p を経路へ投影した距離を S_proj、TOMLの lookahead_m を注視距離
L_d とすると、

S_goal = S_proj + L_d
q = P(S_goal)

です。L_dはユークリッド弦長ではなく経路弧長です。たとえば
lookahead_m=6.0なら、まっすぐな区間の6m先を使います。S_goalが検証済み経路の
終端を越える場合は無効で、終点へclampして有効扱いにはしません。

車体の向きをψ、前向きと左向きの単位ベクトルを
f=(cosψ, sinψ)、l=(-sinψ, cosψ) とします。Δ=q-pから

x_g = Δ・f
y_g = Δ・l

を求め、観測へ入れるときだけ上の n(v) を適用します。位置・向き・経路投影が
有限でない、注視点が前方でない、開始レーン参照が失われた、または逆走などの
状態では preview_valid=false とします。停止や低速だけを理由に無効にはしません。

## PP操舵参照

PPは注視点qを後輪車軸基準で使います。後輪までの距離を rear_wheelbase_m、
前後車軸間の距離を wheelbase_m とし、これらはadapterがhostの車両定義から
読みます。注視距離 L_d と車軸間距離 wheelbase_m（式では L_w）は別の量です。

p_rear = p - rear_wheelbase_m f
x_rear = (q - p_rear)・f
y_rear = (q - p_rear)・l

κ = 2 y_rear / (x_rear² + y_rear²)
δ = atan(L_w κ)
u_pp = clip(steering_sign × deg(δ) / max_steering_deg, -1, 1)

車軸長、最大操舵角、操舵符号の単位と根拠はhost adapterの監査対象です。車体の
数値から単位を推測したり、移植先のActionをPP値へ置き換えたりしません。
後輪距離や分母が不正、値が有限でない場合は pp_valid=false とし、追加項を
マスクします。

## 1 decisionの処理

LookaheadEnvは次の順序で一つのdecisionを処理します。

1. resetで基底env.resetを一度だけ呼び、固定経路とpre-action snapshotを作る。
2. snapshotの注視点、preview_valid、pp_valid、u_ppを保持する。
3. 基底env.step(action)を一度だけ呼び、返ったr_base、terminated、truncated、
   infoを受け取る。基底reward_functionを呼び直さない。
4. hostが実際に適用した操舵 u_applied を読み、post-action snapshotを作る。
5. post-action側の注視点3値を次観測へ付け、次のsnapshotを保存する。

PP報酬を使うとき、pre-actionの参照に対するマスクは

m = 1[pre-actionの pp_valid] 1[not terminated] 1[not truncated]

です。hostの実際のdecision幅を Δt とすると、

e_pp = |u_applied - u_pp| / 2
r_pp = -pp_weight × Δt × m × e_pp
r_total = r_base + r_pp

となります。終了stepは追加項を0にしてhostの終端報酬を保ちます。行動前の
注視点が有効なら、行動後snapshotが無効になってもそのstepの参照を後から消し
ません。Δtはhost設定の物理step幅とdecision repeatから求め、コードに固定値を
埋め込みません。

## 移植時の境界

通常経路は次の呼出しです。

configs.experiment_config.select_experiment
→ ExperimentProfile.lookahead_config
→ train.py/evaluate.py
→ env_factory.make_training_env/make_evaluation_env
→ env_factory.make_env
→ adapter.wrap_lookahead_env
→ LookaheadEnv.reset/step

config loaderは [lookahead] がない場合 None、ある場合は有限で正の
lookahead_mと0以上のpp_weightを解決します。train/evaluateはそのmappingを
factoryへ渡し、trainは model.lookahead_config と schema versionをZIPへ保存し、
evaluateはZIPと選択TOMLを照合します。

既存の start_lane_env.py にあるStartLane系Subclassの入力生成、報酬関数、終了条件、
Action適用は基底envの責務として保持します。adapterはNavigation、vehicle、Actionの
意味と単位をその境界で読み、wrapperは返却値を利用して追加情報だけを計算します。
MetaDrivePreviewProviderはactiveなlookahead全てで使い、MetaDrivePPProviderは
pp_weight>0の場合だけ使います。評価のreset/stepはwrapperを通し、MetaDrive固有の
custom propertyやrenderはenv.unwrappedから読みます。
