"""既存ファイルを編集しない、前方注視学習用の追加モジュール。

起動はホストリポジトリ直下で ``python -B -m lookahead_learning --help``。
配布するのはこのフォルダの .py 一式だけでよい。YAML、JSON、shell
スクリプト、pip install -e、既存 train.py への import 追加は不要。
同じ意味の改修済み262次元ホスト、対応依存ライブラリ、MetaDrive assetsは
移植先に必要であり、この追加モジュールがそれらを生成・更新することはない。

この作業で調べたホストは259次元であり、依頼で想定された開始レーン3次元は
観測に存在しなかった。そのため、262次元の意味まで確認できる実装が登録
されるまで baseline/lookahead_obs/lookahead_obs_pp_reward の実験起動を拒否する。padding、切り捨て、
元ソースへのパッチ、monkey patch でこの不一致を隠すことはない。
``doctor --probe`` の259次元診断を262次元baselineの検証とは扱わない。

要求する実験の意味:
- baseline: 既存262次元と既存step返却報酬。
- lookahead_obs: 既存262次元をそのまま残し、前方注視点の幾何3次元を追加。
- lookahead_obs_pp_reward: lookahead_obsと同じ観測・行動・物理遷移に、Pure Pursuit参照との操舵不一致ペナルティだけ追加。
PPは報酬参照であり、RLが選んだ操舵を上書きしない。

参照の処理:
``adapter.build_fixed_navigation_route`` が開始レーンから予定経路に沿う
Laneの接続を検証し、``geometry.project_to_path`` が有限区間へpを射影
してS_projを求める。``geometry.compute_preview`` はS_goal=S_proj+6m、
q=P(S_goal)、forward=(cos(psi),sin(psi))、left=(-sin(psi),cos(psi))から
x_g/y_gを計算する。clip(z/10,-1,1)を[0,1]へ写した2値とvalidを追加し、
無効時は[0.5,0.5,0]にする。終端や未検証境界を越えるqは外挿しない。

``env.LookaheadEnv.reset`` は基底resetを一度だけ呼び、経路と履歴を初期化。
``_make_snapshot`` に保存した行動前s_tのu_ppと、基底step(a_t)後に読み取る
u_appliedを比較する。dtは物理step幅と実際のdecision_repeatに対応し、
r_pp=-pp_weight*dt*mask*abs(u_applied-u_pp)/2、r_total=r_base+r_pp。
maskは行動前pp_validかつ非terminated・非truncated。r_baseは基底stepが
返した値で、reward_functionを再実行しない。次観測は行動後snapshotに対応。

PPは同じqを後輪車軸中心から参照し、kappa=2*y_rear/(x_rear**2+y_rear**2)、
delta=atan(L*kappa)で求める。車軸位置・操舵角の単位・符号はadapterの監査
対象であり、車体長や数値の大小から推測しない。pp_weightはlookahead_obs_pp_rewardで明示指定が
必要。0は同一遷移の同値性試験、0.1は暫定smoke値で、学習効果を確認した推奨値
ではない。離散操舵でも連続PP教師値を離散化しないため、小さい旋回指令が
ゼロ操舵を優遇し得る。

``telemetry`` は時間重み付き横ずれ、操舵変化、速度・停止・進行量、欠測、
未完走を共通指標として集計する。走行距離と経路進行量を区別する。比較時は
道路区間と速度帯、seed、観測・車両・制御・終了・学習条件を照合し、根拠が
欠ける場合は改善と判定しない。速度等の5%条件警告と符号反転deadbandは
暫定の管理値であり、文献から確立された成功閾値ではない。


確認と診断のコマンド例（ホストリポジトリ直下、各outputは未使用の名前）::

    python -B -m lookahead_learning --help
    python -B -m lookahead_learning doctor --base-config configs/official_start_lane_return.toml
    python -B -m lookahead_learning doctor --base-config configs/official_start_lane_return.toml --probe --pulse --output outputs/lookahead_learning/doctor_probe_new
    python -B -m lookahead_learning test --output outputs/lookahead_learning/tests_new
    python -B -m lookahead_learning test --portability --base-config configs/official_start_lane_return.toml --output outputs/lookahead_learning/portability_new

実262ホストの意味・符号・順序を確認し追加adapterへ登録した後に使うコマンド例::

    python -B -m lookahead_learning train --base-config configs/official_start_lane_return.toml --mode baseline --seed 0 --timesteps 300000 --model-name baseline --output outputs/lookahead_learning/baseline_s0
    python -B -m lookahead_learning train --base-config configs/official_start_lane_return.toml --mode lookahead_obs --seed 0 --timesteps 300000 --model-name lookahead_obs --output outputs/lookahead_learning/lookahead_obs_s0
    python -B -m lookahead_learning train --base-config configs/official_start_lane_return.toml --mode lookahead_obs_pp_reward --pp-weight 0.1 --seed 0 --timesteps 300000 --model-name lookahead_obs_pp_reward --output outputs/lookahead_learning/lookahead_obs_pp_reward_s0
    python -B -m lookahead_learning evaluate --base-config configs/official_start_lane_return.toml --mode lookahead_obs --checkpoint outputs/lookahead_learning/lookahead_obs_s0/lookahead_obs.zip --scenario-seeds 5 --output outputs/lookahead_learning/lookahead_obs_eval_s0
    python -B -m lookahead_learning compare outputs/lookahead_learning/baseline_eval_s0/summary.json outputs/lookahead_learning/lookahead_obs_eval_s0/summary.json --output outputs/lookahead_learning/compare_baseline_lookahead_obs

上記の0.1は実行形式を示す暫定値。学習効果は未測定であり、学習を実行した
という意味でもない。seed 1、2等を追加するときも全モードの学習seedと予算を
そろえる。公式設定のscenarioは5のみで、学習seedをscenarioへ流用しない。
既存checkpointは262/265へ強制ロードせず、必要に応じてdoctorの
--checkpoint <既存zip> --legacy-evaluate --max-steps 500で別診断とする。

移植はホスト直下にlookahead_learningディレクトリを置き、このディレクトリの.pyを
すべてコピーする。生成済みoutputs、assets、モデルは配布物に含めない。
別checkoutのホストを使う場合は--project-rootで指定でき、ホストモジュールの
import元が一致しなければ拒否する。PYTHONPATHの永続変更は必要ない。

詳細は [README](docs/README.md)、[実行ガイド](docs/run.md)、[移植ガイド](docs/porting.md)、[手法](docs/methods.md) を参照すること。
作業・検証状況は実行出力と最終報告を参照すること。単体テスト、現行259
次元MetaDriveの診断、実262/265次元PPO学習、長時間比較は別の検証段階。
"""

__version__ = "0.1.0"
