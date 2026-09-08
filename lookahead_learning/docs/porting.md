# 移植手順

この追加モジュールは、既存のMetaDriveホストのルートへ
`lookahead_learning/`を置いて使います。ホストの環境・観測・assetsをこの
モジュールから作り直すことはありません。前方注視の仕様は
[`methods.md`](methods.md)、文書一覧は[`README.md`](README.md)にあります。

## コピー後の配置

コピー元のcheckoutにある`lookahead_learning/`の直下の`.py` 17個を、
移植先ホストのルート直下へコピーします。移植先には、ホスト側の3個の
Pythonファイルと`configs`のPythonファイル、使用するTOMLも必要です。

コピー元からコピーする対象は、次の1行の関係です。

```text
<source checkout>/lookahead_learning/*.py  →  <destination host root>/lookahead_learning/
```

コピー後の移植先は次のtreeになります。

```text
<destination host root>/
├── env_factory.py                         (既存ホスト)
├── start_lane_env.py                      (既存ホスト)
├── project_paths.py                       (既存ホスト)
├── configs/
│   ├── __init__.py                         (既存ホスト)
│   ├── experiment_config.py                (既存ホスト)
│   └── official_start_lane_return.toml    (選択する設定)
└── lookahead_learning/
    ├── __init__.py
    ├── __main__.py
    ├── adapter.py
    ├── checkpoint.py
    ├── diagnostics.py
    ├── env.py
    ├── geometry.py
    ├── portability.py
    ├── runner.py
    ├── telemetry.py
    ├── test_checkpoint.py
    ├── test_env.py
    ├── test_geometry.py
    ├── test_portability.py
    ├── test_runner.py
    ├── test_telemetry.py
    ├── tests.py
    └── docs/                                (読解用、任意)
        ├── README.md
        ├── methods.md
        ├── run.md
        ├── porting.md
        └── copilot_porting_prompt.md
```

実行に必要なファイルの最小構成は、上記17個の追加モジュール、ホスト側の
`env_factory.py`・`start_lane_env.py`・`project_paths.py`、
`configs/__init__.py`・`configs/experiment_config.py`、そして選択したTOML
です。これはファイルの最小構成であり、実行には監査済みPython依存ライブラリと
MetaDrive assetsが別途すでに必要です。読み物としては、
`lookahead_learning/docs/`内の`README.md`・`methods.md`・`run.md`・
`porting.md`・`copilot_porting_prompt.md`の5つのMarkdownを推奨しますが、これらは実行の最小構成には含めません。
GitHub Copilotへ移植作業を依頼する場合は、[移植依頼プロンプト](copilot_porting_prompt.md)を使えます。

`__pycache__/`、その他のbytecode、`outputs/`、`models/`、`logs/`、
MetaDriveの`assets/`はこのコピーに含めません。チェックポイントを持ち込む
場合だけは、下の「checkpointを移す場合」の規則に従ってください。

## ファイルをコピーする

Linux/macOSのシェルでは、コピー元と移植先を明示して、package直下の`.py`
だけをコピーします。パスに空白があっても引用符で囲めば動きます。

```bash
SOURCE_ROOT="/path/to/metadrive_rl-lookahead"
HOST_ROOT="/path/to/metadrive-host"
mkdir -p "$HOST_ROOT/lookahead_learning"
cp "$SOURCE_ROOT"/lookahead_learning/*.py "$HOST_ROOT/lookahead_learning/"
# 読み物も持ち込む場合（任意）
cp -R "$SOURCE_ROOT"/lookahead_learning/docs "$HOST_ROOT/lookahead_learning/"
```

選択したTOMLが移植先の`configs/`にまだない場合は、同じ内容の
`configs/official_start_lane_return.toml`を用意します。既存の別設定を確認
なしに上書きせず、使用する設定の`map`・scenario範囲・PPO条件が監査対象と
一致していることを確認してください。

PowerShellでは次の形で同じ17個をコピーできます。

```powershell
$SourceRoot = 'C:\path\to\metadrive_rl-lookahead'
$HostRoot = 'C:\path\to\metadrive-host'
$AddonDestination = Join-Path $HostRoot 'lookahead_learning'
New-Item -ItemType Directory -Force $AddonDestination | Out-Null
Get-ChildItem (Join-Path $SourceRoot 'lookahead_learning') -File -Filter '*.py' |
    Copy-Item -Destination $AddonDestination
```

`-Recurse`は付けていないため、`lookahead_learning`の直下にある`.py`
だけが対象です。`PYTHONPATH`を恒久設定したり、editable installを作ったり
する必要はありません。

## 移植先の前提

移植先は、単に同じshapeを返すホストでは足りません。運用モードには、
hostが提供する`(262,)`・`float32`の観測、そのうちindex 259--261の意味・
順序・encoding・正規化をソースで確認した証拠、そして追加adapterのregistry
登録が必要です。`lookahead_obs`と`lookahead_obs_pp_reward`は、その262値に
前方注視の3値を追加するため、policy observationは`(265,)`になります。
paddingや切り捨てでこの条件を満たすことはできません。

具体的には、[`../adapter.py`](../adapter.py)の`PrefixFeatureEvidence`で
index 259・260・261それぞれの`name`・`meaning`・`encoding`・`source`を
ソース根拠付き（source-backed）に確認し、hostクラスの対応を[`../runner.py`](../runner.py)の
`_VERIFIED_HOST_PREFIX_EVIDENCE`へ登録した状態が必要です。shapeだけを見て
registryへ登録することはできません。

このcheckoutの実hostはraw `(259,)`・`float32`で、追加3値は未登録です。
そのため、このcheckoutでの移植チェックはファイル配置・import・help・
static doctorの確認までで、運用学習や評価の合格を意味しません。別PCで
262値が得られても、ソース根拠（source-backed evidence）とregistryがなければ同じ拒否に
なります。

監査で記録された依存関係とassetsは次の組み合わせです。移植先に既に存在
していることを確認してください。この追加モジュールはpip installやassets
downloadを行いません。

| 項目 | 監査値 |
| --- | --- |
| Python | 3.12.3 |
| MetaDrive | 0.4.3、source commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| Panda3D | 1.10.16 |
| PyTorch | 2.13.0 |
| NumPy | 2.5.2 |
| MetaDrive assets | version 0.4.3、監査時はversion一致・更新なし |

ソース側では [`../adapter.py`](../adapter.py) がhost importと観測・行動契約を
確認し、[`../runner.py`](../runner.py) がsource identityと学習・評価を管理
します。移設検証の実装は [`../portability.py`](../portability.py) です。

## 移植後の確認順

移植先ホストのルートで、選択したPythonを使います。現在のcheckoutで監査に
使ったPythonを参照する場合は`../metadrive_rl-main/.venv/bin/python`です。
別PCでは、依存関係が入っている環境の`python`（またはその環境の
`python3`）に置き換えます。

```bash
cd "/path/to/metadrive-host"
LOOKAHEAD_PY="/path/to/audited-python"
"$LOOKAHEAD_PY" -B -m lookahead_learning --help
```

PowerShellでは、空白を含むcwdとPythonのパスを引用し、`&`で選択した実行
ファイルを呼び出します。

```powershell
Set-Location 'C:\path with spaces\to\metadrive-host'
$LOOKAHEAD_PY = 'C:\path with spaces\to\Python\python.exe'
& $LOOKAHEAD_PY -B -m lookahead_learning --help
```

続けてstatic doctorと単体・模擬環境テスト（実MetaDrive起動・学習なし）を実行します。どちらもこの段階ではengineを
起動しません。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor \
  --config configs/official_start_lane_return.toml \
  --output "outputs/lookahead_learning_docs/porting-doctor-static"

"$LOOKAHEAD_PY" -B -m lookahead_learning test \
  --output "outputs/lookahead_learning_docs/porting-test-unit"
```

`test --portability`は、空白を含む新しい一時的な配置へ必要な`.py`と設定を
コピーし、fresh subprocessのhelpとstatic doctorを確認します。`--portability`
だけの実行はstaticで、MetaDrive engineを起動しません。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning test --portability \
  --config configs/official_start_lane_return.toml \
  --output "outputs/lookahead_learning_docs/porting-portability-static"
```

`--integration`はraw `doctor --probe`だけを実行するruntime診断です。static
portabilityと同時に指定して両方が検証されると考えず、必要なら別コマンドで
実行してください。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning test --integration \
  --config configs/official_start_lane_return.toml \
  --seed 5 --steps 3 \
  --output "outputs/lookahead_learning_docs/porting-integration-raw-s5"
```

2 workerのspawnを含む移設先runtimeを明示的に調べる場合は、次の1コマンドを
使えます。これはホストの未包装環境（raw環境）をreset/stepする診断であり、262/265の
前方注視比較の合格判定
ではありません。`--seeds`はspawn workerが2個なので2値を指定します。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning.portability \
  --spawn-probe --project-root . \
  --config configs/official_start_lane_return.toml \
  --output "outputs/lookahead_learning_docs/porting-spawn-raw-s5" \
  --seeds 5 5 --steps 1
```

このruntime診断や`doctor --probe --pulse`を使うときだけ、MetaDrive/Panda3D
engineが起動します。出力先は毎回新しくし、既存runに追記しないでください。

## checkpointを移す場合

checkpointは必須の移植物ではありません。持ち込む場合は、学習時にrunnerが
作った対応する学習runディレクトリをまとめてコピーします。最低限、同じディレ
クトリに次の2つを残してください。

```text
<training-run>/
├── <model-name>.zip
└── metadata.json
```

同じrunのmonitorやtelemetryを後から読む場合は、それらもrunディレクトリの
他のファイルとして一緒にコピーします。zipだけを別のmetadata.jsonと組み
合わせたり、同じshapeに見える別runのsidecarを付けたりしないでください。

`metadata.json`にはmodelのSHA-256、mode、観測・行動契約、設定、学習seed、
source identityが保存されます。evaluateはこのsidecarと、移植先で計算した
source identityをPPOのロード前に照合します。したがって、別host・別mode・
別PP weight・別ソースのrunは、shapeが一致しても拒否されます。

package名やソースが現在の`lookahead_learning`と異なる場合、
sidecarの相対source keyやSHA-256も異なります。旧metadataを
書き換えて互換に見せる手順はありませんし、任意の古いsidecarが新sourceで
使えるとも仮定しません。古いraw checkpointを履歴診断したい場合だけ、
`doctor --checkpoint <existing.zip> --legacy-evaluate`を使います。この経路
はraw shapeとActionが一致する旧checkpointの`legacy_raw259_diagnostic`で、
正規packageによる運用学習・評価の代用ではありません。新しい正規packageで運用
するcheckpointが必要なら、条件を満たすhost上で新しいrunを作ります。
