# GitHub Copilotへ渡す移植依頼の詳細

配布フォルダ直下の [START_HERE.md](../START_HERE.md) から読む実施手順です。追加の依頼文を求めず、この手順に従ってバックアップ・不足差分の適用・軽量検証・結果報告まで進めてください。

移植用コードを `lookahead_learning_update/` に配置しました。稼働中の `lookahead_learning/` とは別フォルダです。両者の同じ相対パスを比較し、必要差分だけ適用してください。フォルダ全体のコピー・リネームによる置換は禁止です。
単一エージェントで調査・差分適用・軽量検証を行い、既存の独自変更を保護してください。
対象は機能3（1step差分の等加速度・一定曲率予測）の追加です。同梱の機能1（時間指定）・機能2（等速予測）と独自変更を保護します。
更新フォルダをruntime import/PYTHONPATHへ追加せず、適用後は既存の通常入口とlookahead_learning/を使います。
元GitHub、元チャット、移植元root全体、Excel・画像・実験出力を読む必要はありません。

## 1. 比較してからバックアップする

branch・HEAD・git statusと未コミット差分を確認してください。まずこの文書、次に必要時だけ更新フォルダ内の `docs/porting.md` の接続例を読みます。
`rg`で下記シンボルと呼出し元を探し、「未導入／機能1・2導入済み／旧版／独自改変・部分適用／適用済み」を根拠付きで判定してください。
フォルダの存在やschema番号だけでは判定しません。全コード通読・数式の再設計は不要です。

### 今回の差分（最初に読む範囲）

| 実変更ファイル・関数 | 確定する接続契約 |
|---|---|
| checkpoint.py: resolver、_effective_settings、metadata helpers | 新キーprediction_motion_modelを1個追加。既定constant_speed、追加constant_acceleration。schema4。旧1/2は距離・予測Off、旧3はconstant_speed。旧schemaへの新キー混入を拒否 |
| prediction.py: estimate_acceleration、constant_acceleration_travel、PredictionReference、prediction_penalty、PredictionEpisodeMetrics | 同梱の純粋関数を利用。Dを1回計算し、自車と報酬参照へ共用。既存位置報酬を選択方式で1回だけ計上 |
| adapter.py: MetaDrivePreviewProvider.reference_at_distance、__call__/_read_preview、wrap_lookahead_env | 通常previewの固定経路・post投影を保存し、同じpostから指定距離の参照点を読取専用で取得。独自adapterでは同じホスト参照経路へ接続 |
| geometry.py: compute_preview | 同じpath/pointの保存済みprojectionを任意で再利用。通常呼出しは従来どおり |
| env.py: LookaheadEnv.__init__ / _prediction_diagnostic | 実効Onの機能3だけ新取得口を要求。reset/terminalは0。既存step/reset/Monitor接続を利用 |
| __init__.py、既存test_prediction/test_prediction_env/test_checkpoint/test_portability | 仕様・schema更新、停止・独立valid・互換性・ZIP展開後の検証 |
| examples/time_prediction_acceleration*.toml、PORTABLE_FILES.txt、START_HERE.md、docs | 4比較条件＋横加速度併用例、配布・説明更新 |
| host（必要な場合だけ） | 既存config→factory→workerのmapping保持、metadata、step/episode出力。移植元のroot runtimeに今回追加変更なし |

数式全体を先に通読しません。計算の確認が必要なときだけ `docs/time_prediction.md` の「3A. 機能3」、
単位は1節、マスクは5節、ログは6節を読みます。最小のAPI契約は `docs/porting.md` の「機能3の読取専用参照取得口」です。

変更予定ファイル、元からある未コミット差分、HEAD、設定を新しい別ディレクトリへ退避し、元から無かったファイルも一覧化してください。
適用後の差分とハッシュも保管し、撤去時に今回変更と後続変更を区別できるようにします。
host adapterは丸ごと上書きせず、独自の入力・開始車線参照・報酬を保持してください。
モデル、認証、共有設定、AGENTS、無関係なファイルは変更しません。

## 2. 既存報酬を残し、予測項の方式を追加する

使用中の横加速度報酬がどこで加算されるか（LookaheadEnv、別wrapper、hostのreward_function）を確認してください。
その式、実効On/Off、上限値、重み、host固有実装を保持します。更新用サンプルのOff設定で上書きしません。
機能1・2導入済みなら、既存r_predictionの方式選択だけを追加します。等速項と等加速度項を二重に加算しません。
host側で既に横加速度項を計上している場合はそれをr_base内に残し、同梱の横加速度項をさらに有効にして二重加算しないでください。
既存wrapper側の横加速度項を使っている場合は、その設定とLateralReference/radius_reader接続を引き継ぎます。
独自報酬を同梱lateral_acceleration.pyの式へ置換しません。

`examples/time_prediction_lateral.toml` と `examples/time_prediction_acceleration_lateral.toml` は同梱の横加速度項との併用例です。
例の上限・重みを移植先の値に強制変更しません。4比較条件の横加速度Offは比較実験用の設定です。
既存横加速度の詳細が必要なときだけ `docs/lateral_acceleration_reward.md` を参照してください。

## 3. 不足差分だけ接続する

- 未導入なら `docs/porting.md` の4接続を通常入口へ追加します。既存版なら同じ接続を再利用し、不足したキー保持・metadata・ログだけを補います。
- 適用済みなら検証だけ行います。重複wrapper、D+6、同梱コードの再実装、報酬の二重加算を禁止します。
- 同梱resolverのmapping全体（既存9キー＋今回prediction_motion_modelの1キー）を通常train/evaluate/factory/workerへ保持します。host独自の許可キーや辞書再構築で落とさないでください。
- raw Env → LookaheadEnv → 既存Monitor → VecEnvの順序を保ちます。Dは実際のBoxから取得します。reset/stepは既存wrapperチェーンを通して各1回です。
- PPO保存前のset helper、ロード直後のvalidate helperを再利用します。旧schema1/2の距離指定・予測Offと旧schema3のconstant_speed互換、有効設定の不一致拒否を迂回しません。旧ZIPを変更せず、sidecarは増やしません。
- 時間指定は既存の固定経路・投影に可変Lを渡します。新しい経路探索や別runner・学習ループは作りません。
- hostソースで車体中心位置[m]、heading[rad]、平面速度[m/s]、符号付き前進速度、dtを確認します。単位を推測しません。独自reader/providerの契約はporting.mdを参照し、機能3実効On時だけ参照取得口を補います。Off／weight0／機能2では要求しません。
- 報酬のpost起点・マスク・未クリップ座標・Off/重み0はtime_prediction.mdの3A・5節と同梱実装が正です。既存PP・横加速度報酬の時刻を変更しません。
- step出力がinfo全体を保持していれば接続を増やしません。episode選択リストには既存値に加えて `episode_r_prediction` と `prediction_episode` を渡します。rootに数式・集計処理を複製しません。
- 同梱examplesは通常CLIで読める独立TOMLです。移植先の学習条件を上書きせず、別名の比較設定へ必要差分だけ反映します。設定優先順位・4比較条件・横加速度との併用例はREADMEを参照します。

契約不明なら推測せず、その接続だけ保留し、不足する根拠を報告してください。他の確認済み作業は進めます。

## 4. 最小検証と報告

```bash
python -B -m unittest discover -s lookahead_learning -t . -p 'test_*.py'
python train.py --help
python evaluate.py --help
```

通常CLIの設定読込とfactory/worker保持も確認します。`test_portability` はPORTABLE_FILESからZIPを生成・展開し、その内容だけを模擬hostへ配置して、元root・MetaDrive・SB3のimportを禁止して設定・純粋関数・fake hostを実行します。
SB3依存のMonitor/VecEnvテストはこの隔離環境でのみskipし、hostの通常環境では実行してください。
D+3、方式省略と機能3Off/重み0の同じaction列での観測・報酬・終了・乱数同値、a=0の両方式一致、post参照、単一加算、旧モデル互換、reset/terminal履歴分離を確認します。
停止後の距離一定、入力有効／報酬無効とその逆、共通経路不正、PP・横加速度参照区間の不変も確認します。
同じT・同じaction列で予測Off/Onを比較し、既存横加速度項が変わらず、報酬差がr_predictionだけであることも確認してください。
利用可能な既存assetsがあれば、単一環境で数stepのsmokeまで行います。長時間学習、依存更新、assets download、元rootの丸ごとコピーは行いません。
commit/pushは移植先利用者から明示的な指示がある場合だけ行います。

最後に判定根拠、変更ファイル、追加/変更/適用不要だった接続、バックアップ先、実行コマンド、テスト結果と未検証事項、今回差分だけの撤去手順を報告してください。
fixtureの成功を実PCの移植完了や学習改善とは記載しません。

## 5. 今回の機能3追加差分だけを戻す場合

適用前原本・適用後diff/ハッシュ・現在を三者比較し、今回の追加行だけを戻します。
元からある機能1・2、独自報酬・adapter、実運用設定・既存モデルと後続変更を保持します。
新規ファイルも後続利用がないものだけを対象にし、フォルダ削除・無条件復元・reset --hardは行いません。
競合箇所だけを保留し、旧constant_speed設定と旧モデルの互換・軽量テストを確認します。
