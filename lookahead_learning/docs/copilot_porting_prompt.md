# GitHub Copilotへ最初に渡す移植依頼

移植用コードを `incoming/lookahead_learning/` に配置しました。稼働中の `lookahead_learning/` は比較前に上書きしないでください。
単一エージェントで調査・差分適用・軽量検証を行い、既存の独自変更を保護してください。
対象は時間指定の注視点（機能1）と等速・一定曲率の予測位置報酬（機能2）です。加速度推定・加速度付き予測は対象外です。
元GitHub、元チャット、移植元root全体、Excel・画像・実験出力を読む必要はありません。

## 1. 比較してからバックアップする

branch・HEAD・git statusと未コミット差分を確認してください。まずこの文書、次に必要時だけ `docs/porting.md` の接続例を読みます。
`rg`で下記シンボルと呼出し元を探し、「未導入／既存版／独自改変・部分適用／適用済み」を根拠付きで判定してください。
フォルダの存在やschema番号だけでは判定しません。全コード通読・数式の再設計は不要です。

| 対象 | 今回確認・更新する関数 |
|---|---|
| checkpoint.py | resolve_lookahead_config、_effective_settings、set/validate_lookahead_model_metadata（schema3） |
| prediction.py | 同梱のMotionState、time_lookahead_distance、prediction_penaltyと集計（再実装しない） |
| adapter.py | read_vehicle_state(require_motion)、MetaDrivePreviewProvider.__call__、wrap_lookahead_env |
| env.py | LookaheadEnvの設定・snapshot・step・reset・prediction診断・info |
| host | config loader、通常train/evaluate→factory→worker、モデル保存/読込み、既存step/episode出力 |

変更予定ファイル、元からある未コミット差分、HEAD、設定を新しい別ディレクトリへ退避し、元から無かったファイルも一覧化してください。
適用後の差分とハッシュも保管し、撤去時に今回変更と後続変更を区別できるようにします。
host adapterは丸ごと上書きせず、独自の入力・開始車線参照・報酬を保持してください。
モデル、認証、共有設定、AGENTS、無関係なファイルは変更しません。

## 2. 不足差分だけ接続する

- 未導入なら `docs/porting.md` の4接続を通常入口へ追加します。既存版なら同じ接続を再利用し、不足したキー保持・metadata・ログだけを補います。
- 適用済みなら検証だけ行います。重複wrapper、D+6、同梱コードの再実装、報酬の二重加算を禁止します。
- 同梱resolverのmapping全体（既存5キー＋今回4キー）を通常train/evaluate/factory/workerへ保持します。host独自の許可キーや辞書再構築で落とさないでください。
- raw Env → LookaheadEnv → 既存Monitor → VecEnvの順序を保ちます。Dは実際のBoxから取得します。reset/stepは既存wrapperチェーンを通して各1回です。
- PPO保存前のset helper、ロード直後のvalidate helperを再利用します。旧schema1/2の距離指定・新項Off互換、有効設定の不一致拒否を迂回しません。旧ZIPを変更せず、sidecarは増やしません。
- 時間指定は既存の固定経路・投影に可変Lを渡します。新しい経路探索や別runner・学習ループは作りません。
- hostソースで車体中心位置[m]、heading[rad]、平面速度[m/s]、符号付き前進速度、dtを確認します。単位を推測しません。独自reader/providerの契約は必要時だけ `docs/time_prediction.md` の1・5節を読みます。
- 報酬のpost起点・マスク・未クリップ座標・Off/重み0は同文書2〜5節と同梱実装が正です。既存PP・横加速度報酬の時刻を変更しません。
- step出力がinfo全体を保持していれば接続を増やしません。episode選択リストには既存値に加えて `episode_r_prediction` と `prediction_episode` を渡します。rootに数式・集計処理を複製しません。
- 同梱examplesは通常CLIで読める独立TOMLです。移植先の学習条件を上書きせず、別名の比較設定へ必要差分だけ反映します。設定優先順位と3条件はREADMEを参照します。

契約不明なら推測せず、その接続だけ保留し、不足する根拠を報告してください。他の確認済み作業は進めます。

## 3. 最小検証と報告

```bash
python -B -m unittest discover -s lookahead_learning -t . -p 'test_*.py'
python train.py --help
python evaluate.py --help
```

通常CLIの設定読込とfactory/worker保持も確認します。`test_portability` はPORTABLE_FILESのみを一時コピーし、元root・MetaDrive・SB3のimportを禁止して設定・純粋関数・fake hostを実行します。
SB3依存のMonitor/VecEnvテストはこの隔離環境でのみskipし、hostの通常環境では実行してください。
D+3、Off/重み0と同じaction列での観測・報酬・終了・乱数同値、post参照、単一加算、旧モデル互換、reset/terminal履歴分離を確認します。
利用可能な既存assetsがあれば、単一環境で数stepのsmokeまで行います。長時間学習、依存更新、assets download、元rootの丸ごとコピーは行いません。
commit/pushは移植先利用者から明示的な指示がある場合だけ行います。

最後に判定根拠、変更ファイル、追加/変更/適用不要だった接続、バックアップ先、実行コマンド、テスト結果と未検証事項、今回差分だけの撤去手順を報告してください。
fixtureの成功を実PCの移植完了や学習改善とは記載しません。
