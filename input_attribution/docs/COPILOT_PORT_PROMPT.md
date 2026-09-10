移植先では最初に `input_attribution/docs/PORTING_QUICKSTART.md`、`InputAttributionAdapter.prepare`／`predict`／`make_env`／`snapshot`／`frame`、`extract_reward_terms`、`schema.py`、config、`test_core.py`／`test_schema.py`／`test_reporting.py`／`test_acceptance.py`だけを読んでください。T13の確認名は `test_synthetic_262_high_index_and_alternate_provider_connect_a_b_js_report` です。未解決点がある場合だけ、移植先の観測生成・通常前処理・報酬計算の該当関数を調べます。全リポジトリ、長いログ、過去設計を先に読んだり、不要なagentを増やしたりしないでください。

まず model のD・入力順序・追加入力・dtype、実観測shape、行動空間、前処理境界を照合します。`D=262` という理由だけで259の意味を先頭へ移したり、末尾3要素と推定したりしません。観測をpadding／切り捨てせず、model・実観測・明示schemaが一致するまで実験を開始しません。`template_262_schema()` は全262位置が未確認です。各位置の意味、正規化、参照レーン、有効フラグ、source commit／file hashを移植先で埋めてください。T13のsynthetic262は人工テスト専用で、実262確認済みとは報告しません。

当該stepの実返却rewardに合わせ、係数・符号・終了時の上書き順を反映した内訳だけを返します。`extract_reward_terms`を使い、合計一致をatol／rtolで検証します。報酬関数を再呼出しせず、差額を仮成分へ足さず、欠測を0補完せず、unavailable／mismatch／nonfiniteを区別してください。共通のA／B、JS式、描画、公式本体、学習、報酬の意味は変更せず、必要な読み取り計測はadapterへ閉じます。

最初に合成の最小テストを通し、127stepの全対象への-1到達、対象外とbaselineの不変、①-Aのreset／step 0回、自然終了50stepの①-B、JSの0と`ln(2)`、報酬一致と呼出し回数を確認します。続いてbaseline＋少数patternの実スモークで、GIF、reward PNG、JS PNG、HTMLを確認します。変更ファイル、実行コマンド、結果、未解決点を短く報告し、未接続内訳や実262未確認を成功扱いにしません。
