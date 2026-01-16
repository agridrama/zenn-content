# Zenn CLI

* [📘 How to use](https://zenn.dev/zenn/articles/zenn-cli-guide)

## 主なマークダウン
https://zenn.dev/zenn/articles/markdown-guide

- 画像+キャプション
```markdown
![代替テキスト](画像のパスまたはURL)
*caption*
```
- メッセージと警告
```markdown
:::message
メッセージの内容
:::
```markdown
:::message alert
警告の内容
:::
```


## 主要なコマンド
- 記事の新規作成
```
npx zenn new:article
```
- プレビューする
```
npx zenn preview
```
- 記事の公開
  - slugの`published`を`true`に変更してからcommit & pushしてください