---
title: "KV cacheによる計算量削減の見積もり"
emoji: "😽"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [Transformer, LLM]
published: true
---

授業中の課題で, KV cacheの実装をしたのですが,その際にどの程度高速化できるかを理論的に見積もる余裕がなかったので, ここで改めてちゃんと計算してみます。
今更&メモ程度なので, そもそものTransformerの構造などは省略します。

:::details 授業内容に関して
授業自体はLLMや機械学習ではなくCUDA/GPUプログラミングの授業なのですが, 最終課題の題材としてLLMの高速化を扱いました。

その際に使用した元実装はこちらです。
[llm.c](https://github.com/karpathy/llm.c)
レポジトリ自体が結構教育的なので, 興味がある方は見てみてください。

授業の課題ではこのレポジトリのcuda実装をprofileして, bottleneckを見つけて高速化しました。
同じ作者のrepositoryに[nanoGPT](https://github.com/karpathy/nanoGPT)もあり, こちらはPyTorchベースでより実用的な実装になっています。

専門とは少し離れた分野ですが, 勉強になったのでここでまとめておきます。
:::

## 前提条件とパラメータ
**推論**における計算量を考えます。 KV cache自体がautoregressiveなモデルに対して有効なので, すでに$T$個のトークンが生成されている状態から, さらに$1$トークンを生成するような場合を考えます。

パラメータを以下のように定義します。
- バッチサイズ: $B$
- すでに生成されたトークン数: $T$
- transformerの層数: $L$
- $1$層あたりのヘッド数: $H$
- $1$ヘッドあたりの次元数: $HS = d_k$
- チャンネル数: $C = H \cdot HS$
- MLPの中間層の次元数: $M$
- 入力の埋め込み次元数: $E$ (通常は $E = C$)

また以下の変数も使用します。
- $Q$: クエリ行列
- $q_t$: $t$番目のトークンに対応するクエリ行列の行ベクトル
- $K$: キー行列
- $k_i$: $i$番目のトークンに対応するキー行列の行ベクトル
- $V$: バリュー行列
- $v_j$: $j$番目のトークンに対応するバリュー行列の行ベクトル
- $d_k$: $1$ヘッドあたりの次元数 ($HS$と同義)
特に$Q,K,V$は $(B, H, T, HS)$の形状を持ちます。

:::message
$Q,K,V$を $(B, T, H \cdot HS)$で扱う場合もありますが, ここではヘッド数$H$を明示的に分けて考えます。
:::

Transformerの計算量は主にAttentionとMLPに分けられます。 例えばGPT-2系のモデル構造は以下のようになっています。
![GPT-2 model architecture](/images/gpt2-arch.png)
このようなGPT-2の場合は $L = 12$ (small), $E = C = 768$, $H = 12$, $HS = 64$ となります。

## ナイーブな場合の計算量の見積もり
ナイーブ = 毎 step、参照対象となる過去長 $T$ に対して full forward を実行し、$Q/K/V$ も全位置で再計算する場合の計算量を見積もります。 ただし主要項として $Q/K/V$ 投影・Attention・MLP のみを見積もります（LayerNorm や活性化関数などは無視します）。

### $Q,K,V$の計算
$Q,K,V$の計算はそれぞれ入力に対して線形変換を行うため, それぞれ以下の計算量が必要です。

入力または,前の層の出力が $(B, T, E)$ の形状を持ち, 射影投影で $(B, T, C)$ に写し reshape することで $Q,K,V$ がそれぞれ $(B, H, T, HS)$ の形状を持つ。 $1$つの線形変換あたり, $2 \cdot B \cdot T \cdot E \cdot C$ の乗算と加算が必要です。 これが$Q,K,V$の3つあるため, 合計で $6 \cdot B \cdot T \cdot E \cdot C = 6 \cdot B \cdot H \cdot T \cdot E \cdot HS$ となります。

### Attention
Attentionは

$$
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$

で計算されます。 ここで、Batch, Headを固定して考えると, $QK^T$の計算量は$Q,K$が $(T, HS)$の行列として扱えるため, $2 \cdot T^2 \cdot HS$ の乗算と加算が必要です。 Softmaxとスケーリングは無視できると考えます。 さらに$V$との行列積で $2 \cdot T^2 \cdot HS$ の乗算と加算が必要です。 よって, Attention全体で $4 \cdot T^2 \cdot HS$ の乗算と加算が必要となります。
これが, $B$バッチ, $H$ヘッド, $L$層あるため, Attention全体での計算量は $4 \cdot B \cdot H \cdot L \cdot T^2 \cdot HS$ となります。

### MLP
MLPは$2$層の全結合層で構成されており, 中間層の次元数が$M$であるので, 活性化関数を無視すると, $1$層あたり $2 \cdot B \cdot T \cdot E \cdot M$ の乗算と加算が必要です。 これが$2$層あるため, 合計で $4 \cdot B \cdot T \cdot E \cdot M$ となります。 各Transformer層にあるため, MLP全体で $4 \cdot B \cdot L \cdot T \cdot E \cdot M$ となります。

## KV cacheを使用した場合の計算量の見積もり
KV cache を使用した場合, $T+1$トークン目を生成する際には, すでに$T$個のトークンが生成されており $T-1$ 個目までの $K,V$ は cache に保存されています。 
この場合, ナイーブな実装では,毎回計算していた$Q,K,V$のうち, $q_T,k_T,v_T$のみを計算すれば良く, Attentionでは$q_T$のみを使用して計算を行うことができます。

:::message
自分用のメモです。 新しい$k_T$に対する$q_i$ ($i < T$)が必要になるのでは？と思うかもしれませんが, autoregressiveなモデルでは, $QK^T$の上三角行列部分はmaskされるため, $q_i$ ($i < T$)は使用されません。 なので, $q_T$のみを使用してAttentionを計算すれば良いです。 
:::

### $Q,K,V$の計算
$Q = q_T, K, V$は$1$トークン分だけ計算すれば良いため, それぞれ $2 \cdot B \cdot H \cdot E \cdot HS$ の計算量が必要です。 よって, 合計で $6 \cdot B \cdot H \cdot E \cdot HS$ となります。
### Attention
Attentionの計算は$QK^T$の計算量が変わります。 $Q$が $(T, HS)$の行列ではなく, $(1, HS)$の行列として扱えるため, $2 \cdot T \cdot HS$ の乗算と加算が必要です。 さらに$V$との行列積で $2 \cdot T \cdot HS$ の乗算と加算が必要です。 よって, Attention全体で $4 \cdot T \cdot HS$ の乗算と加算が必要となります。
これが, $B$バッチ, $H$ヘッド, $L$層あるため, Attention全体での計算量は $4 \cdot B \cdot H \cdot L \cdot T \cdot HS$ となります。
$T$だけ線形に計算量が増えるため, $T$が大きくなるほど効果が大きくなります。

:::message
- ナイーブな場合でも$q_T$のみを使用してAttentionを計算すれば良いのでは？
- ナイーブな場合で長さ$T$も再計算するのはおかしいのでは？（不利すぎる）
と思うかもしれませんが, KV cacheなしでAttentionを計算するには以前の$K,V$をすべて使用する必要があります。

しかしこれらを計算するには, 一つ前の層の出力をすべて計算する必要があります。ここで

例えば, KV cache なしで $2$層目の $K,V$ を得るには、$1$層目の出力 $X_{1..T}$（全トークン分）が必要である。ところが 
$X_i$はトークン$i$ の self-attention / MLP 出力を含むため、キャッシュを持たない限り $1$層目でも結局 トークン$i=1..T$ に対する計算が必要になる。したがって「最後のクエリだけで attention を計算する」最適化は、KV cache 等で中間結果を保持しない限り上位層へ伝播できず、結果として full recompute に近い計算量となってしまいます。
:::

### MLP
MLPの計算量は$1$トークン分だけ計算すれば良いため, $1$層あたり $2 \cdot B \cdot E \cdot M$ の乗算と加算が必要です。 これが$2$層あるため, 合計で $4 \cdot B \cdot E \cdot M$ となります。 よってMLP全体で $4 \cdot B \cdot L \cdot E \cdot M$ となります。

:::message
MLPはトークンごとの独立した全結合層であるため, KV cacheの有無にかかわらず, T番目のトークン分だけ計算すれば良いです。
:::

## まとめ

|　| ナイーブ | KV cache |
|---|---|---|
| $Q,K,V$ | $6BHTE \cdot HS$ | $6BHE \cdot HS$ |
| Attention | $4B  H  L  T^2 \cdot HS$ | $4  B  H  L  T \cdot HS$ |
| MLP | $4  B  L  T  E  M$ | $4  B  L  E  M$ |

モデルパラメータ($B,H,L$ など)を固定して計算量を$Q,K,V$, Attention, MLPの合計としてまとめると, ナイーブな場合で$\Theta(T^2)$, KV cache で $\Theta(T)$ となります。つまり, $1$トークン生成あたりの計算量は$T$が十分大きい時におおよそ$T$倍高速化されます。
実際には, GPUでの行列演算のオーバーヘッド（キリの良い数字に次元が揃わない場合）や, メモリアクセスの影響もあるため, 理論上の計算量削減とは異なる場合がありますが, KV cacheを使用することで, 長いシーケンスに対しても効率的に推論を行うことが可能となります。
メモリに関しては, KV cacheを使用することで, VRAM使用量が増加し帯域幅も必要になるため, そのトレードオフも考慮する必要があります。例えば先ほど例に挙げたGPT-2 smallの場合, $B=1, L=12, H=12, HS=64$であるため, 1トークンあたりのKV cacheのサイズは32-bit float (FP32) の場合, 73KBとなり, 1,024トークンで約75MBとなります。この程度ならば, 現代のGPUでは十分に扱える範囲だと思いますが, より大きなモデルや長いシーケンスを扱う場合は注意が必要です。