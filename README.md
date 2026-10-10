# 動画中の物体をYOLOで分類

動画をフレームごとにYOLOで検出し、検出スコアを動画全体で集約して、最も強く検出されたクラスを1つ出力します。モデルはリンゴ・バナナ・オレンジの3クラスを想定しています。別のクラスにする場合は、データセットYAMLの `names` を変更してください。

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 学習データ

YOLOの物体検出形式で、画像と同じファイル名の `.txt` ラベルを用意します。各ラベル行は `クラスID x中心 y中心 幅 高さ` で、座標・幅・高さは画像サイズに対する0から1の値です。動画そのものではなく、動画から切り出したフレーム画像を学習データにします。

データセットYAMLの例（`dataset.yaml`）:

```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: apple
  1: banana
  2: orange
```

画像・ラベルはそれぞれ `dataset/images/train` と `dataset/labels/train`、検証用は `dataset/images/val` と `dataset/labels/val` に配置します。クラスIDはYAMLの順番に合わせます。

## 学習

```powershell
python main.py train --data dataset.yaml --epochs 100
```

学習後に表示される保存先の `best.pt` を推論に使います。GPUを使える環境ではUltralyticsが通常自動選択します。

## 動画を分類

```powershell
python main.py classify --weights runs/detect/train/weights/best.pt --source test.mp4
```

標準では全フレームを推論します。処理を軽くしたい場合は `--frame-stride 2` のように指定できます。結果はJSONで出力され、`class` が動画全体の判定結果です。検出がない場合は `null` になります。`mean_score` は全サンプルフレームに対する検出スコアの平均で、確率として校正された値ではありません。