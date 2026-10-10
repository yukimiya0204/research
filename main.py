import argparse #ターミナルで入力した --source　のような指定を読み取る
import json # JSON形式で結果を出力するために使用
import sys # インタプリタや情報取得に関する制御をサポートする
from pathlib import Path # path操作をサポートする path.name でファイル名、 path.suffix で拡張子、path.parentで親ディレクトリを取得する


CLASS_COUNT = 3 # 分類するクラスの数　今回は apple, banana, orange の3クラス


def load_yolo(weights): # def: 関数定義　weights: は引数, Yoloを読み込む関数
	try:                # try: 一旦これを試すという意味
		from ultralytics import YOLO
	except ImportError as error: # except ImportError: ultralyticsがないときは以下を実行する。
		raise RuntimeError( # raise: わざと例外を発生させる
			"ultralytics がありません。`pip install -r requirements.txt` を実行してください。"
		) from error

	return YOLO(weights) # YOLOモデルを返す



def class_names_from_model(model): #model:　は引数、　クラス名を取り出す関数
	names = model.names #modelの名前を代入  modelは0なら犬などの表を持っている
	if isinstance(names, dict): #dict: は辞書　この関数は後で調べる
		names = [names[class_id] for class_id in sorted(names)]

	if len(names) != CLASS_COUNT:# len:　個数をかぞえる。　3個でなければエラーになる
		raise ValueError(
			f"モデルのクラス数は{len(names)}です。3クラスのモデルを指定してください。"
		)

	return list(names)


def classify_video(args): # このプログラムの中心　
	try:
		import cv2 #import cv2 は動画や画像を読み込むための道具（openCV）関数の中で読み込む 
	except ImportError as error:
		raise RuntimeError(
			"opencv-python がありません。`pip install -r requirements.txt` を実行してください。"
		) from error

	if not Path(args.source).is_file(): #動画ファイルがなければ強制的に止める処理
		raise FileNotFoundError(f"動画ファイルが見つかりません: {args.source}")

	model = load_yolo(args.weights)
	class_names = class_names_from_model(model)
	capture = cv2.VideoCapture(args.source)
	
	if not capture.isOpened(): #開けないときエラー
		capture.release()
		raise ValueError(f"動画を開けません: {args.source}")

	score_totals = {name: 0.0 for name in class_names} #自信度の合計
	detected_frames = {name: 0 for name in class_names} #検出できたコマ数
	sampled_frames = 0              #調べたコマ数
	frame_index = 0                 #現在何コマ目か
	predict_options = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
	if args.device:
		predict_options["device"] = args.device

	try:
		while True:
			success, frame = capture.read()
			if not success:
				break

			if frame_index % args.frame_stride == 0:
				sampled_frames += 1
				results = model.predict(source=frame, **predict_options)
				frame_scores = {}

				for result in results:
					if result.boxes is None:
						continue
					for class_id, confidence in zip(
						result.boxes.cls.tolist(), result.boxes.conf.tolist()
					):
						name = class_names[int(class_id)]
						frame_scores[name] = max(
							frame_scores.get(name, 0.0), float(confidence)
						)

				for name, score in frame_scores.items():
					score_totals[name] += score
					detected_frames[name] += 1

			frame_index += 1
	finally:
		capture.release()

	if sampled_frames == 0:
		raise ValueError("動画からフレームを読み込めませんでした。")

	class_scores = {
		name: score_totals[name] / sampled_frames for name in class_names
	}
	prediction = max(class_scores, key=class_scores.get)
	if score_totals[prediction] == 0:
		prediction = None

	report = {
		"class": prediction,
		"mean_score": round(class_scores[prediction], 6) if prediction else 0.0,
		"sampled_frames": sampled_frames,
		"detected_frames": detected_frames,
		"class_scores": {
			name: round(score, 6) for name, score in class_scores.items()
		},
	}
	print(json.dumps(report, ensure_ascii=False, indent=2))


def train_model(args):
	if not Path(args.data).is_file():
		raise FileNotFoundError(f"データ設定ファイルが見つかりません: {args.data}")

	model = load_yolo(args.model)
	model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


def build_parser():
	parser = argparse.ArgumentParser(
		description="YOLOで動画中を通過する物体を3クラス分類します。"
	)
	commands = parser.add_subparsers(dest="command", required=True)

	train_parser = commands.add_parser("train", help="ラベル付きデータでYOLOを学習")
	train_parser.add_argument("--data", required=True, help="データセットYAML")
	train_parser.add_argument("--model", default="yolo11n.pt", help="初期モデル")
	train_parser.add_argument("--epochs", type=int, default=100)
	train_parser.add_argument("--imgsz", type=int, default=640)
	train_parser.set_defaults(handler=train_model)

	classify_parser = commands.add_parser("classify", help="動画を分類")
	classify_parser.add_argument("--weights", required=True, help="学習済み .pt")
	classify_parser.add_argument("--source", required=True, help="入力動画")
	classify_parser.add_argument("--conf", type=float, default=0.25)
	classify_parser.add_argument("--imgsz", type=int, default=640)
	classify_parser.add_argument(
		"--frame-stride", type=int, default=1, help="何フレームごとに推論するか"
	)
	classify_parser.add_argument("--device", help="例: cpu, 0")
	classify_parser.set_defaults(handler=classify_video)
	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()
	if getattr(args, "frame_stride", 1) < 1:
		parser.error("--frame-stride は1以上にしてください。")

	try:
		args.handler(args)
	except (FileNotFoundError, RuntimeError, ValueError) as error:
		print(f"エラー: {error}", file=sys.stderr)
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
