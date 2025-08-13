# 文書画像セグメンテーションアプリケーション

Detectron2で学習されたMask R-CNNモデル（model.onnx）を使用して、文書画像から文字列を抽出・セグメンテーションするGUIアプリケーションです。

## 機能

### 1. ファイル読み込み
- **対応形式**: PDF、PNG、JPG、JPEG
- **操作**: 「ファイルを開く」ボタンからファイルを選択
- **表示**: 選択したファイルがUI上に表示される

### 2. PDFページ操作
- **ページ切り替え**: 「前」「次」ボタンでページを切り替え
- **ページ表示**: 現在のページ数と総ページ数を表示

### 3. 表示制御
- **拡大率**: スライダーで0.1倍～3.0倍まで調整可能
- **表示位置**: X、Y位置スライダーで表示位置を調整
- **スクロール**: マウスホイールやスクロールバーで画像を移動

### 4. 前処理機能
- **二値化**: チェックボックスで有効/無効、閾値スライダーで調整（0-255）
- **回転**: スライダーで-180度～+180度まで回転可能
- **歪み補正**: チェックボックスで簡易的な歪み補正を適用

### 5. 文字認識
- **認識実行**: 「認識実行」ボタンでONNXモデルによる文字領域検出
- **結果表示**: 検出されたバウンディングボックスが画像上に表示

### 6. バウンディングボックス操作
- **一括サイズ調整**: スライダーで全てのバウンディングボックスを一括拡大・縮小
- **個別編集**: マウスで各バウンディングボックスを個別に操作
  - クリックで選択（赤色で表示）
  - 辺をドラッグしてサイズ変更
  - 中央をドラッグして移動
  - 角の操作ポイントが表示される

## 使用方法

### 1. 環境構築
```bash
# uvでプロジェクトを初期化（既に完了）
uv init

# 依存関係のインストール（既に完了）
uv add opencv-python onnxruntime pillow numpy PyMuPDF matplotlib
```

### 2. アプリケーション起動
```bash
uv run python main.py
```

### 3. 基本操作手順
1. **ファイル選択**: 「ファイルを開く」ボタンから画像またはPDFを選択
2. **表示調整**: 拡大率や位置スライダーで見やすく調整
3. **前処理**: 必要に応じて二値化、回転、歪み補正を適用
4. **認識実行**: 「認識実行」ボタンで文字領域を検出
5. **結果調整**: バウンディングボックスのサイズや位置を調整

## ファイル構成

```
line_segmentation_app/
├── main.py          # メインアプリケーション
├── model.onnx       # Detectron2 Mask R-CNNモデル
├── pyproject.toml   # プロジェクト設定
└── README.md        # このファイル
```

## モデルについて

- **形式**: ONNX
- **ベース**: Detectron2 Mask R-CNN
- **用途**: 文書画像からの文字列抽出
- **入力**: 画像データ（640x640にリサイズされます）
- **出力**: バウンディングボックス座標のリスト（出力インデックス0番目）

## カスタマイズ

### モデルの出力形式調整
モデルの出力形式に応じて、`run_recognition()`メソッド内の出力処理部分を調整してください：

```python
# 出力の最初の要素がバウンディングボックスと仮定
predictions = outputs[0]

# 出力形式に応じて調整が必要
if len(predictions.shape) >= 2:
    for detection in predictions[0]:  # バッチの最初を取得
        if len(detection) >= 4:
            # 座標を元の画像サイズに戻す
            x1 = int(detection[0] / scale_x)
            y1 = int(detection[1] / scale_y)
            x2 = int(detection[2] / scale_x)
            y2 = int(detection[3] / scale_y)
            
            # 信頼度チェック（信頼度が含まれている場合）
            confidence = detection[4] if len(detection) > 4 else 1.0
            if confidence > 0.5:  # 閾値
                self.bounding_boxes.append([x1, y1, x2, y2])
```

### 前処理の拡張
より高度な前処理を追加する場合は、`apply_preprocessing()`メソッドを拡張してください。

## 注意事項

- model.onnxファイルが同じディレクトリに存在する必要があります
- モデルの出力形式によっては、認識結果の処理部分の調整が必要です
- 大きなPDFファイルの場合、メモリ使用量にご注意ください
- バウンディングボックスの操作は、拡大表示時により精密に行えます

## トラブルシューティング

### モデル読み込みエラー
- model.onnxファイルの存在を確認
- ファイルが破損していないか確認
- ONNXランタイムのバージョンが適合しているか確認

### 認識結果が表示されない
- モデルの出力形式を確認
- 信頼度閾値を調整
- 前処理パラメータを調整

### 表示が崩れる
- 拡大率を調整
- 表示位置を初期化（スライダーを中央に戻す）
- アプリケーションを再起動

---

# Document Image Segmentation Application

A GUI application that uses a Detectron2-trained Mask R-CNN model (model.onnx) to extract and segment text strings from document images.

## Features

### 1. File Loading
- **Supported formats**: PDF, PNG, JPG, JPEG
- **Operation**: Select files using the "Open File" button
- **Display**: Selected files are displayed in the UI

### 2. PDF Page Operations
- **Page navigation**: Switch pages using "Previous" and "Next" buttons
- **Page display**: Shows current page number and total page count

### 3. Display Controls
- **Zoom**: Adjustable from 0.1x to 3.0x using slider
- **Position**: Adjust display position with X, Y position sliders
- **Scroll**: Move image using mouse wheel or scroll bars

### 4. Preprocessing Features
- **Binarization**: Enable/disable with checkbox, adjust threshold with slider (0-255)
- **Rotation**: Rotate from -180° to +180° using slider
- **Distortion correction**: Apply simple distortion correction with checkbox

### 5. Text Recognition
- **Run recognition**: Detect text regions using ONNX model with "Run Recognition" button
- **Result display**: Detected bounding boxes are displayed on the image

### 6. Bounding Box Operations
- **Batch size adjustment**: Scale all bounding boxes simultaneously using sliders
- **Individual editing**: Manipulate each bounding box individually with mouse
  - Click to select (displayed in red)
  - Drag edges to resize
  - Drag center to move
  - Corner control points are displayed

## Usage

### 1. Environment Setup
```bash
# Initialize project with uv (already completed)
uv init

# Install dependencies (already completed)
uv add opencv-python onnxruntime pillow numpy PyMuPDF matplotlib
```

### 2. Launch Application
```bash
uv run python main.py
```

### 3. Basic Operation Steps
1. **File selection**: Select image or PDF using "Open File" button
2. **Display adjustment**: Adjust zoom and position sliders for better viewing
3. **Preprocessing**: Apply binarization, rotation, or distortion correction as needed
4. **Run recognition**: Detect text regions using "Run Recognition" button
5. **Result adjustment**: Adjust bounding box sizes and positions

## File Structure

```
line_segmentation_app/
├── main.py          # Main application
├── model.onnx       # Detectron2 Mask R-CNN model
├── pyproject.toml   # Project configuration
└── README.md        # This file
```

## About the Model

- **Format**: ONNX
- **Base**: Detectron2 Mask R-CNN
- **Purpose**: Text extraction from document images
- **Input**: Image data (resized to 640x640)
- **Output**: List of bounding box coordinates (output index 0)

## Customization

### Model Output Format Adjustment
Adjust the output processing in the `run_recognition()` method according to your model's output format:

```python
# Assume first element of output is bounding boxes
predictions = outputs[0]

# Adjust according to output format
if len(predictions.shape) >= 2:
    for detection in predictions[0]:  # Get first batch
        if len(detection) >= 4:
            # Convert coordinates to original image size
            x1 = int(detection[0] / scale_x)
            y1 = int(detection[1] / scale_y)
            x2 = int(detection[2] / scale_x)
            y2 = int(detection[3] / scale_y)
            
            # Confidence check (if confidence is included)
            confidence = detection[4] if len(detection) > 4 else 1.0
            if confidence > 0.5:  # Threshold
                self.bounding_boxes.append([x1, y1, x2, y2])
```

### Extending Preprocessing
To add more advanced preprocessing, extend the `apply_preprocessing()` method.

## Notes

- The model.onnx file must exist in the same directory
- Processing of recognition results may need adjustment depending on the model's output format
- For large PDF files, please be mindful of memory usage
- Bounding box operations can be performed more precisely when zoomed in

## Troubleshooting

### Model Loading Error
- Check if model.onnx file exists
- Verify the file is not corrupted
- Ensure ONNX Runtime version compatibility

### Recognition Results Not Displayed
- Check model output format
- Adjust confidence threshold
- Adjust preprocessing parameters

### Display Issues
- Adjust zoom level
- Reset display position (return sliders to center)
- Restart the application
