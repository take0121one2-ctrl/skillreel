"""
MusicGen 動作確認スクリプト
Meta製MusicGenをHugging Face経由でロードしてJ-pop伴奏を生成します。
AMD GPU / CPU環境に対応。
"""

import sys
import os
import torch
import soundfile as sf
import numpy as np

# Windows コンソールのUTF-8対応
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "instrumental")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.wav")

def main():
    print("=" * 55)
    print("  MusicGen J-pop 伴奏生成スクリプト")
    print("=" * 55)

    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用デバイス: {device}")
    if device == "cpu":
        print("[INFO] CPUモードで動作します（生成に数分かかります）")

    print("[INFO] MusicGenモデルをロード中...")
    print("       初回は数百MBのダウンロードが発生します...")

    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError:
        print("[ERROR] transformers がインストールされていません")
        print("        実行: python -m pip install transformers")
        sys.exit(1)

    # モデルロード（small = 約300MB、高速・軽量）
    model_id = "facebook/musicgen-small"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] モデルのロードに失敗しました: {e}")
        print("        インターネット接続を確認してください")
        sys.exit(1)

    print("[OK] モデルロード完了")

    # 生成プロンプト
    prompt = (
        "J-pop emotional piano guitar, "
        "cinematic build-up, King Gnu style, "
        "modern Japanese pop, 30 seconds"
    )
    print(f"[INFO] プロンプト: {prompt}")

    # トークナイズ
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # サンプル数の計算
    # MusicGen は 50トークン/秒、モデルの上限は2048トークン
    sampling_rate = model.config.audio_encoder.sampling_rate
    FRAME_RATE    = 50          # MusicGen の固定フレームレート
    DURATION_SEC  = 30
    MAX_MODEL_POS = 2047        # embed_positions の上限インデックス
    max_new_tokens = min(int(DURATION_SEC * FRAME_RATE), MAX_MODEL_POS)

    print(f"[INFO] 音楽を生成中... ({DURATION_SEC}秒 / {max_new_tokens}トークン)")
    print("       CPUでは5〜15分かかる場合があります。お待ちください...")

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    # テンソルをnumpy配列に変換
    audio_data = audio_values[0, 0].cpu().numpy()

    # 正規化（クリッピング防止）
    if audio_data.max() > 0:
        audio_data = audio_data / np.abs(audio_data).max() * 0.9

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sf.write(OUTPUT_PATH, audio_data, samplerate=sampling_rate)

    duration = len(audio_data) / sampling_rate
    print(f"[OK] 生成完了!")
    print(f"     保存先: {OUTPUT_PATH}")
    print(f"     長さ  : {duration:.1f}秒")
    print(f"     サンプリングレート: {sampling_rate}Hz")
    print()
    print("次のステップ: python mix_vocals.py を実行してボーカルと合成できます")


if __name__ == "__main__":
    main()
