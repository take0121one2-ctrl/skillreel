"""
ボーカル録音 & ミックススクリプト
- AI生成の伴奏 (instrumental/output.wav) を読み込む
- マイクでボーカルをリアルタイム録音
- リバーブ・EQを簡易適用
- 伴奏とボーカルをミックスして mixed/final_mix.wav に出力
"""

import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal
from scipy.signal import butter, sosfilt

# Windows コンソールのUTF-8対応
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR      = os.path.dirname(__file__)
INSTRUMENTAL  = os.path.join(BASE_DIR, "instrumental", "output.wav")
VOCALS_DIR    = os.path.join(BASE_DIR, "vocals")
MIXED_DIR     = os.path.join(BASE_DIR, "mixed")
OUTPUT_PATH   = os.path.join(MIXED_DIR, "final_mix.wav")


# ─────────────────────────────────────────
#  エフェクト関数
# ─────────────────────────────────────────

def apply_reverb(audio: np.ndarray, sr: int,
                 room_size: float = 0.4, decay: float = 0.3) -> np.ndarray:
    """
    シンプルなコンボリューションリバーブ（FIRフィルタで近似）
    room_size: 0.0〜1.0  反響の長さ
    decay    : 0.0〜1.0  残響の減衰
    """
    delay_ms  = int(sr * room_size * 0.2)      # 最大 200ms
    ir_length = delay_ms
    if ir_length < 1:
        return audio

    # 指数減衰インパルス応答を生成
    ir = np.zeros(ir_length)
    ir[0] = 1.0
    for i in range(1, ir_length):
        ir[i] = ir[i - 1] * (1.0 - decay / ir_length * 10)
    ir = ir / (ir.sum() + 1e-8)

    # 畳み込み
    wet = np.convolve(audio, ir, mode="full")[: len(audio)]
    return audio * 0.7 + wet * 0.3


def apply_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    ボーカル向け簡易EQ
    - ローカット  : 80Hz以下を削減（ローエンドノイズ除去）
    - プレゼンス  : 2kHz〜5kHz をわずかに強調
    - ハイカット  : 12kHz以上を少し下げる（耳障りな高域除去）
    """
    nyq = sr / 2

    # ローカットフィルタ (80Hz, 4次)
    sos_lo = butter(4, 80 / nyq, btype="high", output="sos")
    audio = sosfilt(sos_lo, audio)

    # プレゼンスブースト (2kHz〜5kHz をわずかに持ち上げる簡易版)
    sos_pres = butter(2, [2000 / nyq, 5000 / nyq], btype="band", output="sos")
    presence = sosfilt(sos_pres, audio)
    audio = audio + presence * 0.25

    # ハイカットフィルタ (12kHz)
    if 12000 / nyq < 1.0:
        sos_hi = butter(2, 12000 / nyq, btype="low", output="sos")
        audio = sosfilt(sos_hi, audio)

    return audio


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """ピーク正規化"""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * target_peak
    return audio


# ─────────────────────────────────────────
#  録音
# ─────────────────────────────────────────

def record_vocals(duration_sec: int, sr: int = 44100) -> np.ndarray:
    """
    マイクからリアルタイム録音
    duration_sec: 録音秒数
    """
    print(f"\n[REC] {duration_sec}秒間録音します...")
    print("      3秒後に開始します。マイクに向かって歌ってください。")

    # カウントダウン
    import time
    for i in range(3, 0, -1):
        print(f"      {i}...")
        time.sleep(1)
    print("      ★ 録音開始! ★")

    recording = sd.rec(
        int(duration_sec * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("[OK] 録音完了")
    return recording.flatten()


# ─────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ボーカル録音 & ミックスツール")
    print("=" * 55)

    # ── 伴奏の読み込み ──
    if not os.path.exists(INSTRUMENTAL):
        print(f"[ERROR] 伴奏ファイルが見つかりません: {INSTRUMENTAL}")
        print("        先に test_musicgen.py を実行して伴奏を生成してください")
        sys.exit(1)

    instrumental, instr_sr = sf.read(INSTRUMENTAL, dtype="float32")
    if instrumental.ndim == 2:
        instrumental = instrumental.mean(axis=1)   # ステレオ→モノラル
    print(f"[OK] 伴奏読み込み完了 ({len(instrumental)/instr_sr:.1f}秒, {instr_sr}Hz)")

    # ── 録音設定 ──
    vocal_sr  = 44100
    duration  = int(len(instrumental) / instr_sr)
    duration  = min(duration, 120)  # 最大2分

    print(f"\n伴奏の長さに合わせて {duration}秒 録音します")
    answer = input("録音を開始しますか？ [Y/n]: ").strip().lower()
    if answer == "n":
        print("録音をスキップします。vocals/vocal.wav が存在すれば使用します。")
        vocal_path = os.path.join(VOCALS_DIR, "vocal.wav")
        if os.path.exists(vocal_path):
            vocals, vocal_sr = sf.read(vocal_path, dtype="float32")
            if vocals.ndim == 2:
                vocals = vocals.mean(axis=1)
            print(f"[OK] 既存ボーカル読み込み: {vocal_path}")
        else:
            print("[ERROR] vocals/vocal.wav も存在しません。終了します。")
            sys.exit(1)
    else:
        vocals = record_vocals(duration, vocal_sr)

    # ── ボーカルのエフェクト適用 ──
    print("\n[INFO] EQ適用中...")
    vocals = apply_eq(vocals, vocal_sr)

    print("[INFO] リバーブ適用中...")
    vocals = apply_reverb(vocals, vocal_sr, room_size=0.35, decay=0.25)

    vocals = normalize(vocals)
    print("[OK] エフェクト適用完了")

    # 録音ファイルを保存
    os.makedirs(VOCALS_DIR, exist_ok=True)
    vocal_save = os.path.join(VOCALS_DIR, "vocal.wav")
    sf.write(vocal_save, vocals, vocal_sr)
    print(f"[OK] 加工済みボーカル保存: {vocal_save}")

    # ── サンプルレートを統一 ──
    if vocal_sr != instr_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(instr_sr, vocal_sr)
        vocals = resample_poly(vocals, instr_sr // g, vocal_sr // g)
        vocal_sr = instr_sr

    # ── 長さを揃える（短い方に合わせる）──
    min_len = min(len(instrumental), len(vocals))
    instrumental = instrumental[:min_len]
    vocals       = vocals[:min_len]

    # ── ミックス ──
    print("\n[INFO] ミックス中...")
    INSTR_VOL = 0.75   # 伴奏の音量比
    VOCAL_VOL = 1.00   # ボーカルの音量比

    mixed = instrumental * INSTR_VOL + vocals * VOCAL_VOL
    mixed = normalize(mixed)

    # ── 出力 ──
    os.makedirs(MIXED_DIR, exist_ok=True)
    sf.write(OUTPUT_PATH, mixed, instr_sr)

    duration_out = len(mixed) / instr_sr
    print(f"\n[完了] ミックス完成!")
    print(f"       保存先 : {OUTPUT_PATH}")
    print(f"       長さ   : {duration_out:.1f}秒")
    print()
    print("exports/ フォルダに入れてSNS投稿用に使えます")


if __name__ == "__main__":
    main()
