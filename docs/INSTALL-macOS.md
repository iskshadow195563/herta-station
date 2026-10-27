# 在 macOS 上執行 Herta Station

支援 Apple Silicon（M1–M4）與 Intel Mac。macOS 沒有「顯存」這個概念：
llama.cpp 走 Metal，GPU 與 CPU 共用統一記憶體，程式會把可用上限估成實體記憶體的 75 %
（Metal 的 recommended working set），UI 上顯示成「統一記憶體」。

---

## 1. 先裝 llama.cpp

```bash
# 最快
brew install llama.cpp              # 提供 /opt/homebrew/bin/llama-server（Intel: /usr/local/bin）

# 想自己編（開 Metal）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(sysctl -n hw.ncpu)
# 產物：build/bin/llama-server
```

驗證：

```bash
llama-server --version     # 應該看到 metal: 相關字樣
```

## 2. 放模型

```bash
mkdir -p ~/llama.cpp/models
# 把 *.gguf 放進去
```

Apple Silicon 的建議：統一記憶體就是上限，`Q4_K_M` / `Q6_K` 這類量化在同樣記憶體下
可以多吃上下文；程式切換模型時會自動算好上下文與 KV 型別。

## 3. 安裝與執行

```bash
git clone <你的 repo> ~/herta-station && cd ~/herta-station

./start.sh --check     # 印出 GPU / 統一記憶體 / 路徑
./start.sh             # 原生 WKWebView 視窗

# 打包成 .app
./build.sh             # -> dist/HertaStation.app（已 ad-hoc 簽名）
./packaging/make-macos-dmg.sh        # -> dist/HertaStation.dmg

open dist/HertaStation.app
```

第一次打開若是用瀏覽器下載的 `.app`，macOS 會說「無法驗證開發者」：
右鍵 → 打開，或 `xattr -dr com.apple.quarantine dist/HertaStation.app`。

啟動後：

* 控制台 UI：<http://127.0.0.1:8899>
* OpenAI 相容 API：`http://0.0.0.0:11433/v1`
* Anthropic 相容端點：`http://0.0.0.0:11435`

## 4. 資料放哪裡

| 內容 | 路徑 |
| --- | --- |
| 設定 / 日誌 / PID | `~/Library/Application Support/HertaStation/` |
| 下載的模型（預設） | `~/llama.cpp/models`（存在時）否則 `~/Library/Application Support/HertaStation/models` |
| `.app` 本體 | 可放在 `/Applications`，唯讀不影響運作 |

macOS 的 `.app` 常被 quarantine 成唯讀，所以設定**不會**寫在 bundle 裡；
要改位置就用 `HERTA_DATA_DIR`。

## 5. 環境變數

```bash
export HERTA_LLAMA_EXE=/opt/homebrew/bin/llama-server   # 自動搜尋不到時
export HERTA_MODELS_DIR="$HOME/Models"
export HERTA_API_KEY="sk-local-xxx"                     # 開啟 Bearer 驗證
export HERTA_PORT=11433
open -a HertaStation                                    # 從 Finder 啟動時讀不到 shell 變數
```

> 從 Finder / Dock 啟動的 app 不會繼承 shell 的環境變數。需要用環境變數時，
> 直接在終端機執行 `dist/HertaStation.app/Contents/MacOS/HertaStation`，
> 或寫進 `~/Library/LaunchAgents`（見下）。

## 6. 開機自動啟動（可選）

```bash
cat > ~/Library/LaunchAgents/com.herta.station.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.herta.station</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Applications/HertaStation.app/Contents/MacOS/HertaStation</string>
    <string>--serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HERTA_LLAMA_EXE</key><string>/opt/homebrew/bin/llama-server</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
</dict>
</plist>
EOF
launchctl load ~/Library/LaunchAgents/com.herta.station.plist
```

## 7. 常見問題

| 症狀 | 原因與處理 |
| --- | --- |
| 視窗空白 / 打不開 | `--browser` 先確認服務正常；WKWebView 需要 macOS 11+ |
| 「無法驗證開發者」 | 右鍵 → 打開，或 `xattr -dr com.apple.quarantine HertaStation.app` |
| `找不到 llama-server` | `brew install llama.cpp`，或 `export HERTA_LLAMA_EXE=$(which llama-server)` |
| 顯存顯示成記憶體的 75 % | 正常：Apple Silicon 共用記憶體，這是 Metal 允許的可用上限 |
| 速度慢、一直 swap | 上下文太大吃了太多統一記憶體；降低上下文或換小一點的模型 |
| 風扇狂轉、GPU 沒吃滿 | `-ngl` 要設 99（預設就是），並確認 llama-server 是 Metal 版而非 CPU 版 |

## 8. 測試

```bash
.venv-unix/bin/python tests/smoke.py
.venv-unix/bin/python tests/test_hostos.py   # 含 sysctl / ps / 統一記憶體解析
```
