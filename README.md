# Herta Station · Qwen3.8-27B UD-Q3_K_XL 本地推理終端

單一 EXE，GUI 內建（WebView2）；OpenAI 相容，區網可呼叫。

## 1. 開程式

```
D:\llama.cpp\HertaStation.exe
```

桌面捷徑 **Herta Space Station** 已指向它。關視窗＝只關介面，模型仍在服務；
要真的釋放顯存，按「停止模組」。

| 按鈕 | 行為 |
| --- | --- |
| 啟動模組 | spawn `llama-server`，載入 27B Q3_K_XL 到 GPU |
| 停止模組 | taskkill 整個程序樹，顯存回到 ~1.5 GB |
| 清空顯存 | 先 `ollama stop` 所有模型，再停 llama-server |
| 連線測試 | 送一個小 chat，回傳延遲＋回覆 |

## 2. 呼叫端點（OpenAI 格式）

```
本機   http://127.0.0.1:11433/v1
區網   http://192.168.31.10:11433/v1     ← 乙太網 2
       http://192.168.100.50:11433/v1    ← 乙太網
Model  qwen3.8-27b-q3xl
Key    目前不需要（見 §2.2）
```

```bash
curl http://192.168.31.10:11433/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-q3xl","messages":[{"role":"user","content":"你好"}],"chat_template_kwargs":{"enable_thinking":false}}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.31.10:11433/v1", api_key="local")
print(client.chat.completions.create(
    model="qwen3.8-27b-q3xl",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
).choices[0].message.content)
```

- 端點：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/health`、`/props`、`/slots`。
- **沒有** Anthropic `/v1/messages`，不要接。
- 思考內容在 `message.reasoning_content`；只要答案就加 `chat_template_kwargs.enable_thinking=false`。
- 視覺：預設關閉，設 `HERTA_MMPROJ=1` 才載入投影層（+0.9 GB，此時把 `HERTA_CTX` 降到 24576）。

### 2.1 區域網路開放

llama-server 綁 `0.0.0.0:11433`，同網段任何機器（手機、另一台 PC、WSL、容器）都能直接打。

- 控制台（8899）**仍只綁 127.0.0.1**：LAN 上碰不到 start/stop，只能呼叫模型。
- 防火牆：`llama-server.exe` 已有 inbound allow 規則（Public 設定檔，剛好對上乙太網 2）。
  換網卡、或該網路被判定成「私人」而被擋時，執行 `D:\llama.cpp\open-lan-firewall.bat`
  （會要求管理員權限，放行 TCP 11433）。
- 已知介面：乙太網 2 `192.168.31.10`（Internet）、乙太網 `192.168.100.50`。

### 2.2 上鎖（可選）

目前沒有 API key，同網段任何人都能燒你的顯卡。設定環境變數後重開模型：

```bat
set HERTA_API_KEY=sk-herta-1234
```

之後所有請求要帶 `Authorization: Bearer sk-herta-1234`（llama.cpp 也認 `LLAMA_API_KEY`）。
GUI 遙測列會顯示「API KEY 已啟用 / 無 API KEY」。

## 3. 等價的 CLI 啟動（不開 GUI）

```bat
D:\llama.cpp\b11756\llama-server.exe ^
  -m "D:\ollama-gguf\qwen3.8-27b\Qwen3.8-27B-UD-Q3_K_XL.gguf" ^
  -a qwen3.8-27b-q3xl --host 0.0.0.0 --port 11433 ^
  -c 65536 -ngl 99 -np 1 -fa on -ctk q4_0 -ctv q4_0 --jinja
```

停止：`D:\llama.cpp\stop-qwen38.bat`（或 GUI）。只給本機用就把 `--host` 換回 `127.0.0.1`。

## 4. 為什麼是 49152 + q5_1

DSH 一輪請求的 prompt 實測 **38085 tokens**（系統提示＋工具定義＋歷史），
8192 的上下文必定爆 `exceed_context_size_error`；接著 DSH 會嘗試 compaction 摘要，
摘要又被同一上限擋住，錯誤路徑才吐出
`format v4 message requires a producer-owned source kind`。兩個錯誤同源。

**16 GB 的隱形懸崖**：VRAM 一旦超過約 **15.3 GB**，生成速度掉 10 倍（WDDM 換頁）：

| ctx | KV | VRAM | 生成速度 |
| ---: | --- | ---: | ---: |
| 8192 | q8_0 | 14.8 GB | 44 tok/s |
| 24576 | q8_0 | 15.5 GB | 42 tok/s |
| 32768 | q8_0 | 15.4 GB | 19.7 tok/s |
| 40960 | q8_0 | 15.6 GB | 12.9 tok/s |
| 49152 | q8_0 | 15.7 GB | 3.1 tok/s |
| **49152** | **q5_1** | **15.25 GB** | **47 tok/s**（38k 上下文時 31.8） |
| 49152 | q4_0 | 15.0 GB | 47 tok/s（38k 上下文時 37.0） |

結論：`-c 49152 -ctk q5_1 -ctv q5_1`。38k prompt prefill 約 1.7k tok/s（≒23 秒）。

## 5. 早期對照

| 設定 | VRAM | 生成速度 |
| --- | ---: | ---: |
| -c 8192 KV=f16 | 15.7 GB | 8.2 tok/s |
| -c 8192 KV=q8_0 | 15.5 GB | 34–42 tok/s |
| -c 16384 KV=q8_0 | 15.5 GB | 19.5 tok/s |

不是「KV 越精確越好」，而是「VRAM 不能超過 15.3 GB」。

## 6. 檔案

| 路徑 | 說明 |
| --- | --- |
| `D:\llama.cpp\HertaStation.exe` | 主程式（GUI 內建，12.3 MB） |
| `D:\llama.cpp\herta\herta_app.py` | EXE 入口（pywebview 視窗） |
| `D:\llama.cpp\herta\server.py` | 控制 API：start / stop / purge / selftest / status |
| `D:\llama.cpp\herta\ui.html` | 黑塔風格介面 |
| `D:\llama.cpp\herta\herta.ico` | 圖示 |
| `D:\llama.cpp\open-lan-firewall.bat` | 放行 TCP 11433（管理員） |
| `D:\llama.cpp\start-qwen38.bat` / `stop-qwen38.bat` | CLI 啟停 |
| `D:\llama.cpp\HertaStation.exe` 同目錄 `llama-server.log` | 執行日誌 |
| `D:\ollama-gguf\qwen3.8-27b\` | 權重 13.15 GB + mmproj 0.93 GB |
| `D:\llama.cpp\herta\hostos.py` | 平台層：程序／埠／GPU／路徑，Windows·Linux·macOS 共用 |
| `D:\llama.cpp\herta\models.py` | 模型庫掃描、GGUF 表頭解析、HF 搜尋、顯存規劃 |
| `D:\llama.cpp\herta\anthropic_proxy.py` | Anthropic `/v1/messages` 轉接（埠 11435） |
| `D:\llama.cpp\herta\build.sh` / `start.sh` | Linux / macOS 打包與啟動 |
| `D:\llama.cpp\herta\packaging\` | `.desktop`、`install-linux.sh`、`make-macos-dmg.sh`、SVG 圖示 |
| `D:\llama.cpp\herta\tests\` | `smoke.py`（57 tests）+ `test_hostos.py`（30 tests） |
| `D:\llama.cpp\herta\docs\INSTALL-linux.md` / `INSTALL-macOS.md` | 兩平臺安裝說明 |

## 7. 重新打包

一鍵產出三平臺安裝包（llama.cpp 與 herta-rs 都塞進去了，使用者不必自己編 llama.cpp）：

```bash
python tools/fetch-llama.py --platform all      # 抓 llama.cpp b11934 預編包到 vendor/
cargo build --release --manifest-path tools/herta-rs/Cargo.toml
python tools/make_dist.py --platform all        # -> dist/ 四份產物
```

| 產物 | 大小 | 內容 |
| --- | --- | --- |
| `HertaStation-Setup-1.1.0.exe` | 58.9 MB | Windows 安裝器：解到 `%LOCALAPPDATA%\Programs\HertaStation`，建桌面／開始選單捷徑，含 Vulkan llama.cpp + herta-rs |
| `HertaStation.exe` | 13.0 MB | Windows 免安裝單檔（自己找 `HERTA_LLAMA_EXE` 或安裝版） |
| `herta-station_1.1.0_amd64.deb` | 46.8 MB | `/opt/herta-station/{herta-station.pyz, llama.cpp, bin/herta-rs}` + `/usr/bin/herta-station` + 桌面檔 |
| `HertaStation-1.1.0-macos-arm64.zip` | 29.2 MB | `HertaStation.app`（Metal llama.cpp 內建，ad-hoc 簽名） |
| `HertaStation-1.1.0-macos-x64.zip` | 22.8 MB | 同上，Intel 版 |

只重打某一平臺：`python tools/make_dist.py --platform win|linux|macos-arm64|macos-x64`。
單檔 exe 還是用 PyInstaller 直接打：

```bat
D:\llama.cpp\herta\build\.venv\Scripts\pyinstaller.exe --noconfirm --onefile --windowed ^
  --name HertaStation --icon D:\llama.cpp\herta\herta.ico ^
  --add-data "D:\llama.cpp\herta\ui.html;." ^
  --collect-all webview --collect-all clr_loader --paths D:\llama.cpp\herta ^
  --distpath D:\llama.cpp\herta\dist --workpath D:\llama.cpp\herta\build\work ^
  --specpath D:\llama.cpp\herta\build D:\llama.cpp\herta\herta_app.py
```

改了 `ui.html` 或 `server.py` 都要重打包。

## 7.1 推薦分頁與下載引擎

模型庫多一個 **推薦** 分頁：八個精選模型（Strata 引薦的 Qwen3.8-Flash-Next Q2_0、27B Q3_K_XL、
9B／4B Q4_K_XL、Qwen3.6-35B-A3B Q3_K_XL、Qwen3-1.7B、Gemma 3 4B QAT、Llama-3.2-3B），
每張卡片直接算好「權重／佔盤／這張卡能不能跑」，按一下 `下載` 就開始，完成後自動設為使用中的模型。

下載走 `tools/herta-rs`（Rust，2 MB）：把檔案切成 6 段並行 HTTP Range、支援 `--resume` 續傳、
每 250 ms 輸出 `{"done":…,"total":…}` 給 Python 解析；沒有 herta-rs 就自動退回純 Python 串流。
實測 907 MB 檔案 12.5 秒（72 MB/s）。`herta-rs info <gguf>` 另外能印出 GGUF 版本／張量數。

## 8. 環境變數

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `HERTA_CTX` | 65536 | 上下文長度（DSH prompt ≈ 38k） |
| `HERTA_PARALLEL` | 1 | 並行 slot |
| `HERTA_FA` | on | flash attention |
| `HERTA_CTK` / `HERTA_CTV` | q4_0 | KV cache 型別（撞懸崖就換 q4_0） |
| `HERTA_MMPROJ` | 0 | 1 = 載入視覺投影（+0.9 GB） |
| `HERTA_BIND` | 0.0.0.0 | 綁定位址；改 127.0.0.1 即只限本機 |
| `HERTA_API_KEY` | 空 | 設定後 LAN 請求要 Bearer token |
| `HERTA_PORT` | 11433 | llama-server 埠 |
| `HERTA_LLAMA_EXE` | 自動搜尋 | llama-server 路徑（三平臺通用；安裝包內建優先） |
| `HERTA_LLAMA_DIR` | 自動 | 內建 llama.cpp 目錄（安裝器／deb 會自己設） |
| `HERTA_RS` | 自動搜尋 | herta-rs 下載器路徑（沒有就用 Python 串流） |
| `HERTA_MODELS_DIR` | 自動 | 模型庫目錄 |
| `HERTA_DATA_DIR` | 自動 | 設定／日誌／PID 目錄 |
| `HERTA_UI_PORT` | 8899 | 控制台埠（測試時可換一個跑第二份） |

## 9. Ubuntu / macOS

三平臺共用同一份原始碼；作業系統差異（程序樹、埠→PID、GPU 探測、資料目錄）全部集中在 `hostos.py`。
**一般使用者不必自己裝 llama.cpp**：安裝包已經把對應版本的 `llama-server` 與 `herta-rs` 放在裡面。

```bash
# Ubuntu / Debian
sudo dpkg -i herta-station_1.1.0_amd64.deb   # 內含 llama.cpp(Vulkan) + herta-rs
herta-station                                 # 或從「應用程式」開
herta-station --check                         # 印出 GPU / 顯存 / 路徑

# macOS
unzip HertaStation-1.1.0-macos-arm64.zip      # 解出 HertaStation.app
xattr -dr com.apple.quarantine HertaStation.app   # 第一次要繞過隔離屬性
open HertaStation.app
```

macOS 另外有兩個免安裝 Python 的版本：`HertaStation-1.1.0-macos-arm64.dmg` 與
`HertaStation-1.1.0-macos-arm64-native.zip`（內建直譯器與 llama.cpp，拖進 Applications 就能用）；
上面那個 `.zip` 則是交給系統 `python3`（3.10+）執行的輕量版。

從原始碼跑（開發用）：

```bash
git clone <repo> ~/herta-station && cd ~/herta-station
./start.sh --check                    # 印出 GPU / 顯存 / 路徑
./build.sh --qt                       # 打包（Qt WebEngine 內建；--headless 則用瀏覽器）
./packaging/install-linux.sh          # 裝到 ~/.local + 桌面捷徑
```

| 差異 | Windows | Linux | macOS |
| --- | --- | --- | --- |
| 停止模型 | `taskkill /T /F` | `killpg(SIGTERM→SIGKILL)` | 同 Linux |
| 埠→PID | `netstat -ano` | `ss -ltnpH` → `lsof` → `/proc/net/tcp` | `lsof` |
| GPU | nvidia-smi / WMI | nvidia-smi / AMD sysfs | Metal 統一記憶體（`sysctl` + `system_profiler`） |
| 顯存上限 | 總量 96.5%，懸崖 15.3 GB | 同 Windows | 實體記憶體 75%（Metal working set） |
| 資料目錄 | EXE 同層 | `$XDG_DATA_HOME/HertaStation` | `~/Library/Application Support/HertaStation` |

細節與疑難排解：`docs/INSTALL-linux.md`、`docs/INSTALL-macOS.md`。

沒有 Linux 機器也可以產出 Linux 版：

```bash
python tools/fetch-llama.py --platform linux
python tools/make_dist.py --platform linux          # -> dist/herta-station_1.1.0_amd64.deb
# 或者整包在容器裡打：
docker build -f packaging/Dockerfile -t herta-build .
docker run --rm -v "$PWD/dist:/out" herta-build
```

`.github/workflows/build.yml` 五個 job：`tests`（全平臺共用測試）、`windows`（exe + 安裝器）、
`linux`（deb，並在 CI 裡 `dpkg -i` 後跑 `--check`）、`macos-arm64`（zip + dmg）、`macos-x64`（zip）。
每個 job 都自己 `cargo build` 出對應版本的 herta-rs、自己抓 llama.cpp 預編包，所以 repo 不必背 360 MB 二進位。

## 10. 跨平臺測試

```bash
./build/.venv/Scripts/python.exe tests/smoke.py   # Windows: 67 tests
.venv-unix/bin/python tests/smoke.py              # Linux / macOS
```

`tests/test_hostos.py` 用固定樣本驗證 netstat / ss / lsof / tasklist / pgrep / sysfs /
`ps` 的解析，所以在 Windows 上也能測 Linux 與 macOS 的分支。

---

原始碼 / Source: https://github.com/iskshadow195563/herta-station
