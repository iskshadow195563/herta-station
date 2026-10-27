# 在 Ubuntu / Linux 上執行 Herta Station

同一份原始碼在 Windows、Linux、macOS 上都能跑：控制台是純標準庫 Python，
GUI 用 pywebview（Linux 走 Qt WebEngine 或 WebKitGTK），沒有 webview 時自動退回瀏覽器。

---

## 1. 先裝 llama.cpp

Herta Station 只是 llama-server 的儀表板，模型推理由 llama.cpp 執行。

```bash
# 方式 A：發行版套件（版本較舊，但最快能用）
sudo apt install llama.cpp                 # 提供 llama-server

# 方式 B：官方預編譯包（推薦）
#   到 https://github.com/ggml-org/llama.cpp/releases 下載
#   llama-bXXXX-bin-ubuntu-x64.zip（NVIDIA 用 cuda 版：llama-bXXXX-bin-ubuntu-cuda-x64.zip）
mkdir -p ~/llama.cpp && cd ~/llama.cpp
unzip ~/Downloads/llama-bXXXX-bin-ubuntu-cuda-x64.zip
ls build/bin/llama-server

# 方式 C：自己編（CUDA 13 / sm_120 這類新卡通常要自己編）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j $(nproc)
```

驗證：

```bash
llama-server --version
```

## 2. 放模型

```bash
mkdir -p ~/llama.cpp/models
# 把 *.gguf 放進去；多層目錄也可以，程式會遞迴掃描
```

預設掃描位置依序是：
`$HERTA_MODELS_DIR` → `~/llama.cpp/models`（若存在）→ `~/.local/share/HertaStation/models`。

## 3. 安裝與執行

```bash
git clone <你的 repo> ~/herta-station && cd ~/herta-station

# 直接從原始碼跑（會用 .venv-unix，沒有就用系統 python3）
./start.sh --check          # 印出偵測到的 GPU / 路徑
./start.sh                  # 開視窗；沒有 webview 就開瀏覽器

# 或打包成單一執行檔
./build.sh --qt             # 連 Qt WebEngine 一起打包（不需系統 GTK）
./build.sh --headless       # 不帶 webview，檔案小，用瀏覽器操作
./packaging/install-linux.sh        # 裝到 ~/.local，含桌面捷徑
sudo ./packaging/install-linux.sh --system   # 裝到 /usr/local
```

啟動後：

* 控制台 UI：<http://127.0.0.1:8899>
* OpenAI 相容 API：`http://0.0.0.0:11433/v1`
* Anthropic 相容端點：`http://0.0.0.0:11435`

### 沒有 Linux 機器？用 Docker 打包

```bash
docker build -f packaging/Dockerfile -t herta-build .
docker run --rm -v "$PWD/dist:/out" herta-build     # 產出 dist/HertaStation
```

映像基底是 Ubuntu 22.04（glibc 2.35），產物可直接在 22.04 以上的發行版執行；
Qt WebEngine 已內含，目標機不需要 GTK / WebKitGTK。

## 4. GUI 後端要怎麼選

| 情況 | 做法 |
| --- | --- |
| 想要原生視窗、不想裝系統套件 | `./build.sh --qt`（pip 安裝 QtWebEngine wheels） |
| 系統已有 GTK / WebKitGTK | `sudo apt install python3-gi gir1.2-webkit2-4.1`，用非 venv 的 python 跑 `start.sh` |
| 純伺服器 / 遠端機器 | `./dist/HertaStation --serve`，用 SSH 通道或反向代理看 8899 |
| 只要最省事 | `./dist/HertaStation --browser`，開系統瀏覽器 |

`--serve` 只綁 127.0.0.1:8899，不會對外開放；要遠端看就用
`ssh -L 8899:127.0.0.1:8899 user@host`。

## 5. 環境變數

| 變數 | 用途 | 預設 |
| --- | --- | --- |
| `HERTA_LLAMA_EXE` | llama-server 路徑 | 自動搜尋 PATH、`/opt/llama.cpp/build/bin`、`~/llama.cpp/build/bin` |
| `HERTA_MODELS_DIR` | 模型庫目錄 | `~/llama.cpp/models` 或 `~/.local/share/HertaStation/models` |
| `HERTA_DATA_DIR` | 設定 / 日誌 / PID | `~/.local/share/HertaStation` |
| `HERTA_PORT` | 模型 API 埠 | `11433` |
| `HERTA_BIND` | 模型 API 綁定位址 | `0.0.0.0` |
| `HERTA_API_KEY` | 啟用 Bearer 驗證 | 空（不驗證） |
| `HERTA_CTX` | 上下文長度 | 由自動配置決定 |
| `HERTA_UI_PORT` | 控制台埠 | `8899` |

`HERTA_CTX` 之外的多數參數也可以在 UI 的「設定」裡改，存到
`~/.local/share/HertaStation/settings.json`。

## 6. 開機自動啟動（可選）

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/herta-station.service <<'EOF'
[Unit]
Description=Herta Station (local llama.cpp switchboard)
After=graphical-session.target

[Service]
ExecStart=%h/.local/bin/HertaStation --serve
Restart=on-failure
Environment=HERTA_LLAMA_EXE=%h/llama.cpp/build/bin/llama-server

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now herta-station
```

## 7. 防火牆

只有模型 API 需要對外（給同網段的 agent / 手機用）：

```bash
sudo ufw allow 11433/tcp        # 或只開給內網：sudo ufw allow from 192.168.0.0/16 to any port 11433
```

控制台 8899 與 Anthropic 11435 建議維持只在本機。

## 8. 常見問題

| 症狀 | 原因與處理 |
| --- | --- |
| 啟動就退回瀏覽器 | 沒有 webview 後端：`./build.sh --qt`，或裝 `python3-gi gir1.2-webkit2-4.1` |
| `找不到 llama-server` | 設 `HERTA_LLAMA_EXE=/實際路徑/llama-server`，或 `sudo ln -s ... /usr/local/bin/llama-server` |
| 模型庫是空的 | 確認 `HERTA_MODELS_DIR` 指到放 `.gguf` 的目錄；UI 模型庫標題會顯示目前路徑 |
| 顯示 `n/a` 的顯存 | 沒裝 `nvidia-smi`（NVIDIA 專有驅動）或沒有 `mem_info_vram_total`（AMD）；純 CPU 執行時屬正常 |
| 速度突然掉十倍 | 上下文超過顯存，驅動退回系統記憶體：到設定降低上下文，或換更小的量化 |
| `Permission denied` 寫設定 | `.app`/`/opt` 目錄不可寫；改用 `HERTA_DATA_DIR=~/.local/share/HertaStation` |
| 埠被佔用 | `ss -ltnp \| grep 11433` 找出佔用者；或用設定改成別的埠 |

## 9. 測試

```bash
.venv-unix/bin/python tests/smoke.py       # 全部（含平台層）
.venv-unix/bin/python tests/test_hostos.py # 只跑 netstat/ss/lsof/sysfs/Metal 解析
```
