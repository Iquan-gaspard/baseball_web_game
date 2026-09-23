# ⚾ Pitching Counterfactual Lab | AI 棒球反事實配球實驗室

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-ee4c2c?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FJS-e34f26?logo=html5&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

透過 PyTorch 深度學習模型與 MLB 真實實戰數據，解構頂級投手（如山本由伸）的配球邏輯。本系統不僅能重現歷史打席，更首創「反事實實驗室」，讓使用者能自由改變配球軌跡，量化「視線落差（Delta）」對打者揮空的影響。

## ✨ 核心亮點 (Key Features)

- 🧠 **雙引擎 Transformer 預測模型**
  - 基於 2026 年大聯盟投球數據訓練。
  - **CSW 引擎**：預測引誘打者出棒揮空或凍結好球的機率。
  - **No Hard-Hit 引擎**：預測成功壓制擊球初速，避免被擊出強勁擊球的防護率。
- 🔬 **反事實實驗室 (Counterfactual Lab)**
  - 支援修改決戰球 (第 N 球) 或佈局球 (第 N-1 球)。
  - 即時運算新球路與歷史軌跡產生的「視線干擾 Delta 特徵」，揭示配球策略的真實期望值變化。
- 🔄 **動態連續配球腳本 (Sequence Builder)**
  - 支援 1~7 球的自訂配球序列推演。
  - 考量打者左右手特性與每一球的進壘點（18 宮格），逐球結算壓制機率。
- 📱 **跨裝置響應式體驗 (RWD)**
  - 純淨的 Vanilla JS + CSS Flexbox 打造，針對手機版（iOS/Android）優化自訂九宮格與資料視覺化介面。

## 🛠️ 企業級全端與部署優化 (Production-Ready & DevOps)

本專案不僅專注於 AI 模型，更導入了多項業界標準的全端開發實踐，確保系統穩定性與使用者體驗：

1.  **無縫冷啟動對策 (Zero Cold-Start Latency)**
    - 針對雲端平台 (Render) 免費方案的休眠特性，整合外部 cron-job.org 進行排程保活 (Keep-alive)，徹底消除使用者初次造訪時長達 45 秒的冷啟動延遲。
2.  **環境變數與資安隔離 (Environment Variables Isolation)**
    - 將 CORS 白名單與系統設定抽離原始碼，本地開發透過 `.env` 讀取，正式上線則依賴雲端控制台動態注入，達成 100% 的組態與邏輯分離，防範資安外洩。
3.  **優雅的錯誤捕捉與限流防護 (Graceful Error Handling & Rate Limiting)**
    - 後端實作 `Flask-Limiter` 與 `ProxyFix` (精準解析代理伺服器後的真實 IP)，阻擋惡意算力消耗 (限制 30 requests / min)。
    - 前端實作完善的錯誤攔截 (Error Interceptor)，當觸發 HTTP 429 限流或 500 異常時，以友善的中文 UI 提示框取代生硬的 Console 報錯，保障前端體驗。

## 🏗️ 系統架構 (Architecture)

- **前端 (Frontend)**: HTML5, CSS3, JavaScript (Fetch API), Tom Select (UI 元件)
- **後端 (Backend)**: Python, Flask, Pandas, Flask-CORS, Flask-Limiter, python-dotenv
- **人工智慧 (AI/ML)**: PyTorch (TransformerEncoderLayer, 自定義神經網路架構)
- **部署 (Deployment)**:
  - Frontend 託管於 **GitHub Pages**
  - Backend API 託管於 **Render**

## 🧪 模型訓練與數據工程 (ML Training Pipeline)

本專案開源了完整的模型訓練軌跡，展現從原始數據清洗到神經網路建構的 MLOps 流程。開發者可透過專案內的 Jupyter Notebook 深入了解底層演算法細節：

- **📂 訓練腳本 (`train_model.ipynb`)**
  - **數據預處理 (Data Preprocessing)**：過濾並清洗 MLB 2026 實戰 CSV 數據，處理缺失值，並精準計算相鄰兩球之間的速度與軌跡位移差 (Delta Features)。
  - **特徵工程 (Feature Engineering)**：將投球序列轉換為高維度張量 (Tensor)，並融合打者左右手、球數等靜態上下文特徵 (Context Features)。
  - **模型架構 (Model Architecture)**：使用 PyTorch 從零建構 `BaseballTransformerSingleTask` 類別，實作 Transformer Encoder 層以捕捉配球的時序依賴性。
  - **訓練與評估 (Training & Evaluation)**：包含自訂的 Loss 函數計算、梯度下降最佳化、以及訓練過程的 Loss 曲線視覺化。最終將驗證集表現最佳的模型權重匯出為 `.pth` 檔，供 Flask API 載入預測。

> **💡 想自行訓練模型？**
> 在本地端啟動 Jupyter Notebook，依序執行儲存格即可重現訓練過程，或替換成您自己感興趣的球星數據庫（如大谷翔平、佐佐木朗希）來打造專屬預測模型。

## 🚀 本地端開發設定 (Local Development Setup)

### 1. 取得程式碼

```bash
git clone [https://github.com/YOUR_USERNAME/baseball_web_game.git](https://github.com/YOUR_USERNAME/baseball_web_game.git)
cd baseball_web_game

cd server
python -m venv .venv
source .venv/bin/activate  # Windows 請使用 .venv\Scripts\activate
pip install -r requirements.txt

python app.py
# 伺服器將運行於 [http://127.0.0.1:5000](http://127.0.0.1:5000)
```
