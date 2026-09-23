import sys
import os
import io
import gc  # 匯入垃圾回收機制
import torch
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pitcher_arsenal_extractor import extract_pitcher_arsenal, get_pitch_physics


# 🌟 救命仙丹：強制 PyTorch 只能用單一執行緒，防止 0.1 vCPU 卡死與記憶體暴增
torch.set_num_threads(1)


# 強制 UTF-8 輸出，避免 Windows 終端機顯示 Emoji 報錯
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 取得上一層目錄的絕對路徑，確保能載入模型與外部 Python 檔
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

app = Flask(__name__)

# 🌟 補回 Render 反向代理修復，確保限流功能可以抓到真實訪客 IP
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
# 🛡️ 防線一：正確的 CORS 白名單
# 注意：Origin 只能是「通訊協定 + 網域 + Port」，不能有後面的路徑！
# 加入了 5500 port 讓您的 VS Code Live Server 可以順利連線

# 🌟 讀取系統環境變數。如果雲端沒有設定，就預設給本地端的這些網址 (方便您開發)
origins_env = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5501,http://localhost:5501,http://127.0.0.1:5000,http://localhost:5000"
)

# 將逗號分隔的字串，自動切成陣列
ALLOWED_ORIGINS = [origin.strip() for origin in origins_env.split(",")]

CORS(app, resources={
    r"/api/*": {"origins": ALLOWED_ORIGINS}
})

# 🛡️ 防線二：IP 速率限制 (Rate Limiting)
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=["500 per day", "30 per minute"],
    # 🌟 必須加入這行：放行瀏覽器的 CORS 預檢請求 (OPTIONS)
    default_limits_exempt_when=lambda: request.method == 'OPTIONS' 
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="rate limit exceeded", message=str(e.description)), 429

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseballTransformerSingleTask(nn.Module):
    def __init__(self, seq_feature_dim=16, ctx_feature_dim=11, d_model=32, n_heads=4, num_layers=2):
        super(BaseballTransformerSingleTask, self).__init__()
        self.input_projection = nn.Linear(seq_feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_flatten = nn.Linear(d_model, 16)
        self.ctx_dense = nn.Linear(ctx_feature_dim, 16)
        self.fc = nn.Sequential(
            nn.Linear(16 + 16, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    def forward(self, x_seq, x_ctx):
        x_seq = self.input_projection(x_seq) + self.pos_embedding
        enc_out = self.transformer_encoder(x_seq)
        seq_repr = torch.relu(self.seq_flatten(enc_out[:, -1, :]))
        ctx_repr = torch.relu(self.ctx_dense(x_ctx))
        return self.fc(torch.cat((seq_repr, ctx_repr), dim=1))

STRIKE_ZONES = {
    '左上': (-0.37, 0.47),  '中上': (0.0, 0.47),  '右上': (0.37, 0.47),
    '左中': (-0.37, 0.0),   '正中': (0.0, 0.0),   '右中': (0.37, 0.0),
    '左下': (-0.37, -0.47), '中下': (0.0, -0.47), '右下': (0.37, -0.47)
}

# 🌟 調整這裡的壞球/引誘區座標，將它們往好球帶（靠近 0.0）收斂
BALL_ZONES = {
    '壞_左上': (-0.65, 0.85),  '壞_中上': (0.0, 0.85),  '壞_右上': (0.65, 0.85),
    '壞_左中': (-0.65, 0.0),                         '壞_右中': (0.65, 0.0),
    '壞_左下': (-0.65, -0.85), '壞_中下': (0.0, -0.85), '壞_右下': (0.65, -0.85),
    '壞_挖地瓜': (0.0, -1.1)  # 原本是 -1.8，若覺得太夸張可以往上修到 -1.3 或 -1.4
}
ALL_ZONES = {**STRIKE_ZONES, **BALL_ZONES}

def get_smart_physics(full_df, p_type, target_zone, arsenal_data):
    samples = full_df[(full_df['pitch_type'] == p_type) & (full_df['zone_name'] == target_zone)]
    if len(samples) >= 3:
        return {
            'speed': float(samples['release_speed'].mean()) / 100.0,
            'pfx_x': float(samples['pfx_x'].mean()),
            'pfx_z': float(samples['pfx_z'].mean())
        }
    fallback_zone = target_zone.replace('壞_', '')
    if fallback_zone == '挖地瓜': fallback_zone = '中下'
    try:
        return get_pitch_physics(arsenal_data, p_type, fallback_zone)
    except:
        pass
    return arsenal_data['overall_arsenal'][p_type]

def get_zone_name(px, pz):
    best_zone, min_dist = '正中', float('inf')
    for z_name, (cx, cz) in ALL_ZONES.items():
        dist = (px - cx)**2 + (pz - cz)**2
        if dist < min_dist:
            min_dist, best_zone = dist, z_name
    return best_zone

print("正在載入 2026 年實戰數據庫 (精簡版)...")
ALL_ARSENALS = {} 
try:
    csv_path = os.path.join(BASE_DIR, 'mlb_pitch_data_2026_strictly_filtered_test.csv.gz')
    
    # 🌟 1. 欄位瘦身：只讀取模型跟前端畫圖絕對需要的欄位，省下 70% 記憶體
    # 🌟 在 use_cols 裡面補上 'balls' 與 'strikes'
    use_cols = [
        'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'pitcher_name', 
        'p_throws', 'pitch_type', 'release_speed', 'plate_x', 'plate_z_norm', 
        'pfx_x', 'pfx_z', 'description', 'stand', 'batter_name', 'batter', 'launch_speed',
        'balls', 'strikes', 'game_date'  # 👈 補上這行，把日期找回來！
    ]
    
    df_2026 = pd.read_csv(csv_path, compression='gzip', usecols=use_cols)
    
    # 🌟 2. 完整收錄您指定的球星群（同時支援英文名字模糊搜尋與固定 ID 雙重保險）
    # 包含：山本由伸、大谷翔平、佐佐木朗希、今永昇太、千賀滉大、菊池雄星、Skubal
    target_names = ['Yamamoto', 'Ohtani', 'Sasaki', 'Imanaga', 'Senga', 'Kikuchi', 'Skubal']
    name_pattern = '|'.join(target_names)
    
    target_ids = [808967] # 確保山本由伸的 ID 絕對不會漏接，亦可在此加入其他球星 ID
    
    # 雙重篩選：只要符合名字關鍵字 或 符合 ID 的通通留下，其餘直接砍掉以省下 90% 記憶體
    df_2026 = df_2026[
        df_2026['pitcher_name'].str.contains(name_pattern, case=False, na=False) | 
        df_2026['pitcher'].isin(target_ids)
    ].copy()

    df_2026['zone_name'] = df_2026.apply(lambda x: get_zone_name(float(x['plate_x']), float(x['plate_z_norm'])), axis=1)
    
    print("正在建構指定球星專屬軍火庫...")
    pitcher_info = df_2026[['pitcher', 'pitcher_name', 'p_throws']].drop_duplicates(subset=['pitcher'])
    for _, row in pitcher_info.iterrows():
        p_id = int(row['pitcher'])
        p_name = "Y. Yamamoto" if p_id == 808967 else str(row['pitcher_name'])
        p_throws = str(row['p_throws'])
        
        try:
            arsenal = extract_pitcher_arsenal(df_2026, p_id, min_pitch_count=10)
            if len(arsenal['valid_pitches']) > 0:
                ALL_ARSENALS[p_id] = {
                    "name": p_name,
                    "p_throws": p_throws,
                    "arsenal_data": arsenal,
                    "valid_pitches": arsenal['valid_pitches']
                }
        except Exception:
            pass
    print(f"✅ 成功建構 {len(ALL_ARSENALS)} 位球星的專屬軍火庫！")
    pitcher_info = df_2026[['pitcher', 'pitcher_name', 'p_throws']].drop_duplicates(subset=['pitcher'])
    for _, row in pitcher_info.iterrows():
        p_id = int(row['pitcher'])
        p_name = str(row['pitcher_name'])
        p_throws = str(row['p_throws'])
        
        # 客製化山本由伸的名字
        if p_id == 808967:
            p_name = "Y. Yamamoto"
            
        try:
            arsenal = extract_pitcher_arsenal(df_2026, p_id, min_pitch_count=10) # 門檻調低一點，確保球星少數球種也能抓到
            if len(arsenal['valid_pitches']) > 0:
                ALL_ARSENALS[p_id] = {
                    "name": p_name,
                    "p_throws": p_throws,
                    "arsenal_data": arsenal,
                    "valid_pitches": arsenal['valid_pitches']
                }
        except Exception:
            pass
    print(f"✅ 成功建構 {len(ALL_ARSENALS)} 位頂級球星的專屬軍火庫！")
except FileNotFoundError:
    print("找不到 2026 年的 CSV 檔案。")
    df_2026 = pd.DataFrame()

# ... (下方模型載入 for matchup in matchups: ... 保持不變)


@app.route('/api/pitchers', methods=['GET'])
def get_pitchers():
    response = {}
    for p_id, data in ALL_ARSENALS.items():
        arsenal_list = []
        for pt in data['valid_pitches']:
            spd = data['arsenal_data']['overall_arsenal'][pt]['speed'] * 100
            arsenal_list.append({"value": pt, "label": f"{pt} ({spd:.1f} mph)"})
        response[str(p_id)] = {
            "name": data['name'],
            "p_throws": data['p_throws'],
            "arsenal": arsenal_list
        }
    return jsonify(response)

@app.route('/api/atbats/<pitcher_id>', methods=['GET'])
def get_atbats(pitcher_id):
    if df_2026.empty: return jsonify([])
    p_df = df_2026[df_2026['pitcher'] == int(pitcher_id)]
    grouped = p_df.groupby(['game_pk', 'at_bat_number'])
    
    atbats_list = []
    for (g_pk, ab_num), group in grouped:
        # 🌟 關鍵修改：過濾掉只有 1 顆球的打席 (一球死)
        if len(group) < 2:
            continue
            
        group = group.sort_values('pitch_number')
        first_pitch = group.iloc[0]

        stand = first_pitch.get('stand', 'R')
        
        # 🌟 解鎖真實姓名：優先讀取 CSV 新增的 batter_name 欄位
        if 'batter_name' in first_pitch and pd.notna(first_pitch['batter_name']):
            batter_name = first_pitch['batter_name']
        else:
            batter_name = f"Batter {first_pitch.get('batter', 'Unknown')}"
            
        label = f"{first_pitch.get('game_date', '2026')} vs {batter_name} ({'左' if stand=='L' else '右'}打)"
        seq = []
        for _, p in group.iterrows():
            px, pz = float(p.get('plate_x', 0)), float(p.get('plate_z_norm', 0))
            is_strike = p['description'] in ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']
            
            # 🌟 新增：安全獲取擊球初速 (Launch Speed)
            ls_val = p.get('launch_speed')
            ls_str = f"{float(ls_val):.1f} mph" if pd.notna(ls_val) and str(ls_val).strip() != "" else ""

            seq.append({
                "num": int(p['pitch_number']), "type": str(p['pitch_type']), "name": str(p['pitch_type']), 
                "speed": f"{float(p['release_speed']):.1f} mph", "result": str(p['description']),
                "isStrike": is_strike, "count": f"{int(p['balls'])} - {int(p['strikes'])}",
                "x": max(0, min(100, ((px + 1.2) / 2.4) * 100)), "y": max(0, min(100, 100 - (((pz + 1.0) / 2.0) * 100))),
                "zone_name": str(p.get('zone_name', '正中')),
                "raw_speed": float(p.get('release_speed', 95.0)), "raw_px": px, "raw_pz": pz,
                "raw_pfx_x": float(p.get('pfx_x', 0.0)), "raw_pfx_z": float(p.get('pfx_z', 0.0)),
                "launch_speed": ls_str  # 🌟 將初速打包傳給前端
            })
        atbats_list.append({"ab_id": f"{g_pk}_{ab_num}", "label": label, "stand": stand, "sequence": seq})
    
    atbats_list.sort(key=lambda x: x['label'])
    return jsonify(atbats_list)

@app.route('/api/simulate', methods=['POST'])
def simulate_counterfactual():
    data = request.json
    sim_mode = data.get('sim_mode', 'last')
    p_type, p_zone, stand = data['pitch_type'], data['pitch_zone'], data.get('stand', 'R')
    pitcher_id = int(data.get('pitcher_id', 808967))
    
    # 抓取該投手專屬資料
    p_data = ALL_ARSENALS.get(pitcher_id, {})
    p_throws = p_data.get('p_throws', 'R')
    p_arsenal = p_data.get('arsenal_data')
    
    matchup_key = f"{p_throws}HP vs {stand}HB" # 🌟 動態判斷左右投 vs 左右打
    model_w = BaseballTransformerSingleTask().to(device)
    model_h = BaseballTransformerSingleTask().to(device)
    csw_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_CSW_{matchup_key}_best.pth')
    hh_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_Hardhit_{matchup_key}_best.pth')

    if os.path.exists(csw_path) and os.path.exists(hh_path):
        model_w.load_state_dict(torch.load(csw_path, map_location=device, weights_only=True))
        model_h.load_state_dict(torch.load(hh_path, map_location=device, weights_only=True))
        model_w.eval()
        model_h.eval()
    else:
        return jsonify({"error": f"找不到對戰模型 {matchup_key}"}), 500
    

    # 🌟 隔離物理引擎：只用該投手的資料去算平均位移
    p_df = df_2026[df_2026['pitcher'] == pitcher_id] if not df_2026.empty else pd.DataFrame()
    physics = get_smart_physics(p_df, p_type, p_zone, p_arsenal) if p_arsenal else {'speed': 0.95, 'pfx_x': 0.0, 'pfx_z': 0.0}
    px_new, pz_new = ALL_ZONES.get(p_zone, (0.0, 0.0))

    # ==========================================
    # 🌟 擷取完整的歷史真實特徵 (Original)
    # ==========================================
    # 1. N-2 (倒數第三球)：用來推算 N-1 的位移落差
    n2 = data.get('n2_pitch', {})
    sn2, pxn2, pzn2, pfx_xn2, pfx_zn2 = float(n2.get('speed', 0.95)), float(n2.get('px', 0.0)), float(n2.get('pz', 0.0)), float(n2.get('pfx_x', 0.0)), float(n2.get('pfx_z', 0.0))
    
    # 2. N-1 (原本的佈局球)
    n1 = data.get('n1_pitch', {})
    sn1, pxn1, pzn1, pfx_xn1, pfx_zn1 = float(n1.get('speed', 0.95)), float(n1.get('px', 0.0)), float(n1.get('pz', 0.0)), float(n1.get('pfx_x', 0.0)), float(n1.get('pfx_z', 0.0))
    n1_outcome = data.get('n1_outcome', [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) # 預設為揮空
    
    # 3. N (原本的決戰球)
    orig = data.get('orig_pitch', {})
    s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig = float(orig.get('speed', 0.95)), float(orig.get('px', 0.0)), float(orig.get('pz', 0.0)), float(orig.get('pfx_x', 0.0)), float(orig.get('pfx_z', 0.0))

    # ==========================================
    # 🌟 基礎對照組 (Original Sequence)
    # ==========================================
    pitch_n1_orig = [sn1, pxn1, pzn1, pfx_xn1, pfx_zn1] + n1_outcome + [sn1 - sn2, pxn1 - pxn2, pzn1 - pzn2, pfx_xn1 - pfx_xn2, pfx_zn1 - pfx_zn2]
    pitch_n_orig = [s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig, 0,0,0,0,0,0, s_orig - sn1, px_orig - pxn1, pz_orig - pzn1, pfx_x_orig - pfx_xn1, pfx_z_orig - pfx_zn1]

    # ==========================================
    # 🌟 反事實變異組 (Counterfactual Sequence)
    # ==========================================
    if sim_mode == 'last':
        # 模式一【改變 N】：N-1 維持真實，N 獲得新位置，重新計算 N 與 真實 N-1 的 Delta
        pitch_n1_cf = pitch_n1_orig
        pitch_n_cf = [
            physics['speed'], px_new, pz_new, physics['pfx_x'], physics['pfx_z'], 
            0,0,0,0,0,0, 
            physics['speed'] - sn1, px_new - pxn1, pz_new - pzn1, physics['pfx_x'] - pfx_xn1, physics['pfx_z'] - pfx_zn1
        ]
    else:
        # 模式二【改變 N-1】：N-1 獲得新位置並與 N-2 算 Delta；N 保持原本實戰的絕對位置，但受到「新 N-1」牽連，產生了全新的 Delta 視線落差！
        pitch_n1_cf = [
            physics['speed'], px_new, pz_new, physics['pfx_x'], physics['pfx_z']
        ] + n1_outcome + [
            physics['speed'] - sn2, px_new - pxn2, pz_new - pzn2, physics['pfx_x'] - pfx_xn2, physics['pfx_z'] - pfx_zn2
        ]
        pitch_n_cf = [
            s_orig, px_orig, pz_orig, pfx_x_orig, pfx_z_orig, 
            0,0,0,0,0,0, 
            s_orig - physics['speed'], px_orig - px_new, pz_orig - pz_new, pfx_x_orig - physics['pfx_x'], pfx_z_orig - physics['pfx_z']
        ]

    # 推論
    ctx = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    t_seq_cf = torch.tensor(np.array([[[0.0]*16]*4 + [pitch_n1_cf, pitch_n_cf]]), dtype=torch.float32).to(device)
    t_seq_orig = torch.tensor(np.array([[[0.0]*16]*4 + [pitch_n1_orig, pitch_n_orig]]), dtype=torch.float32).to(device)
    t_ctx = torch.tensor(np.array([ctx]), dtype=torch.float32).to(device)

    with torch.no_grad():
        prob_w_cf, prob_h_cf = float(torch.sigmoid(model_w(t_seq_cf, t_ctx))), float(torch.sigmoid(model_h(t_seq_cf, t_ctx)))
        prob_w_orig, prob_h_orig = float(torch.sigmoid(model_w(t_seq_orig, t_ctx))), float(torch.sigmoid(model_h(t_seq_orig, t_ctx)))
    
    del model_w
    del model_h
    gc.collect()

    return jsonify({
        "matchup_used": matchup_key, "sim_mode": sim_mode,
        "csw_prob": round(prob_w_cf * 100, 1), "hh_prob": round(prob_h_cf * 100, 1),
        "orig_csw_prob": round(prob_w_orig * 100, 1), "orig_hh_prob": round(prob_h_orig * 100, 1)
    })

@app.route('/api/simulate_sequence', methods=['POST'])
@limiter.limit("10 per minute")  # 針對這個高耗能 API，限制同一個 IP 一分鐘只能算 10 次
def simulate_sequence():
    """動態多球對決：接收任意長度 (1~6球) 的配球陣列，逐球計算期望值"""
    data = request.json
    pitcher_id = int(data.get('pitcher_id', 808967))
    stand = data.get('stand', 'R')
    pitches = data.get('pitches', []) # 接收 [{'type': 'FF', 'zone': '正中'}, ...]
    
    p_data = ALL_ARSENALS.get(pitcher_id, {})
    p_throws = p_data.get('p_throws', 'R')
    p_arsenal = p_data.get('arsenal_data')
    
    matchup_key = f"{p_throws}HP vs {stand}HB"

    model_w = BaseballTransformerSingleTask().to(device)
    model_h = BaseballTransformerSingleTask().to(device)
    csw_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_CSW_{matchup_key}_best.pth')
    hh_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_Hardhit_{matchup_key}_best.pth')

    if os.path.exists(csw_path) and os.path.exists(hh_path):
        model_w.load_state_dict(torch.load(csw_path, map_location=device, weights_only=True))
        model_h.load_state_dict(torch.load(hh_path, map_location=device, weights_only=True))
        model_w.eval()
        model_h.eval()
    else:
        return jsonify({"error": f"找不到對戰模型 {matchup_key}"}), 500
    
    p_df = df_2026[df_2026['pitcher'] == pitcher_id] if not df_2026.empty else pd.DataFrame()
    
    results = []
    history_features = [] # 儲存推演過程的歷史特徵
    
    # 🌟 核心：無論前端傳幾顆球，用迴圈逐顆推演 Delta 與期望值
    for i, p_info in enumerate(pitches):
        p_type = p_info['type']
        p_zone = p_info['zone']
        
        phys = get_smart_physics(p_df, p_type, p_zone, p_arsenal) if p_arsenal else {'speed': 0.95, 'pfx_x': 0.0, 'pfx_z': 0.0}
        px, pz = ALL_ZONES.get(p_zone, (0.0, 0.0))
        
        # 計算與前一球的 Delta (第一球無 Delta)
        if i == 0:
            d_s, d_px, d_pz, d_pfx, d_pfz = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            prev_phys = history_features[-1]['phys']
            prev_px, prev_pz = history_features[-1]['px'], history_features[-1]['pz']
            d_s = phys['speed'] - prev_phys['speed']
            d_px, d_pz = px - prev_px, pz - prev_pz
            d_pfx, d_pfz = phys['pfx_x'] - prev_phys['pfx_x'], phys['pfx_z'] - prev_phys['pfx_z']
            
        # 組裝該球的 16 維特徵 (將打擊結果預設為 0 以維持中立)
        pitch_vec = [
            phys['speed'], px, pz, phys['pfx_x'], phys['pfx_z'],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            d_s, d_px, d_pz, d_pfx, d_pfz
        ]
        
        history_features.append({'phys': phys, 'px': px, 'pz': pz, 'vec': pitch_vec})
        
        # 準備送入 Transformer (最高支援 6 球，往前填充 0)
        current_seq = [f['vec'] for f in history_features[-6:]]
        padded_seq = [[0.0]*16] * (6 - len(current_seq)) + current_seq
        
        # 設定打者左右手 Context
        stand_val = 1.0 if stand == 'L' else 0.0
        ctx = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, stand_val, 1.0, 0.0, 0.0]
        
        t_seq = torch.tensor(np.array([padded_seq]), dtype=torch.float32).to(device)
        t_ctx = torch.tensor(np.array([ctx]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prob_w = float(torch.sigmoid(model_w(t_seq, t_ctx)).cpu().numpy().flatten()[0])
            prob_h = float(torch.sigmoid(model_h(t_seq, t_ctx)).cpu().numpy().flatten()[0])
            
        results.append({
            "pitch_num": i + 1,
            "type": p_type,
            "zone": p_zone,
            "csw_prob": round(prob_w * 100, 1),
            "hh_prob": round(prob_h * 100, 1)
        })
        
    # 🌟 修改這裡：算完立刻砍掉模型
    del model_w
    del model_h
    gc.collect()

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)