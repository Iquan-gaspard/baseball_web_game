import sys
import os
import io


# 強制 UTF-8 輸出，避免 Windows 終端機顯示 Emoji 報錯
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 取得上一層目錄的絕對路徑，確保能載入模型與外部 Python 檔
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pitcher_arsenal_extractor import extract_pitcher_arsenal, get_pitch_physics

app = Flask(__name__)
CORS(app)
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
    '左上': (-0.55, 0.7),  '中上': (0.0, 0.7),  '右上': (0.55, 0.7),
    '左中': (-0.55, 0.0),  '正中': (0.0, 0.0),  '右中': (0.55, 0.0),
    '左下': (-0.55, -0.7), '中下': (0.0, -0.7), '右下': (0.55, -0.7)
}
BALL_ZONES = {
    '壞_左上': (-1.2, 1.2),  '壞_中上': (0.0, 1.2),  '壞_右上': (1.2, 1.2),
    '壞_左中': (-1.2, 0.0),                         '壞_右中': (1.2, 0.0),
    '壞_左下': (-1.2, -1.2), '壞_中下': (0.0, -1.2), '壞_右下': (1.2, -1.2),
    '壞_挖地瓜': (0.0, -1.8)  
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

print("正在載入 2026 年實戰數據庫...")
ALL_ARSENALS = {} # 🌟 建立全域軍火庫字典
try:
    csv_path = os.path.join(BASE_DIR, 'mlb_pitch_data_2026_strictly_filtered_test.csv.gz')
    df_2026 = pd.read_csv(csv_path, compression='gzip')
    df_2026['zone_name'] = df_2026.apply(lambda x: get_zone_name(float(x['plate_x']), float(x['plate_z_norm'])), axis=1)
    
    print("正在建構全聯盟投手軍火庫 (這可能需要幾秒鐘)...")
    pitcher_info = df_2026[['pitcher', 'pitcher_name', 'p_throws']].drop_duplicates(subset=['pitcher'])
    for _, row in pitcher_info.iterrows():
        p_id = int(row['pitcher'])
        p_name = str(row['pitcher_name'])
        p_throws = str(row['p_throws'])
        
        # 🌟 客製化山本由伸的名字：去掉漢字，只留拼音
        if p_id == 808967:
            p_name = "Y. Yamamoto"
        elif p_name == "Unknown Pitcher":
            p_name = f"Pitcher {p_id}"
            
        try:
            arsenal = extract_pitcher_arsenal(df_2026, p_id, min_pitch_count=30)
            if len(arsenal['valid_pitches']) > 0:
                ALL_ARSENALS[p_id] = {
                    "name": p_name,
                    "p_throws": p_throws,
                    "arsenal_data": arsenal,
                    "valid_pitches": arsenal['valid_pitches']
                }
        except Exception:
            pass
    print(f"✅ 成功建構 {len(ALL_ARSENALS)} 位投手的專屬軍火庫！")
except FileNotFoundError:
    print("找不到 2026 年的 CSV 檔案。")
    df_2026 = pd.DataFrame()

# ... (下方模型載入 for matchup in matchups: ... 保持不變)

print("正在載入 PyTorch 雙引擎 8 大對戰模型...")
models = {'CSW': {}, 'Hardhit': {}}
matchups = ['LHP vs LHB', 'LHP vs RHB', 'RHP vs LHB', 'RHP vs RHB']

for matchup in matchups:
    csw_path = os.path.join(BASE_DIR, f'baseball_transformer_weights_CSW_{matchup}_best.pth')
    hh_path  = os.path.join(BASE_DIR, f'baseball_transformer_weights_Hardhit_{matchup}_best.pth')
    if os.path.exists(csw_path) and os.path.exists(hh_path):
        model_w = BaseballTransformerSingleTask().to(device)
        model_h = BaseballTransformerSingleTask().to(device)
        model_w.load_state_dict(torch.load(csw_path, map_location=device, weights_only=True))
        model_h.load_state_dict(torch.load(hh_path, map_location=device, weights_only=True))
        model_w.eval(); model_h.eval()
        models['CSW'][matchup] = model_w
        models['Hardhit'][matchup] = model_h
        print(f"✅ 成功載入: {matchup}")

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
    model_w = models['CSW'].get(matchup_key)
    model_h = models['Hardhit'].get(matchup_key)
    if not model_w or not model_h:
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

    return jsonify({
        "matchup_used": matchup_key, "sim_mode": sim_mode,
        "csw_prob": round(prob_w_cf * 100, 1), "hh_prob": round(prob_h_cf * 100, 1),
        "orig_csw_prob": round(prob_w_orig * 100, 1), "orig_hh_prob": round(prob_h_orig * 100, 1)
    })

@app.route('/api/simulate_sequence', methods=['POST'])
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
    model_w = models['CSW'].get(matchup_key)
    model_h = models['Hardhit'].get(matchup_key)
    
    if not model_w or not model_h:
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
        
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)