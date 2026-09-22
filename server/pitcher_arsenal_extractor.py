import pandas as pd
import numpy as np

def extract_pitcher_arsenal(df_raw, pitcher_id, min_pitch_count=5):
    """
    動態計算指定投手的球種庫 (Arsenal) 與九宮格專屬物理特徵
    """
    df_pitcher = df_raw[df_raw['pitcher'] == pitcher_id].copy()
    if df_pitcher.empty:
        raise ValueError(f"在資料集中找不到 ID 為 {pitcher_id} 的投手資料。")

    # 1. 排除樣本數過少的罕見球種
    pitch_counts = df_pitcher['pitch_type'].value_counts()
    valid_pitches = pitch_counts[pitch_counts >= min_pitch_count].index.tolist()
    df_pitcher = df_pitcher[df_pitcher['pitch_type'].isin(valid_pitches)].copy()

    # 2. 全域球種平均特徵 (Overall Arsenal)
    overall_arsenal = {}
    for p_type in valid_pitches:
        p_df = df_pitcher[df_pitcher['pitch_type'] == p_type]
        overall_arsenal[p_type] = {
            'speed': float(round(p_df['release_speed'].mean() / 100.0, 4)),
            'pfx_x': float(round(p_df['pfx_x'].mean(), 4)),
            'pfx_z': float(round(p_df['pfx_z'].mean(), 4)),
            'count': int(len(p_df))
        }

    # 3. 切分九宮格空間
    df_pitcher['zone_x'] = pd.cut(
        df_pitcher['plate_x'], 
        bins=[-np.inf, -0.28, 0.28, np.inf], 
        labels=['左', '中', '右']
    )
    df_pitcher['zone_z'] = pd.cut(
        df_pitcher['plate_z_norm'], 
        bins=[-np.inf, -0.33, 0.33, np.inf], 
        labels=['下', '中', '上']
    )
    
    # 組合宮格名稱 (例如: "左" + "上" = "左上", "中" + "中" 轉為 "正中")
    raw_zone = df_pitcher['zone_x'].astype(str) + df_pitcher['zone_z'].astype(str)
    df_pitcher['zone_9'] = raw_zone.replace({'中中': '正中'})

    # 4. 計算九宮格專屬物理特徵
    zone_group = df_pitcher.groupby(['pitch_type', 'zone_9']).agg({
        'release_speed': lambda x: round(x.mean() / 100.0, 4),
        'pfx_x': lambda x: round(x.mean(), 4),
        'pfx_z': lambda x: round(x.mean(), 4),
        'pitch_number': 'count'
    }).reset_index()

    zone_arsenal = {}
    for _, row in zone_group.iterrows():
        key = (row['pitch_type'], row['zone_9'])
        zone_arsenal[key] = {
            'speed': float(row['release_speed']),
            'pfx_x': float(row['pfx_x']),
            'pfx_z': float(row['pfx_z']),
            'count': int(row['pitch_number'])
        }

    return {
        'pitcher_id': pitcher_id,
        'valid_pitches': valid_pitches,
        'overall_arsenal': overall_arsenal,
        'zone_arsenal': zone_arsenal
    }


def get_pitch_physics(arsenal_data, pitch_type, zone_name):
    """
    查詢特定球種在特定宮格的物理數值；若無該宮格樣本則回退至全域平均
    """
    key = (pitch_type, zone_name)
    if key in arsenal_data['zone_arsenal']:
        return arsenal_data['zone_arsenal'][key]
    # Fallback 到該球種的全域平均
    return arsenal_data['overall_arsenal'][pitch_type]