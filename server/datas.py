import pandas as pd
import numpy as np
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pybaseball import statcast, batting_stats, chadwick_register
import warnings
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore', category=FutureWarning)

def download_mlb_statcast_data(start_date, end_date):
    """按月分批下載 MLB Statcast 逐球數據"""
    current_start = datetime.strptime(start_date, '%Y-%m-%d')
    final_end = datetime.strptime(end_date, '%Y-%m-%d')
    all_data = []
    
    while current_start <= final_end:
        current_end = current_start + relativedelta(months=1) - relativedelta(days=1)
        if current_end > final_end:
            current_end = final_end
            
        str_start = current_start.strftime('%Y-%m-%d')
        str_end = current_end.strftime('%Y-%m-%d')
        
        print(f"正在下載: {str_start} 至 {str_end} ...")
        try:
            df_month = statcast(start_dt=str_start, end_dt=str_end)
            if not df_month.empty:
                all_data.append(df_month)
        except Exception as e:
            print(f"下載 {str_start} 區間發生錯誤: {e}")
            
        current_start += relativedelta(months=1)
        time.sleep(3)
        
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def process_and_filter_data(df):
    print("\n嚴格篩選欄位並執行特徵工程...")
    
    RAW_COLS = [
        'game_date', 'game_pk', 'at_bat_number', 'pitch_number',
        'pitcher', 'pitcher_name', 'batter', 'batter_name', # 🌟 這裡新增了兩欄
        'pitch_type', 'release_speed', 'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
        'balls', 'strikes',
        'stand', 'p_throws',
        'sz_top', 'sz_bot', 'inning', 'inning_topbot',
        'description', 'launch_speed','events',
        'avg_bat_speed', 'hard_swing_rate','Pull%'
    ]
    df = df[[col for col in RAW_COLS if col in df.columns]].copy()
    df = df.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    
    # 升級版 SP/RP 判定邏輯：找出該場比賽、該半局(主/客隊)「最先登場的打席」所對應的投手
    first_ab_per_team = df.groupby(['game_pk', 'inning_topbot'])['at_bat_number'].transform('min')
    sp_list = df[df['at_bat_number'] == first_ab_per_team]['pitcher'].unique()
    # 如果該投手在該場比賽是作為第一任投手登場，就是 SP，其餘皆為 RP
    df['pitcher_role'] = np.where(df['pitcher'].isin(sp_list) & (df['inning'] <= 2), 'SP', 'RP')
    
    # 棒次推算
    team_ab_rank = df.groupby(['game_pk', 'inning_topbot'])['at_bat_number'].rank(method='dense').astype(int)
    df['batting_order'] = ((team_ab_rank - 1) % 9) + 1
    
    # 對決次數
    df['times_faced'] = df.groupby(['game_pk', 'pitcher', 'batter'])['at_bat_number'].rank(method='dense').astype(int)
    
    # 刪除 1 球打席
    # max_pitch = df.groupby(['game_pk', 'at_bat_number'])['pitch_number'].transform('max')
    # df = df[max_pitch > 1].copy()
    
    # 群組與標準化
    df['Matchup_Group'] = df['p_throws'] + "_" + df['stand'] + "_" + df['pitcher_role']
    df['plate_z_norm'] = (df['plate_z'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    
    # 補回 pitcher 與 batter ID 的最終名單
    FINAL_COLS = [
        'game_date', 'game_pk', 'at_bat_number', 'pitch_number',
        'pitcher', 'pitcher_name', 'batter', 'batter_name', # 🌟 最終輸出也保留這兩欄
        'p_throws', 'stand', 'pitcher_role', 'Matchup_Group',
        'batting_order', 'times_faced',
        'avg_bat_speed', 'hard_swing_rate','Pull%',
        'balls', 'strikes',
        'pitch_type', 'release_speed', 'plate_x', 'plate_z_norm', 'pfx_x', 'pfx_z',
        'description', 'launch_speed','events'
    ]
    
    return df[FINAL_COLS].copy()

if __name__ == "__main__":
    TARGET_YEAR = 2026  # 獨立出年份變數，方便下方抓 FanGraphs 使用
    START_DATE = f'{TARGET_YEAR}-03-25'
    END_DATE = f'{TARGET_YEAR}-09-05'
    OUTPUT_FILENAME = f'mlb_pitch_data_{TARGET_YEAR}_strictly_filtered_test.csv.gz'
    BAT_TRACKING_CSV = f'savant_bat_tracking_{TARGET_YEAR}.csv'
    
    df_raw = download_mlb_statcast_data(START_DATE, END_DATE)
    
    if not df_raw.empty:
        print(f"\n原始下載球數: {len(df_raw)}")
        df_raw = df_raw[df_raw['game_type'] == 'R'].copy()
        print(f"已篩選例行賽 (game_type == 'R')，剩餘球數: {len(df_raw)}")

        print("\n正在從 Chadwick 資料庫獲取並映射球員真實姓名...")
        try:
            chadwick = chadwick_register()
            # 整理出包含 MLBAM ID 與姓名的對應表
            name_map = chadwick[['key_mlbam', 'name_first', 'name_last']].dropna(subset=['key_mlbam']).copy()
            name_map['key_mlbam'] = name_map['key_mlbam'].astype(int)
            
            # 將名字首字母大寫並合併 (例如: "aaron", "judge" 變成 "Aaron Judge")
            name_map['full_name'] = name_map['name_first'].astype(str).str.title() + ' ' + name_map['name_last'].astype(str).str.title()
            
            # 轉為 Dictionary 以提升映射效率
            name_dict = name_map.set_index('key_mlbam')['full_name'].to_dict()

            # 將姓名寫入 df_raw
            df_raw['pitcher_name'] = df_raw['pitcher'].map(name_dict).fillna("Unknown Pitcher")
            df_raw['batter_name'] = df_raw['batter'].map(name_dict).fillna("Unknown Batter")
            
            print(f"   => 姓名映射完成！")
        except Exception as e:
            print(f"   [警告] 姓名映射發生錯誤: {e}")
            df_raw['pitcher_name'] = "Unknown Pitcher"
            df_raw['batter_name'] = "Unknown Batter"


        # ==============================================================================
        # 【合併揮棒速度與快速揮棒比例】(此處統一縮排 8 個空格)
        # ==============================================================================
        try:
            print(f"\n正在讀取並合併揮棒數據: {BAT_TRACKING_CSV} ...")
            bat_df = pd.read_csv(BAT_TRACKING_CSV)
            bat_df = bat_df[['id', 'avg_bat_speed', 'hard_swing_rate']].copy()
            bat_df = bat_df.dropna(subset=['id'])
            bat_df['id'] = bat_df['id'].astype(int)
            
            df_raw = df_raw.merge(bat_df, left_on='batter', right_on='id', how='left')
            df_raw = df_raw.drop(columns=['id'])
            print(f"   => 成功合併！有效揮棒數據筆數: {df_raw['avg_bat_speed'].notna().sum()}")
            
        except FileNotFoundError:
            print(f"找不到 {BAT_TRACKING_CSV}，程式將填入空值。")
            df_raw['avg_bat_speed'] = pd.NA
            df_raw['hard_swing_rate'] = pd.NA
        # ==============================================================================

        # ==============================================================================
        # 【合併當季拉打率 (支援 id 與 pull_rate，自動判斷 ID 類型)】
        # ==============================================================================
        try:
            FANGRAPHS_CSV = f'fangraphs_batted_ball_{TARGET_YEAR}.csv'
            print(f"\n正在讀取並整合 {TARGET_YEAR} 年拉打率 ({FANGRAPHS_CSV})...")
            
            pull_df = pd.read_csv(FANGRAPHS_CSV)
            
            # 1. 欄位提取與清理
            pull_df = pull_df[['id', 'pull_rate']].dropna(subset=['id']).copy()
            pull_df['id'] = pull_df['id'].astype(int)
            
            # 若數值帶有 % 字尾 (例如 "42.5%") 自動去除轉浮點數
            if pull_df['pull_rate'].dtype == object:
                pull_df['pull_rate'] = pull_df['pull_rate'].astype(str).str.rstrip('%').astype(float)
            
            # 轉為後續特徵工程統一使用的欄位名稱 'Pull%'
            pull_df = pull_df.rename(columns={'pull_rate': 'Pull%'})

            # 2. 自動判定 ID 類型 (MLBAM ID 或是 FanGraphs ID)
            # 若檔案中的 ID 超過 20% 能直接對上 Statcast batter，代表該檔案已是 MLBAM ID (如 Savant 下載)
            direct_match_count = pull_df['id'].isin(df_raw['batter']).sum()
            
            if direct_match_count > (len(pull_df) * 0.2):
                # 直接對接
                df_raw = df_raw.merge(pull_df, left_on='batter', right_on='id', how='left')
                df_raw = df_raw.drop(columns=['id'])
            else:
                # 透過 Chadwick 橋接 FanGraphs ID -> MLBAM ID
                chadwick = chadwick_register()
                mapping = chadwick[['key_mlbam', 'key_fangraphs']].dropna().copy()
                mapping['key_mlbam'] = mapping['key_mlbam'].astype(int)
                mapping['key_fangraphs'] = mapping['key_fangraphs'].astype(int)

                batter_pull = mapping.merge(pull_df, left_on='key_fangraphs', right_on='id', how='inner')
                batter_pull = batter_pull[['key_mlbam', 'Pull%']].drop_duplicates(subset=['key_mlbam'])
                batter_pull = batter_pull.rename(columns={'key_mlbam': 'batter'})

                df_raw = df_raw.merge(batter_pull, on='batter', how='left')

            print(f"   => 成功合併 Pull%！有效筆數: {df_raw['Pull%'].notna().sum()}")

        except FileNotFoundError:
            print(f"   [警告] 找不到 {FANGRAPHS_CSV}，填入空值。")
            df_raw['Pull%'] = pd.NA
        except Exception as e:
            print(f"   [警告] 處理 Pull% 發生錯誤: {e}，填入空值。")
            df_raw['Pull%'] = pd.NA
        # ==============================================================================

        df_final = process_and_filter_data(df_raw)
        
        # ==============================================================================
        # 【整打席連根剔除邏輯】
        # ==============================================================================
        print("\n開始檢查並剔除無效打席（整打席剔除）...")
        total_pas_before = df_final[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]

        TRACK_COLS = ['release_speed', 'plate_x', 'plate_z_norm', 'pfx_x', 'pfx_z','Pull%']
        nan_condition = df_final[TRACK_COLS].isna().any(axis=1)
        hbp_condition = df_final['description'] == 'hit_by_pitch'

        bad_pas = df_final[nan_condition | hbp_condition][['game_pk', 'at_bat_number']].drop_duplicates()

        df_final = df_final.merge(bad_pas.assign(is_bad=True), on=['game_pk', 'at_bat_number'], how='left')
        df_final = df_final[df_final['is_bad'].isna()].drop(columns=['is_bad']).copy()

        total_pas_after = df_final[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]
        print(f"   - 原始打席數: {total_pas_before}")
        print(f"   - 剔除打席數: {total_pas_before - total_pas_after} 個 (含觸身球或缺漏軌跡)")
        print(f"   - 最終保留完整打席數: {total_pas_after}")
        # ==============================================================================

        print(f"\n輸出至 {OUTPUT_FILENAME}，總筆數: {len(df_final)}")
        df_final.to_csv(OUTPUT_FILENAME, index=False, compression='gzip')
        print("資料輸出完成。")
        
        print(f"\n最終欄位清單 (共 {len(df_final.columns)} 個):")
        print(df_final.columns.tolist())
    else:
        print("未獲取資料。")