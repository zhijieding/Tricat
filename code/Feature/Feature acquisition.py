import pandas as pd
import numpy as np
import os

# --- 配置参数 ---
# Windows路径处理：使用原始字符串r""
data_directory_path = "/mnt/train-decision-agent/dingzhijie/Trident-Cat code modification/数据/赛季数据"
# season_csv_files = [
#     "0506.csv", "0607.csv", "0708.csv", "0809.csv", "0910.csv",
#     "1011.csv", "1112.csv", "1213.csv", "1314.csv", "1415.csv",
#     "1516.csv"
# ]

# 根据您的图片和描述，赛季文件从0506.csv到1516.csv
season_csv_files = [
"1415.csv","1516.csv","1617.csv", "1718.csv", "1819.csv", "1920.csv", "2021.csv",
"2122.csv", "2223.csv", "2324.csv"
]
gamma = 0.3 # 更新规则中的 gamma 值
k_value = 6 # 参数k
# 从第 (10k+1) 场比赛开始计算，0-based索引是 10*k
start_calculation_match_index = 10 * k_value 

# --- 直接执行的区域 ---
# 遍历并处理每个赛季文件
for csv_file_name in season_csv_files:
    full_file_path = os.path.join(data_directory_path, csv_file_name)
    
    if not os.path.exists(full_file_path):
        print(f"文件 {full_file_path} 不存在，跳过。")
        continue

    season_file_name_display = os.path.basename(full_file_path) # 用于打印的文件名
    print(f"\n--- 正在处理赛季文件: {season_file_name_display} ---")

    # 一、读取表格，输出路径也是这个
    # 二、存储为dataframe名字为df
    try:
        # 尝试使用常见的编码读取CSV
        try:
            df = pd.read_csv(full_file_path)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(full_file_path, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(full_file_path, encoding='iso-8859-1')
        
        if df.empty:
            print(f"  文件 {full_file_path} 为空或读取失败，跳过。")
            continue
    except Exception as e_read:
        print(f"  读取文件 {full_file_path} 时出错: {e_read}，跳过。")
        continue

    # 三、在表格最后加入两列。一列为主队表现值(Hform)，一列为客队表现值(Aform)
    df['Hform'] = np.nan
    df['Aform'] = np.nan
    
    # 四、使用字典存储球队当前的form值,每支球队初始的form值为1
    #    这个字典将在每个赛季开始时重新初始化
    team_values = {} 
    
    # 获取本赛季所有参赛队伍并初始化form值为1.0
    # 确保球队名称是字符串类型，并去除可能的NaN值
    home_teams_in_season = df['HomeTeam'].dropna().astype(str).unique()
    away_teams_in_season = df['AwayTeam'].dropna().astype(str).unique()
    all_teams_this_season = np.union1d(home_teams_in_season, away_teams_in_season)
    
    for team_name_init in all_teams_this_season:
        if pd.notna(team_name_init) and team_name_init.lower() != 'nan': # 确保不是 'nan' 字符串
             team_values[team_name_init] = 1.0
    
    print(f"  赛季 {season_file_name_display} 初始化的球队数量: {len(team_values)}")
    if not team_values:
        print(f"  警告: 赛季 {season_file_name_display} 未找到任何有效球队名称，请检查 'HomeTeam' 和 'AwayTeam' 列。")
    
    # 五、遍历每一场比赛进行主场表现(Hform)值、客场表现(Aform)值更新
    # 六、主场表现(Hform)值、客场表现(Aform)值更新规则
    print(f"  开始处理 {season_file_name_display} 的比赛数据...")
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data['HomeTeam'])
        away_team = str(row_data['AwayTeam'])
        ftr_result = str(row_data.get('FTR', '')) # 获取FTR，确保是字符串

        # 检查球队名称是否有效 (不是 'nan' 字符串且存在于 team_values)
        valid_home_team = pd.notna(home_team) and home_team.lower() != 'nan' and home_team in team_values
        valid_away_team = pd.notna(away_team) and away_team.lower() != 'nan' and away_team in team_values

        if not (valid_home_team and valid_away_team):
            print(f"    比赛 {match_idx + 1}: 跳过，无效球队名称 Home='{home_team}', Away='{away_team}' 或球队未初始化。")
            continue
            
        # 获取比赛前球队的Form值
        current_home_form = team_values.get(home_team, 1.0) # 如果意外丢失，默认为1.0
        current_away_form = team_values.get(away_team, 1.0)

        # 将赛前Form值填入当前比赛的Hform和Aform列
        df.loc[match_idx, 'Hform'] = current_home_form
        df.loc[match_idx, 'Aform'] = current_away_form
        
        # 初始化本次更新后的form值（如果比赛结果无效则保持不变）
        new_home_form = current_home_form
        new_away_form = current_away_form

        # 根据比赛结果FTR更新team_values中的Form值
        if pd.notna(ftr_result) and ftr_result in ['H', 'D', 'A']:
            if ftr_result == 'H':
                new_home_form = current_home_form + gamma * current_away_form
                new_away_form = current_away_form - gamma * current_away_form
            elif ftr_result == 'A':
                new_away_form = current_away_form + gamma * current_home_form
                new_home_form = current_home_form - gamma * current_home_form
            elif ftr_result == 'D':
                # 确保分母不为0，尽管在此不太可能
                if (current_home_form - current_away_form) != 0:
                     new_home_form = current_home_form - gamma * (current_home_form - current_away_form)
                # 确保分母不为0
                if (current_away_form - current_home_form) != 0:
                     new_away_form = current_away_form - gamma * (current_away_form - current_home_form)
            
            # 更新team_values字典中球队的form值，供后续比赛使用
            team_values[home_team] = new_home_form
            team_values[away_team] = new_away_form
            
            # 每次都打印更新的两支球队名字及form值
            print(f"    比赛 {match_idx + 1} ({home_team} vs {away_team}, 结果: {ftr_result}):")
            print(f"      {home_team}: 原Form={current_home_form:.4f}, 更新后Form={new_home_form:.4f}")
            print(f"      {away_team}: 原Form={current_away_form:.4f}, 更新后Form={new_away_form:.4f}")
        else:
            print(f"    比赛 {match_idx + 1} ({home_team} vs {away_team}): FTR值 '{ftr_result}' 无效或缺失，未更新Form值。")


    # 再在表格最后一列加入列FormDifferential
    # 本场比赛的FormDifferential的值=Hform的值-Aform的值
    df['FormDifferential'] = df['Hform'] - df['Aform']

    # 七、输出Hform、Aform、FormDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。Hform, Aform, FormDifferential (前5行):")
    if not df.empty:
        print(df[['HomeTeam','AwayTeam','FTR','Hform', 'Aform', 'FormDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")

    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}")


    # 三、在表格最后加入两列。一列为Hst，一列为ASt
    df['Hst'] = np.nan
    df['ASt'] = np.nan
    
    # 四、用于存储每支球队的res历史记录 (每个赛季独立)
    # team_res_history[team_name] = [res1, res2, ...]
    team_res_history = {} 

    # 五、遍历每场比赛
    print(f"  --- 开始处理 {season_file_name_display} 的比赛数据 ---")
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', '')) # 使用 .get() 并确保是字符串
        away_team = str(row_data.get('AwayTeam', '')) # 使用 .get() 并确保是字符串
        ftr_result = str(row_data.get('FTR', ''))    # 使用 .get() 并确保是字符串

        # 1. 每场比赛记录两支球队的res值
        home_res_current = np.nan
        away_res_current = np.nan

        valid_ftr = ftr_result in ['H', 'D', 'A']
        valid_home_team_name = pd.notna(home_team) and home_team.lower() != 'nan' and home_team != ''
        valid_away_team_name = pd.notna(away_team) and away_team.lower() != 'nan' and away_team != ''

        if valid_ftr and valid_home_team_name and valid_away_team_name:
            if ftr_result == 'H':
                home_res_current = 3
                away_res_current = 0
            elif ftr_result == 'D':
                home_res_current = 1
                away_res_current = 1
            elif ftr_result == 'A':
                home_res_current = 0
                away_res_current = 3
            
            # 输出两个球队的名称及res值
            print(f"    比赛 {match_idx + 1}: {home_team} vs {away_team}, FTR: {ftr_result}")
            print(f"      {home_team} (主队) res: {home_res_current}")
            print(f"      {away_team} (客队) res: {away_res_current}")
        else:
            print(f"    比赛 {match_idx + 1}: Home='{home_team}', Away='{away_team}', FTR='{ftr_result}'. 跳过res计算 (无效球队名或FTR)。")


        # 2. 从（10k+1）场比赛开始同时更新主场连胜场次(Hst)和客场连胜场次(ASt)
        if match_idx >= start_calculation_match_index:
            denominator = (3 * k_value)
            if denominator == 0: # 避免除以零
                if match_idx == start_calculation_match_index: print("      错误: k_value 导致分母为零，跳过 Hst/ASt 计算。")
            else:
                # 计算主队 Hst
                if valid_home_team_name and home_team in team_res_history and len(team_res_history[home_team]) >= k_value:
                    # "本场比赛前这支球队前k个res的值"
                    previous_k_res_home = [res for res in team_res_history[home_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_home) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_home = sum(previous_k_res_home)
                        df.loc[match_idx, 'Hst'] = sum_k_res_home / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {home_team} 计算Hst时，前k个有效res不足。")
                
                # 计算客队 ASt
                if valid_away_team_name and away_team in team_res_history and len(team_res_history[away_team]) >= k_value:
                    previous_k_res_away = [res for res in team_res_history[away_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_away) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_away = sum(previous_k_res_away)
                        df.loc[match_idx, 'ASt'] = sum_k_res_away / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {away_team} 计算ASt时，前k个有效res不足。")
        
        # 将当前比赛的res加入对应球队的历史记录 (在计算完Hst/ASt之后)
        # 只有当res有效且球队名有效时才添加
        if valid_home_team_name:
            if home_team not in team_res_history:
                team_res_history[home_team] = []
            if pd.notna(home_res_current): # 只添加有效的res值
                team_res_history[home_team].append(home_res_current)

        if valid_away_team_name:
            if away_team not in team_res_history:
                team_res_history[away_team] = []
            if pd.notna(away_res_current): # 只添加有效的res值
                team_res_history[away_team].append(away_res_current)
            
    # 再在表格最后一列加入列StDifferential
    df['StDifferential'] = df['Hst'] - df['ASt']

    # 输出Hst、Ast、StDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。Hst, ASt, StDifferential (部分结果):")
    # 打印包含计算开始点附近的一些行，以便观察
    start_print_idx = max(0, start_calculation_match_index - 2) 
    end_print_idx = min(len(df)-1, start_calculation_match_index + 3) 
    
    if not df.empty :
        # 确保切片索引有效
        if start_print_idx <= end_print_idx and start_print_idx < len(df):
             print(df.loc[start_print_idx:end_print_idx, ['HomeTeam','AwayTeam','FTR','Hst', 'ASt', 'StDifferential']])
        else: # 如果计算的索引范围无效（例如，文件行数太少），则打印头部
             print(df[['HomeTeam','AwayTeam','FTR','Hst', 'ASt', 'StDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}")

    # 三、在表格最后加入2列。一列为HCKPP，一列为ACKPP
    df['HCKPP'] = np.nan
    df['ACKPP'] = np.nan
    
    # 四、用于存储每支球队的mu1历史记录 (每个赛季独立)
    team_mu1_history = {} 

    # 五、遍历每场比赛 (针对CKPP)
    missing_hc_reported = False
    missing_ac_reported = False
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', ''))
        away_team = str(row_data.get('AwayTeam', ''))
        
        # 1. 每场比赛分别记录两支球队的mu1值
        mu1_home_current = np.nan
        mu1_away_current = np.nan
        
        if 'HC' in df.columns:
            mu1_home_current = row_data['HC']
        elif not missing_hc_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'HC' 列。HCKPP 相关计算可能不准确。")
            missing_hc_reported = True
        
        if 'AC' in df.columns:
            mu1_away_current = row_data['AC']
        elif not missing_ac_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'AC' 列。ACKPP 相关计算可能不准确。")
            missing_ac_reported = True

        # 2. 从（10k+1）场比赛同时开始更新HCKPP和ACKPP
        if match_idx >= start_calculation_match_index:
            # 更新主队HCKPP
            if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
                if home_team in team_mu1_history and len(team_mu1_history[home_team]) >= k_value:
                    # "本场比赛前这支球队前k个mu1的值"
                    previous_k_mu1_home = [m for m in team_mu1_history[home_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu1_home) == k_value: # 确保有k个有效的历史mu1值
                         df.loc[match_idx, 'HCKPP'] = sum(previous_k_mu1_home) / k_value
                    # else:
                        # if match_idx == start_calculation_match_index: print(f"      注意: {home_team} 计算HCKPP时，前k个有效mu1不足。")
            
            # 更新客队ACKPP
            if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
                if away_team in team_mu1_history and len(team_mu1_history[away_team]) >= k_value:
                    previous_k_mu1_away = [m for m in team_mu1_history[away_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu1_away) == k_value: # 确保有k个有效的历史mu1值
                        df.loc[match_idx, 'ACKPP'] = sum(previous_k_mu1_away) / k_value
                    # else:
                        # if match_idx == start_calculation_match_index: print(f"      注意: {away_team} 计算ACKPP时，前k个有效mu1不足。")
        
        # 将当前比赛的mu1加入历史记录 (在计算完HCKPP/ACKPP之后，供下一场使用)
        if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
            if home_team not in team_mu1_history:
                team_mu1_history[home_team] = []
            team_mu1_history[home_team].append(mu1_home_current) # mu1_home_current可能是nan

        if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
            if away_team not in team_mu1_history:
                team_mu1_history[away_team] = []
            team_mu1_history[away_team].append(mu1_away_current) # mu1_away_current可能是nan
            
    df['CKPP'] = df['HCKPP'] - df['ACKPP']
    print(f"  CKPP 相关特征计算完成。部分结果 (HCKPP, ACKPP, CKPP):")
    start_print_idx_ckpp = max(0, start_calculation_match_index - 1)
    end_print_idx_ckpp = min(len(df)-1, start_calculation_match_index + 2)
    if not df.empty and start_print_idx_ckpp <= end_print_idx_ckpp :
        print(df.loc[start_print_idx_ckpp:end_print_idx_ckpp, ['HomeTeam','AwayTeam','HC','AC','HCKPP', 'ACKPP', 'CKPP']])
    elif not df.empty:
        print(df[['HomeTeam','AwayTeam','HC','AC','HCKPP', 'ACKPP', 'CKPP']].head())


    # --- Block 2: 处理 HSTKPP, ASTKPP, STKPP ---
    print(f"\n  --- 处理 STKPP 相关特征 ({season_file_name_display}) ---")
    # 六、在表格最后加入2列。一列为HSTKPP，一列为ASTKPP
    df['HSTKPP'] = np.nan
    df['ASTKPP'] = np.nan

    # 七、用于存储每支球队的mu2历史记录 (每个赛季独立)
    team_mu2_history = {}

    # 八、遍历每场比赛 (针对STKPP)
    missing_hst_reported = False
    missing_ast_reported = False
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', ''))
        away_team = str(row_data.get('AwayTeam', ''))

        # 1. 每场比赛分别记录两支球队的mu2值
        mu2_home_current = np.nan
        mu2_away_current = np.nan

        if 'HST' in df.columns:
            mu2_home_current = row_data['HST']
        elif not missing_hst_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'HST' 列。HSTKPP 相关计算可能不准确。")
            missing_hst_reported = True
        
        if 'AST' in df.columns:
            mu2_away_current = row_data['AST']
        elif not missing_ast_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'AST' 列。ASTKPP 相关计算可能不准确。")
            missing_ast_reported = True

        # 3. 从（10k+1）场比赛同时开始更新HSTKPP和ASTKPP
        if match_idx >= start_calculation_match_index:
            if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
                if home_team in team_mu2_history and len(team_mu2_history[home_team]) >= k_value:
                    previous_k_mu2_home = [m for m in team_mu2_history[home_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu2_home) == k_value:
                        df.loc[match_idx, 'HSTKPP'] = sum(previous_k_mu2_home) / k_value
            
            if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
                if away_team in team_mu2_history and len(team_mu2_history[away_team]) >= k_value:
                    previous_k_mu2_away = [m for m in team_mu2_history[away_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu2_away) == k_value:
                        df.loc[match_idx, 'ASTKPP'] = sum(previous_k_mu2_away) / k_value
        
        if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
            if home_team not in team_mu2_history:
                team_mu2_history[home_team] = []
            team_mu2_history[home_team].append(mu2_home_current)

        if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
            if away_team not in team_mu2_history:
                team_mu2_history[away_team] = []
            team_mu2_history[away_team].append(mu2_away_current)

    df['STKPP'] = df['HSTKPP'] - df['ASTKPP']
    print(f"  STKPP 相关特征计算完成。部分结果 (HSTKPP, ASTKPP, STKPP):")
    start_print_idx_stkpp = max(0, start_calculation_match_index - 1)
    end_print_idx_stkpp = min(len(df)-1, start_calculation_match_index + 2)
    if not df.empty and start_print_idx_stkpp <= end_print_idx_stkpp:
        print(df.loc[start_print_idx_stkpp:end_print_idx_stkpp, ['HomeTeam','AwayTeam','HST','AST','HSTKPP', 'ASTKPP', 'STKPP']])
    elif not df.empty:
        print(df[['HomeTeam','AwayTeam','HST','AST','HSTKPP', 'ASTKPP', 'STKPP']].head())


    # --- Block 3: 处理 HGKPP, AGKPP, GKPP ---
    print(f"\n  --- 处理 GKPP 相关特征 ({season_file_name_display}) ---")
    # 九、在表格最后加入2列。一列为HGKPP，一列为AGKPP
    df['HGKPP'] = np.nan
    df['AGKPP'] = np.nan

    # 十、用于存储每支球队的mu3历史记录 (每个赛季独立)
    team_mu3_history = {}

    # 十一、遍历每场比赛 (针对GKPP)
    missing_fthg_reported = False
    missing_ftag_reported = False
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', ''))
        away_team = str(row_data.get('AwayTeam', ''))

        # 1. 每场比赛分别记录两支球队的mu3值
        mu3_home_current = np.nan
        mu3_away_current = np.nan

        if 'FTHG' in df.columns:
            mu3_home_current = row_data['FTHG']
        elif not missing_fthg_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'FTHG' 列。HGKPP 相关计算可能不准确。")
            missing_fthg_reported = True

        if 'FTAG' in df.columns:
            mu3_away_current = row_data['FTAG']
        elif not missing_ftag_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'FTAG' 列。AGKPP 相关计算可能不准确。")
            missing_ftag_reported = True
            
        # 4. 从（10k+1）场比赛同时开始更新HGKPP和AGKPP
        if match_idx >= start_calculation_match_index:
            if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
                if home_team in team_mu3_history and len(team_mu3_history[home_team]) >= k_value:
                    previous_k_mu3_home = [m for m in team_mu3_history[home_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu3_home) == k_value:
                        df.loc[match_idx, 'HGKPP'] = sum(previous_k_mu3_home) / k_value
            
            if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
                if away_team in team_mu3_history and len(team_mu3_history[away_team]) >= k_value:
                    previous_k_mu3_away = [m for m in team_mu3_history[away_team][-k_value:] if pd.notna(m)]
                    if len(previous_k_mu3_away) == k_value:
                        df.loc[match_idx, 'AGKPP'] = sum(previous_k_mu3_away) / k_value
        
        if pd.notna(home_team) and home_team != '' and home_team.lower() != 'nan':
            if home_team not in team_mu3_history:
                team_mu3_history[home_team] = []
            team_mu3_history[home_team].append(mu3_home_current)

        if pd.notna(away_team) and away_team != '' and away_team.lower() != 'nan':
            if away_team not in team_mu3_history:
                team_mu3_history[away_team] = []
            team_mu3_history[away_team].append(mu3_away_current)

    df['GKPP'] = df['HGKPP'] - df['AGKPP']
    print(f"  GKPP 相关特征计算完成。部分结果 (HGKPP, AGKPP, GKPP):")
    start_print_idx_gkpp = max(0, start_calculation_match_index - 1)
    end_print_idx_gkpp = min(len(df)-1, start_calculation_match_index + 2)
    if not df.empty and start_print_idx_gkpp <= end_print_idx_gkpp:
        print(df.loc[start_print_idx_gkpp:end_print_idx_gkpp, ['HomeTeam','AwayTeam','FTHG','FTAG','HGKPP', 'AGKPP', 'GKPP']])
    elif not df.empty:
        print(df[['HomeTeam','AwayTeam','FTHG','FTAG','HGKPP', 'AGKPP', 'GKPP']].head())


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}")

    # 三、在表格最后加入2列。一列为HTDG，一列为ATDG
    df['HTDG'] = np.nan
    df['ATDG'] = np.nan
    
    # 四、用于存储每支球队的alpha1和alpha2历史记录 (每个赛季独立)
    # team_alpha_history[team_name] = {'alpha1': [], 'alpha2': []}
    team_alpha_history = {} 

    # 五、遍历每场比赛
    print(f"  --- 开始处理 {season_file_name_display} 的比赛数据 ---")
    missing_fthg_reported = False
    missing_ftag_reported = False

    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', '')) # 使用 .get() 并确保是字符串
        away_team = str(row_data.get('AwayTeam', '')) # 使用 .get() 并确保是字符串
        
        fthg_current_match = row_data.get('FTHG') 
        ftag_current_match = row_data.get('FTAG')

        if 'FTHG' not in df.columns and not missing_fthg_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'FTHG' 列。HTDG/ATDG 相关计算可能不准确。")
            missing_fthg_reported = True
        if 'FTAG' not in df.columns and not missing_ftag_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'FTAG' 列。HTDG/ATDG 相关计算可能不准确。")
            missing_ftag_reported = True

        # 1. 每场比赛分别记录两支球队的alpha1（进球数）、alpha2（失球数）值
        alpha1_home_current = np.nan
        alpha2_home_current = np.nan
        alpha1_away_current = np.nan
        alpha2_away_current = np.nan

        valid_home_team_name = pd.notna(home_team) and home_team.lower() != 'nan' and home_team != ''
        valid_away_team_name = pd.notna(away_team) and away_team.lower() != 'nan' and away_team != ''

        print(f"    比赛 {match_idx + 1}: {home_team} vs {away_team}")
        if valid_home_team_name and pd.notna(fthg_current_match) and pd.notna(ftag_current_match):
            alpha1_home_current = fthg_current_match
            alpha2_home_current = ftag_current_match
            print(f"      {home_team} (主队): alpha1 (进球) = {alpha1_home_current}, alpha2 (失球) = {alpha2_home_current}")
        elif valid_home_team_name:
            print(f"      {home_team} (主队): FTHG/FTAG数据缺失或无效，无法记录alpha值。")
        
        if valid_away_team_name and pd.notna(ftag_current_match) and pd.notna(fthg_current_match):
            alpha1_away_current = ftag_current_match
            alpha2_away_current = fthg_current_match
            print(f"      {away_team} (客队): alpha1 (进球) = {alpha1_away_current}, alpha2 (失球) = {alpha2_away_current}")
        elif valid_away_team_name:
            print(f"      {away_team} (客队): FTHG/FTAG数据缺失或无效，无法记录alpha值。")


        # 2. 从（10k+1）场比赛同时开始更新HTDG和ATDG
        if match_idx >= start_calculation_match_index:
            # 更新主队HTDG
            if valid_home_team_name and home_team in team_alpha_history and \
               len(team_alpha_history[home_team]['alpha1']) >= k_value and \
               len(team_alpha_history[home_team]['alpha2']) >= k_value:
                
                # "本场比赛前这支球队前k个alpha1/alpha2的值"
                prev_k_alpha1_home = [val for val in team_alpha_history[home_team]['alpha1'][-k_value:] if pd.notna(val)]
                prev_k_alpha2_home = [val for val in team_alpha_history[home_team]['alpha2'][-k_value:] if pd.notna(val)]

                if len(prev_k_alpha1_home) == k_value and len(prev_k_alpha2_home) == k_value: 
                    sum_alpha1_home = sum(prev_k_alpha1_home)
                    sum_alpha2_home = sum(prev_k_alpha2_home)
                    df.loc[match_idx, 'HTDG'] = sum_alpha1_home - sum_alpha2_home
                # else:
                    # if match_idx == start_calculation_match_index : print(f"      注意: {home_team} 在计算HTDG时历史数据不足k个有效值。")
            
            # 更新客队ATDG
            if valid_away_team_name and away_team in team_alpha_history and \
               len(team_alpha_history[away_team]['alpha1']) >= k_value and \
               len(team_alpha_history[away_team]['alpha2']) >= k_value:
                
                prev_k_alpha1_away = [val for val in team_alpha_history[away_team]['alpha1'][-k_value:] if pd.notna(val)]
                prev_k_alpha2_away = [val for val in team_alpha_history[away_team]['alpha2'][-k_value:] if pd.notna(val)]

                if len(prev_k_alpha1_away) == k_value and len(prev_k_alpha2_away) == k_value: 
                    sum_alpha1_away = sum(prev_k_alpha1_away)
                    sum_alpha2_away = sum(prev_k_alpha2_away)
                    df.loc[match_idx, 'ATDG'] = sum_alpha1_away - sum_alpha2_away
                # else:
                    # if match_idx == start_calculation_match_index : print(f"      注意: {away_team} 在计算ATDG时历史数据不足k个有效值。")
        
        # 将当前比赛的alpha1, alpha2加入对应球队的历史记录
        if valid_home_team_name:
            if home_team not in team_alpha_history:
                team_alpha_history[home_team] = {'alpha1': [], 'alpha2': []}
            team_alpha_history[home_team]['alpha1'].append(alpha1_home_current) # Appending potential NaN if FTHG/FTAG was NaN
            team_alpha_history[home_team]['alpha2'].append(alpha2_home_current) # Appending potential NaN

        if valid_away_team_name:
            if away_team not in team_alpha_history:
                team_alpha_history[away_team] = {'alpha1': [], 'alpha2': []}
            team_alpha_history[away_team]['alpha1'].append(alpha1_away_current) # Appending potential NaN
            team_alpha_history[away_team]['alpha2'].append(alpha2_away_current) # Appending potential NaN
            
    # 再在表格最后一列加入列GDDifferential
    df['GDDifferential'] = df['HTDG'] - df['ATDG']

    # 六、输出HTDG、ATDG、GDDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。HTDG, ATDG, GDDifferential (部分结果):")
    # 打印包含计算开始点附近的一些行，以便观察
    start_print_idx = max(0, start_calculation_match_index - 1)
    end_print_idx = min(len(df)-1, start_calculation_match_index + 2)
    
    if not df.empty :
        if start_print_idx <= end_print_idx and start_print_idx < len(df):
             print(df.loc[start_print_idx:end_print_idx, ['HomeTeam','AwayTeam','FTHG','FTAG','HTDG', 'ATDG', 'GDDifferential']])
        else: # 如果计算的索引范围无效（例如，文件行数太少），则打印头部
             print(df[['HomeTeam','AwayTeam','FTHG','FTAG','HTDG', 'ATDG', 'GDDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}")

    # 三、在表格最后加入两列。一列为HStWeighted，一列为AStWeighted
    df['HStWeighted'] = np.nan
    df['AStWeighted'] = np.nan
    
    # 四、用于存储每支球队的res历史记录 (每个赛季独立)
    # team_res_history[team_name] = [res1, res2, ...]
    team_res_history = {} 

    # 五、遍历每场比赛
    print(f"  --- 开始处理 {season_file_name_display} 的比赛数据 ---")
    missing_ftr_reported = False
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', '')) 
        away_team = str(row_data.get('AwayTeam', '')) 
        ftr_result = str(row_data.get('FTR', ''))    

        if 'FTR' not in df.columns and not missing_ftr_reported:
            print(f"    警告: 文件 {csv_file_name} 缺少 'FTR' 列。res 及相关加权计算将不准确。")
            missing_ftr_reported = True

        # 1. 每场比赛记录两支球队的res值
        home_res_current = np.nan
        away_res_current = np.nan

        valid_home_team_name = pd.notna(home_team) and home_team.lower() != 'nan' and home_team != ''
        valid_away_team_name = pd.notna(away_team) and away_team.lower() != 'nan' and away_team != ''

        print(f"    比赛 {match_idx + 1}: {home_team} vs {away_team}, FTR: {ftr_result}")
        if valid_home_team_name and valid_away_team_name and ftr_result in ['H', 'D', 'A']:
            if ftr_result == 'H':
                home_res_current = 3
                away_res_current = 0
            elif ftr_result == 'D':
                home_res_current = 1
                away_res_current = 1
            elif ftr_result == 'A':
                home_res_current = 0
                away_res_current = 3
            
            if valid_home_team_name:
                 print(f"      {home_team} (主队) res: {home_res_current}")
            if valid_away_team_name:
                 print(f"      {away_team} (客队) res: {away_res_current}")
        else:
            if not (valid_home_team_name and valid_away_team_name):
                print(f"      跳过res计算，无效球队名称: Home='{home_team}', Away='{away_team}'")
            elif not ftr_result in ['H', 'D', 'A']:
                print(f"      跳过res计算，无效FTR值: '{ftr_result}'")


        # 从（10k+1）场比赛开始同时更新主场加权连胜场次（HStWeighted）和客场加权连胜场次（AStWeighted）
        if match_idx >= start_calculation_match_index:
            denominator = (3 * k_value) * (k_value + 1)
            if denominator == 0: 
                if match_idx == start_calculation_match_index: print("      错误: k_value 导致分母为零，跳过 HStWeighted/AStWeighted 计算。")
            else:
                # 计算主队 HStWeighted
                if valid_home_team_name and home_team in team_res_history and len(team_res_history[home_team]) >= k_value:
                    # "本场比赛前这支球队前k个res的值"
                    previous_k_res_home_all = team_res_history[home_team][-k_value:]
                    # 确保所有值都是数字 (过滤掉可能的NaN)
                    previous_k_res_home_valid = [res for res in previous_k_res_home_all if pd.notna(res)]

                    if len(previous_k_res_home_valid) == k_value: # 严格要求k个有效值
                        weighted_sum_home = 0
                        # "第1个res值*1...第k个res值*k", "离的最远的为第1个"
                        # previous_k_res_home_valid is [oldest_of_k, ..., newest_of_k]
                        for i in range(k_value):
                            weighted_sum_home += previous_k_res_home_valid[i] * (i + 1)
                        
                        w_home = (2 * weighted_sum_home) / denominator
                        df.loc[match_idx, 'HStWeighted'] = w_home
                    # else:
                        # if match_idx == start_calculation_match_index: print(f"      注意: {home_team} 计算HStWeighted时，前k个有效res不足 {len(previous_k_res_home_valid)}/{k_value}。")
                
                # 计算客队 AStWeighted
                if valid_away_team_name and away_team in team_res_history and len(team_res_history[away_team]) >= k_value:
                    previous_k_res_away_all = team_res_history[away_team][-k_value:]
                    previous_k_res_away_valid = [res for res in previous_k_res_away_all if pd.notna(res)]

                    if len(previous_k_res_away_valid) == k_value: # 严格要求k个有效值
                        weighted_sum_away = 0
                        for i in range(k_value):
                            weighted_sum_away += previous_k_res_away_valid[i] * (i + 1)
                        
                        w_away = (2 * weighted_sum_away) / denominator
                        df.loc[match_idx, 'AStWeighted'] = w_away
                    # else:
                        # if match_idx == start_calculation_match_index: print(f"      注意: {away_team} 计算AStWeighted时，前k个有效res不足 {len(previous_k_res_away_valid)}/{k_value}。")
        
        # 将当前比赛的res加入对应球队的历史记录 (在计算完HStWeighted/AStWeighted之后)
        # 只有当res有效且球队名有效时才添加
        if valid_home_team_name:
            if home_team not in team_res_history:
                team_res_history[home_team] = []
            team_res_history[home_team].append(home_res_current) # home_res_current可能是nan

        if valid_away_team_name:
            if away_team not in team_res_history:
                team_res_history[away_team] = []
            team_res_history[away_team].append(away_res_current) # away_res_current可能是nan
            
    # 再在表格最后一列加入列StWeightedDifferential
    df['StWeightedDifferential'] = df['HStWeighted'] - df['AStWeighted']

    # 输出HStWeighted、AStWeighted、StWeightedDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。HStWeighted, AStWeighted, StWeightedDifferential (部分结果):")
    # 打印包含计算开始点附近的一些行，以便观察
    start_print_idx = max(0, start_calculation_match_index - 1) 
    end_print_idx = min(len(df)-1, start_calculation_match_index + 2) 
    
    if not df.empty :
        if start_print_idx <= end_print_idx and start_print_idx < len(df):
             print(df.loc[start_print_idx:end_print_idx, ['HomeTeam','AwayTeam','FTR','HStWeighted', 'AStWeighted', 'StWeightedDifferential']])
        else: 
             print(df[['HomeTeam','AwayTeam','FTR','HStWeighted', 'AStWeighted', 'StWeightedDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}")


    # 三、在表格最后加入两列。一列为Hst，一列为ASt
    df['HF'] = np.nan
    df['AF'] = np.nan
    
    # 四、用于存储每支球队的res历史记录 (每个赛季独立)
    # team_res_history[team_name] = [res1, res2, ...]
    team_res_history = {} 

    # 五、遍历每场比赛
    print(f"  --- 开始处理 {season_file_name_display} 的比赛数据 ---")
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', '')) # 使用 .get() 并确保是字符串
        away_team = str(row_data.get('AwayTeam', '')) # 使用 .get() 并确保是字符串
        ftr_result = str(row_data.get('FTR', ''))    # 使用 .get() 并确保是字符串

        # 1. 每场比赛记录两支球队的res值
        home_res_current = np.nan
        away_res_current = np.nan

        valid_ftr = ftr_result in ['H', 'D', 'A']
        valid_home_team_name = pd.notna(home_team) and home_team.lower() != 'nan' and home_team != ''
        valid_away_team_name = pd.notna(away_team) and away_team.lower() != 'nan' and away_team != ''

        if valid_ftr and valid_home_team_name and valid_away_team_name:
            if ftr_result == 'H':
                home_res_current = 3
                away_res_current = 0
            elif ftr_result == 'D':
                home_res_current = 1
                away_res_current = 1
            elif ftr_result == 'A':
                home_res_current = 0
                away_res_current = 3
            
            # 输出两个球队的名称及res值
            print(f"    比赛 {match_idx + 1}: {home_team} vs {away_team}, FTR: {ftr_result}")
            print(f"      {home_team} (主队) res: {home_res_current}")
            print(f"      {away_team} (客队) res: {away_res_current}")
        else:
            print(f"    比赛 {match_idx + 1}: Home='{home_team}', Away='{away_team}', FTR='{ftr_result}'. 跳过res计算 (无效球队名或FTR)。")


        # 2. 从（10k+1）场比赛开始同时更新主场连胜场次(H)和客场连胜场次(ASt)
        if match_idx >= start_calculation_match_index:
            denominator = (3 * k_value)
            if denominator == 0: # 避免除以零
                if match_idx == start_calculation_match_index: print("      错误: k_value 导致分母为零，跳过 HF/AF 计算。")
            else:
                # 计算主队 Hst
                if valid_home_team_name and home_team in team_res_history and len(team_res_history[home_team]) >= k_value:
                    # "本场比赛前这支球队前k个res的值"
                    previous_k_res_home = [res for res in team_res_history[home_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_home) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_home = sum(previous_k_res_home)
                        df.loc[match_idx, 'HF'] = sum_k_res_home / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {home_team} 计算HF时，前k个有效res不足。")
                
                # 计算客队 ASt
                if valid_away_team_name and away_team in team_res_history and len(team_res_history[away_team]) >= k_value:
                    previous_k_res_away = [res for res in team_res_history[away_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_away) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_away = sum(previous_k_res_away)
                        df.loc[match_idx, 'AF'] = sum_k_res_away / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {away_team} 计算AF时，前k个有效res不足。")
        
        # 将当前比赛的res加入对应球队的历史记录 (在计算完Hst/ASt之后)
        # 只有当res有效且球队名有效时才添加
        if valid_home_team_name:
            if home_team not in team_res_history:
                team_res_history[home_team] = []
            if pd.notna(home_res_current): # 只添加有效的res值
                team_res_history[home_team].append(home_res_current)

        if valid_away_team_name:
            if away_team not in team_res_history:
                team_res_history[away_team] = []
            if pd.notna(away_res_current): # 只添加有效的res值
                team_res_history[away_team].append(away_res_current)
            
    # 再在表格最后一列加入列StDifferential
    df['FDifferential'] = df['HF'] - df['AF']

    # 输出Hst、Ast、StDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。HF, AF, FDifferential (部分结果):")
    # 打印包含计算开始点附近的一些行，以便观察
    start_print_idx = max(0, start_calculation_match_index - 2) 
    end_print_idx = min(len(df)-1, start_calculation_match_index + 3) 
    
    if not df.empty :
        # 确保切片索引有效
        if start_print_idx <= end_print_idx and start_print_idx < len(df):
             print(df.loc[start_print_idx:end_print_idx, ['HomeTeam','AwayTeam','FTR','HF', 'AF', 'FDifferential']])
        else: # 如果计算的索引范围无效（例如，文件行数太少），则打印头部
             print(df[['HomeTeam','AwayTeam','FTR','HF', 'AF', 'FDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}") 


    # 三、在表格最后加入两列。一列为Hst，一列为ASt
    df['HS'] = np.nan
    df['AS'] = np.nan
    
    # 四、用于存储每支球队的res历史记录 (每个赛季独立)
    # team_res_history[team_name] = [res1, res2, ...]
    team_res_history = {} 

    # 五、遍历每场比赛
    print(f"  --- 开始处理 {season_file_name_display} 的比赛数据 ---")
    for match_idx, row_data in df.iterrows():
        home_team = str(row_data.get('HomeTeam', '')) # 使用 .get() 并确保是字符串
        away_team = str(row_data.get('AwayTeam', '')) # 使用 .get() 并确保是字符串
        ftr_result = str(row_data.get('FTR', ''))    # 使用 .get() 并确保是字符串

        # 1. 每场比赛记录两支球队的res值
        home_res_current = np.nan
        away_res_current = np.nan

        valid_ftr = ftr_result in ['H', 'D', 'A']
        valid_home_team_name = pd.notna(home_team) and home_team.lower() != 'nan' and home_team != ''
        valid_away_team_name = pd.notna(away_team) and away_team.lower() != 'nan' and away_team != ''

        if valid_ftr and valid_home_team_name and valid_away_team_name:
            if ftr_result == 'H':
                home_res_current = 3
                away_res_current = 0
            elif ftr_result == 'D':
                home_res_current = 1
                away_res_current = 1
            elif ftr_result == 'A':
                home_res_current = 0
                away_res_current = 3
            
            # 输出两个球队的名称及res值
            print(f"    比赛 {match_idx + 1}: {home_team} vs {away_team}, FTR: {ftr_result}")
            print(f"      {home_team} (主队) res: {home_res_current}")
            print(f"      {away_team} (客队) res: {away_res_current}")
        else:
            print(f"    比赛 {match_idx + 1}: Home='{home_team}', Away='{away_team}', FTR='{ftr_result}'. 跳过res计算 (无效球队名或FTR)。")


        # 2. 从（10k+1）场比赛开始同时更新主场连胜场次(H)和客场连胜场次(ASt)
        if match_idx >= start_calculation_match_index:
            denominator = (3 * k_value)
            if denominator == 0: # 避免除以零
                if match_idx == start_calculation_match_index: print("      错误: k_value 导致分母为零，跳过 HS/AS 计算。")
            else:
                # 计算主队 Hst
                if valid_home_team_name and home_team in team_res_history and len(team_res_history[home_team]) >= k_value:
                    # "本场比赛前这支球队前k个res的值"
                    previous_k_res_home = [res for res in team_res_history[home_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_home) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_home = sum(previous_k_res_home)
                        df.loc[match_idx, 'HS'] = sum_k_res_home / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {home_team} 计算Hst时，前k个有效res不足。")
                
                # 计算客队 ASt
                if valid_away_team_name and away_team in team_res_history and len(team_res_history[away_team]) >= k_value:
                    previous_k_res_away = [res for res in team_res_history[away_team][-k_value:] if pd.notna(res)]
                    if len(previous_k_res_away) == k_value: # 确保有k个有效的历史res值
                        sum_k_res_away = sum(previous_k_res_away)
                        df.loc[match_idx, 'AS'] = sum_k_res_away / denominator
                    else:
                        if match_idx == start_calculation_match_index: print(f"      注意: {away_team} 计算AS时，前k个有效res不足。")
        
        # 将当前比赛的res加入对应球队的历史记录 (在计算完Hst/ASt之后)
        # 只有当res有效且球队名有效时才添加
        if valid_home_team_name:
            if home_team not in team_res_history:
                team_res_history[home_team] = []
            if pd.notna(home_res_current): # 只添加有效的res值
                team_res_history[home_team].append(home_res_current)

        if valid_away_team_name:
            if away_team not in team_res_history:
                team_res_history[away_team] = []
            if pd.notna(away_res_current): # 只添加有效的res值
                team_res_history[away_team].append(away_res_current)
            
    # 再在表格最后一列加入列StDifferential
    df['SDifferential'] = df['HS'] - df['AS']

    # 输出Hst、Ast、StDifferential的结果
    print(f"\n  赛季 {season_file_name_display} 处理完毕。HS, AS, SDifferential (部分结果):")
    # 打印包含计算开始点附近的一些行，以便观察
    start_print_idx = max(0, start_calculation_match_index - 2) 
    end_print_idx = min(len(df)-1, start_calculation_match_index + 3) 
    
    if not df.empty :
        # 确保切片索引有效
        if start_print_idx <= end_print_idx and start_print_idx < len(df):
             print(df.loc[start_print_idx:end_print_idx, ['HomeTeam','AwayTeam','FTR','HS', 'AS', 'SDifferential']])
        else: # 如果计算的索引范围无效（例如，文件行数太少），则打印头部
             print(df[['HomeTeam','AwayTeam','FTR','HS', 'AS', 'SDifferential']].head())
    else:
        print("    DataFrame为空，无法显示结果。")


    # 保存修改后的DataFrame回原文件
    try:
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            print(f"\n  已将更新后的数据保存回: {full_file_path}")
        else:
            print(f"\n  DataFrame为空，未保存文件: {full_file_path}")
    except Exception as e_save:
        print(f"\n  保存文件 {full_file_path} 时出错: {e_save}") 