import json
import os

file_path = 'Advanced_PPO_Model.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4 (index 3) is where the data preparation logic is.
new_code = '''import os

FRED_API_KEY = "9c3f2227440e6d8c815f7996a4d253b5" # <- 입력 필요 (추후 재수집 시)
download_start = '2009-01-01'
actual_start_date = '2010-01-01'
end_date = '2026-04-02'

data_cache_path = 'cached_raw_data.csv'

if os.path.exists(data_cache_path):
    print("📦 로컬 캐시 파일이 존재합니다! API를 호출하지 않고 'cached_raw_data.csv'에서 데이터를 불러옵니다...")
    df_raw_combined = pd.read_csv(data_cache_path, index_col=0, parse_dates=True)
else:
    print("🚀 로컬 데이터가 없습니다. API를 통해 데이터 수집을 시작합니다...")
    
    assets = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'SHV', 'GLD', 'DBC', 'VNQ', 'RSP']
    df_raw = yf.download(assets, start=download_start, end=end_date, progress=False)
    df_assets = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns else df_raw['Close']
    df_assets.dropna(inplace=True)

    macro_tickers = {'^VIX': 'VIX', 'DX-Y.NYB': 'DXY', 'KRW=X': 'USDKRW', 'CL=F': 'WTI_Oil', 'HG=F': 'Copper', 'GC=F': 'Gold'}
    df_macro_yf_raw = yf.download(list(macro_tickers.keys()), start=download_start, end=end_date, progress=False)
    df_macro_yf = df_macro_yf_raw['Adj Close'] if 'Adj Close' in df_macro_yf_raw.columns else df_macro_yf_raw['Close']
    df_macro_yf.rename(columns=macro_tickers, inplace=True)

    def get_fred_api_data(series_id, api_key):
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        response = requests.get(url)
        data = response.json().get('observations', [])
        if not data: return pd.Series(dtype=np.float64)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.set_index('date', inplace=True)
        return df['value']

    fred_series = {'DGS10': 'US10Y', 'DGS2': 'US2Y', 'DGS3MO': 'US3M', 'BAMLH0A0HYM2': 'HY_Spread', 'WM2NS': 'M2_Supply', 'WALCL': 'Fed_Balance', 'ICSA': 'Jobless_Claims'}
    fred_list = [get_fred_api_data(sid, FRED_API_KEY).rename(name) for sid, name in fred_series.items()]
    df_macro_fred = pd.concat(fred_list, axis=1).loc[download_start:end_date].ffill()

    df_assets.index = pd.to_datetime(df_assets.index).tz_localize(None).normalize()
    df_macro_yf.index = pd.to_datetime(df_macro_yf.index).tz_localize(None).normalize()
    df_macro_fred.index = pd.to_datetime(df_macro_fred.index).tz_localize(None).normalize()
    df_raw_combined = pd.concat([df_assets, df_macro_yf, df_macro_fred], axis=1, join='inner').ffill()
    
    # 📌 새로 수집한 데이터 저장!
    df_raw_combined.to_csv(data_cache_path)
    print("💾 모든 수집이 완료되었습니다. 다음부터는 API 호출 없이 빠르게 로드하도록 'cached_raw_data.csv'에 저장했습니다!")

# --- 이후 피처 엔지니어링 ---
print("⚙️ 피처 엔지니어링 및 8대 핵심 센서 스케일링을 진행합니다...")
df = df_raw_combined.copy()
df['Equity_vs_Bond'] = df['SPY'] / df['EEM']
df['Market_Breadth'] = df['RSP'] / df['SPY']
df['spread_10y2y'] = df['US10Y'] - df['US2Y']
df['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']
df['Jobless_Claims_MA'] = df['Jobless_Claims'].rolling(window=20).mean()

df.dropna(inplace=True)
df = df.loc[actual_start_date:]

final_8_features = ["Equity_vs_Bond", "US10Y", "Jobless_Claims_MA", "VIX", "Market_Breadth", "spread_10y2y", "DXY", "Copper_Gold_Ratio"]
scaler = StandardScaler()
df_final_scaled = df.copy()
df_final_scaled[final_8_features] = scaler.fit_transform(df[final_8_features])
X_hmm = df_final_scaled[final_8_features]

print("✅ 데이터 병합 및 최종 예측용 8개 센서 준비 완료!")
'''

nb['cells'][3]['source'] = [line + '\n' for line in new_code.split('\n')[:-1]] + [new_code.split('\n')[-1]]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
