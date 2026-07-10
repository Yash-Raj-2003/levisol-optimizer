import streamlit as st
import pandas as pd
import pulp as pl
import re
import io
import os
import glob
import platform
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Levisol Network Optimizer", layout="wide", page_icon="🛢️")

# ==========================================
# SOLVER HELPER (Windows-on-ARM64 Fix)
# ==========================================
def get_cbc_solver(msg=False):
    """
    Returns a working CBC solver instance for PuLP.

    - On every normal platform (Linux/macOS/Windows-x64 -- including
      Streamlit Community Cloud, which runs Linux x64): just use the
      default PULP_CBC_CMD with no custom path.
    - On Windows-on-ARM64 only: PuLP has no bundled win/arm64 binary, so we
      point at the bundled win/i64 (x64) binary instead, which Windows 11
      on ARM runs fine via its built-in x64 emulation.
      IMPORTANT: recent PuLP versions raise
      "PulpSolverError: Use COIN_CMD if you want to set a path"
      if you pass `path=` to PULP_CBC_CMD. COIN_CMD accepts the exact same
      arguments and IS allowed to take a custom path, so we use that class
      instead whenever a custom path is required.
    """
    is_windows_arm = (
        platform.system() == "Windows"
        and platform.machine().lower() in ("arm64", "aarch64")
    )
    if not is_windows_arm:
        return pl.PULP_CBC_CMD(msg=msg)

    pulp_dir = os.path.dirname(pl.__file__)
    candidates = glob.glob(os.path.join(pulp_dir, "solverdir", "cbc", "win", "*", "cbc.exe"))
    candidates.sort(key=lambda p: "64" not in os.path.basename(os.path.dirname(p)))

    for cbc_exe in candidates:
        if os.path.isfile(cbc_exe):
            return pl.COIN_CMD(msg=msg, path=cbc_exe)

    raise RuntimeError(
        "No usable CBC solver binary found for Windows on ARM64.\n"
        "Run:  pip install pulp[cbc]   then use pl.COIN_CMD(msg=False) with no path,\n"
        "or deploy on Streamlit Community Cloud (Linux x64), which works with no changes."
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def map_pack_to_line(pack_string):
    pack_string = str(pack_string).upper()
    match = re.search(r'([\d\.]+)\s*(ML|L|LT|KG)', pack_string)
    if not match: return None
    val = float(match.group(1))
    unit = match.group(2)
    if unit == 'ML': val /= 1000.0
    if val <= 1.5: return '<=1.5 LT'
    elif 3 <= val <= 5.5: return '3- 5 LT'
    elif 7 <= val <= 22: return '7- 20 LT'
    elif 45 <= val <= 55: return '50 LT'
    elif 180 <= val <= 220: return '180- 210LT'
    return None

def sanitize_hub_cols(df):
    df.columns = ['MHW' if 'MHW' in str(c) else 'MHE' if 'MHE' in str(c) else c for c in df.columns]
    return df

def strip_cfa_suffix(x):
    x = str(x).strip()
    if x.upper().endswith(' CFA'):
        return x[:-4].strip()
    return x

def load_clean_sheet(xls, sheet_name, anchor_col):
    temp_df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    anchor_lower = anchor_col.strip().lower()

    def has_exact_match(row):
        return any(str(cell).strip().lower() == anchor_lower for cell in row.values)

    mask = temp_df.apply(has_exact_match, axis=1)
    if not mask.any():
        raise ValueError(f"CRITICAL ERROR: Anchor '{anchor_col}' not found in '{sheet_name}'")

    header_idx = mask.idxmax()
    return pd.read_excel(xls, sheet_name=sheet_name, header=header_idx)

@st.cache_data
def load_and_preprocess_data(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    df_plants = load_clean_sheet(xls, 'A - Plants & Production', 'Plant Code')
    df_p2h = load_clean_sheet(xls, 'B - Plant-Hub Transport', 'From Plant')
    df_h2c = load_clean_sheet(xls, 'C -Hub-CFA Transport', 'CFA')
    df_sku = load_clean_sheet(xls, 'D -SKU Portfolio+Penalty matrix', 'Pack size')
    df_inv = load_clean_sheet(xls, 'I - Expected opening Inventory', 'CFA')
    df_demand = load_clean_sheet(xls, 'J - Jan Forecast', 'CFA')

    # Force numeric coercion to turn text footers into NaNs, then drop them
    df_plants['Production Cost (₹/kl)'] = pd.to_numeric(df_plants['Production Cost (₹/kl)'], errors='coerce')
    df_plants = df_plants.dropna(subset=['Production Cost (₹/kl)'])

    df_sku['Penalty cost (per kL)'] = pd.to_numeric(df_sku['Penalty cost (per kL)'], errors='coerce')
    df_sku = df_sku.dropna(subset=['Penalty cost (per kL)'])

    jan_col_inv = [col for col in df_inv.columns if 'Jan' in str(col)][0]
    df_inv[jan_col_inv] = pd.to_numeric(df_inv[jan_col_inv], errors='coerce')
    df_inv = df_inv.dropna(subset=[jan_col_inv])

    jan_col_dem = [col for col in df_demand.columns if 'Jan' in str(col)][0]
    df_demand[jan_col_dem] = pd.to_numeric(df_demand[jan_col_dem], errors='coerce')
    df_demand = df_demand.dropna(subset=[jan_col_dem])

    # Normalize plant identifiers: Sheet B uses full Location names, everywhere
    # else uses Plant Code
    loc_to_code = df_plants.set_index('Location')['Plant Code'].to_dict()
    df_p2h['From Plant'] = df_p2h['From Plant'].map(loc_to_code).fillna(df_p2h['From Plant'])

    # Normalize CFA identifiers: Sheets I/J use "<City> CFA", Sheet C uses "<City>"
    df_inv['CFA'] = df_inv['CFA'].apply(strip_cfa_suffix)
    df_demand['CFA'] = df_demand['CFA'].apply(strip_cfa_suffix)

    df_p2h = sanitize_hub_cols(df_p2h)
    df_h2c = sanitize_hub_cols(df_h2c)

    df_p2h_melt = df_p2h.melt(id_vars=['From Plant'], var_name='Hub', value_name='Cost')
    df_p2h_melt['Cost'] = pd.to_numeric(df_p2h_melt['Cost'], errors='coerce')
    df_p2h_melt = df_p2h_melt.dropna(subset=['Cost'])

    df_h2c_melt = df_h2c.melt(id_vars=['CFA', 'Region'], var_name='Hub', value_name='Cost')
    df_h2c_melt['Cost'] = pd.to_numeric(df_h2c_melt['Cost'], errors='coerce')
    df_h2c_melt = df_h2c_melt.dropna(subset=['Cost'])

    # Drop hub-level pseudo-CFA rows ("Mother Hub West/East") -- not real CFAs
    valid_cfas = set(df_h2c_melt['CFA'].unique())
    df_inv = df_inv[df_inv['CFA'].isin(valid_cfas)]
    df_demand = df_demand[df_demand['CFA'].isin(valid_cfas)]

    data = {}
    data['cost_prod'] = df_plants.set_index('Plant Code')['Production Cost (₹/kl)'].to_dict()

    cap_line = {}
    for _, row in df_plants.iterrows():
        plant = row['Plant Code']
        for col in df_plants.columns:
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.isna(val): val = 0.0
            if '<=1.5' in str(col): cap_line[(plant, '<=1.5 LT')] = val
            elif '3-' in str(col) and '5' in str(col): cap_line[(plant, '3- 5 LT')] = val
            elif '7-' in str(col) and '20' in str(col): cap_line[(plant, '7- 20 LT')] = val
            elif '50' in str(col): cap_line[(plant, '50 LT')] = val
            elif '180' in str(col) and '210' in str(col): cap_line[(plant, '180- 210LT')] = val
    data['cap_line'] = cap_line

    data['cost_p2h'] = df_p2h_melt.set_index(['From Plant', 'Hub'])['Cost'].to_dict()
    data['cost_h2c'] = df_h2c_melt.set_index(['Hub', 'CFA'])['Cost'].to_dict()

    BIG_M = 10_000_000
    penalty_s = {}
    sku_line_map = {}
    sku_col = 'SKU' if 'SKU' in df_sku.columns else 'Product Name'

    for _, row in df_sku.iterrows():
        sku = row[sku_col]
        sku_line_map[sku] = map_pack_to_line(row.get('Pack size', ''))
        if str(row.get('Contractual?', '')).strip().upper().startswith('YES'):
            penalty_s[sku] = BIG_M
        else:
            penalty_s[sku] = row.get('Penalty cost (per kL)', 100000)

    data['penalty_s'] = penalty_s
    data['sku_line_map'] = sku_line_map
    data['inv_cfa'] = df_inv.set_index(['Product Name', 'CFA'])[jan_col_inv].to_dict()
    data['demand'] = df_demand.set_index(['Product Name', 'CFA'])[jan_col_dem].to_dict()

    data['Plants'] = df_plants['Plant Code'].dropna().unique().tolist()
    data['Hubs'] = df_p2h_melt['Hub'].dropna().unique().tolist()
    data['CFAs'] = df_h2c_melt['CFA'].dropna().unique().tolist()
    data['SKUs'] = df_sku[sku_col].dropna().unique().tolist()
    data['Lines'] = ['<=1.5 LT', '3- 5 LT', '7- 20 LT', '50 LT', '180- 210LT']

    return data

def run_optimization(data, freight_multiplier, demand_multiplier, ss_penalty_cfa):
    Plants, Hubs, CFAs, SKUs, Lines = data['Plants'], data['Hubs'], data['CFAs'], data['SKUs'], data['Lines']

    ss_hub = {(s, h): 50 for s in SKUs for h in Hubs}
    ss_cfa = {(s, c): 25 for s in SKUs for c in CFAs}

    prob = pl.LpProblem("Levisol_January_Plan", pl.LpMinimize)

    B = pl.LpVariable.dicts("Batch", [(s, p) for s in SKUs for p in Plants], lowBound=0, cat='Integer')
    X = pl.LpVariable.dicts("Flow_P2H", [(s, p, h) for s in SKUs for p in Plants for h in Hubs], lowBound=0, cat='Continuous')
    Y = pl.LpVariable.dicts("Flow_H2C", [(s, h, c) for s in SKUs for h in Hubs for c in CFAs], lowBound=0, cat='Continuous')
    ClosingInv = pl.LpVariable.dicts("ClosingInv", [(s, c) for s in SKUs for c in CFAs], lowBound=0, cat='Continuous')
    U = pl.LpVariable.dicts("Unmet_Demand", [(s, c) for s in SKUs for c in CFAs], lowBound=0, cat='Continuous')
    W = pl.LpVariable.dicts("SS_Shortfall_CFA", [(s, c) for s in SKUs for c in CFAs], lowBound=0, cat='Continuous')
    V = pl.LpVariable.dicts("SS_Shortfall_Hub", [(s, h) for s in SKUs for h in Hubs], lowBound=0, cat='Continuous')

    prob += (
        pl.lpSum(float(data['cost_prod'].get(p, 999999)) * 25 * B[s, p] for s in SKUs for p in Plants) +
        pl.lpSum((float(data['cost_p2h'].get((p, h), 999999)) * freight_multiplier) * X[s, p, h] for s in SKUs for p in Plants for h in Hubs) +
        pl.lpSum((float(data['cost_h2c'].get((h, c), 999999)) * freight_multiplier) * Y[s, h, c] for s in SKUs for h in Hubs for c in CFAs) +
        pl.lpSum(float(data['penalty_s'].get(s, 999999)) * U[s, c] for s in SKUs for c in CFAs) +
        pl.lpSum(50_000 * V[s, h] for s in SKUs for h in Hubs) +
        pl.lpSum(ss_penalty_cfa * W[s, c] for s in SKUs for c in CFAs)
    )

    for s in SKUs:
        for p in Plants:
            prob += pl.lpSum(X[s, p, h] for h in Hubs) == 25 * B[s, p]

    for p in Plants:
        for l in Lines:
            prob += pl.lpSum(25 * B[s, p] for s in SKUs if data['sku_line_map'].get(s) == l) <= data['cap_line'].get((p, l), 0)

    for s in SKUs:
        for h in Hubs:
            inbound_h = pl.lpSum(X[s, p, h] for p in Plants)
            outbound_h = pl.lpSum(Y[s, h, c] for c in CFAs)
            prob += (inbound_h - outbound_h + V[s, h]) >= ss_hub.get((s, h), 0)

    for s in SKUs:
        for c in CFAs:
            inbound_c = pl.lpSum(Y[s, h, c] for h in Hubs)
            opening_c = float(data['inv_cfa'].get((s, c), 0))
            fcst = float(data['demand'].get((s, c), 0)) * demand_multiplier

            prob += ClosingInv[s, c] - U[s, c] == opening_c + inbound_c - fcst
            prob += ClosingInv[s, c] + W[s, c] >= ss_cfa.get((s, c), 0)

    # Use the platform-safe solver getter instead of a bare prob.solve()
    prob.solve(get_cbc_solver(msg=False))

    prod_data, route_data, short_data = [], [], []
    for (s, p), var in B.items():
        if var.varValue and var.varValue > 0.001:
            prod_data.append({'SKU': s, 'Plant': p, 'Batches': var.varValue, 'Volume_kL': var.varValue * 25, 'Line': data['sku_line_map'].get(s)})

    for (s, p, h), var in X.items():
        if var.varValue and var.varValue > 0.001:
            route_data.append({'Leg': 'Plant-to-Hub', 'Origin': p, 'Dest': h, 'Volume_kL': var.varValue})

    for (s, h, c), var in Y.items():
        if var.varValue and var.varValue > 0.001:
            route_data.append({'Leg': 'Hub-to-CFA', 'Origin': h, 'Dest': c, 'Volume_kL': var.varValue})

    for (s, c), var in U.items():
        if var.varValue and var.varValue > 0.001:
            short_data.append({'CFA': c, 'Type': 'Lost Sales', 'Volume_kL': var.varValue})

    for (s, c), var in W.items():
        if var.varValue and var.varValue > 0.001:
            short_data.append({'CFA': c, 'Type': 'SS Depletion', 'Volume_kL': var.varValue})

    raw_cost = pl.value(prob.objective)
    total_cost = float(raw_cost) if raw_cost is not None else 0.0
    total_demand = sum(float(v) for v in data['demand'].values()) * demand_multiplier
    total_unmet = sum(float(v.varValue) for v in U.values() if v.varValue is not None and v.varValue > 0)
    fill_rate = ((total_demand - total_unmet) / total_demand * 100) if total_demand > 0 else 0

    df_prod = pd.DataFrame(prod_data) if len(prod_data) > 0 else pd.DataFrame(columns=['SKU', 'Plant', 'Batches', 'Volume_kL', 'Line'])
    df_route = pd.DataFrame(route_data) if len(route_data) > 0 else pd.DataFrame(columns=['Leg', 'Origin', 'Dest', 'Volume_kL'])
    df_short = pd.DataFrame(short_data) if len(short_data) > 0 else pd.DataFrame(columns=['CFA', 'Type', 'Volume_kL'])

    return df_prod, df_route, df_short, total_cost, fill_rate, total_unmet, data['cap_line']

# ==========================================
# STREAMLIT UI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/87/Castrol_logo.svg/1200px-Castrol_logo.svg.png", width=150)
st.sidebar.title("Control Room")

uploaded_file = st.sidebar.file_uploader("Upload Data (Baseline or Shock)", type=['xlsx'])
freight_surcharge = st.sidebar.slider("Freight Cost Surcharge (%)", 0, 100, 0, 5)
demand_surge = st.sidebar.slider("Network Demand Multiplier", 1.0, 2.0, 1.0, 0.1)
ss_policy = st.sidebar.selectbox("Safety Stock Ring-Fencing", ["Strict (High Penalty)", "Relaxed (Buffer Drain)"])

run_btn = st.sidebar.button("Execute Network Optimization", type="primary")

st.title("Levisol Supply Chain Control Tower")

if uploaded_file is None:
    st.info("👈 Please upload the master dataset in the sidebar to initialize the network.")
else:
    file_bytes = uploaded_file.getvalue()
    data = load_and_preprocess_data(file_bytes)

    if run_btn:
        with st.spinner("Compiling Network Variables & Executing MILP Solver..."):
            try:
                f_mult = 1 + (freight_surcharge / 100.0)
                ss_pen = 5_000 if "Relaxed" in ss_policy else 500_000
                df_prod, df_route, df_short, t_cost, f_rate, t_unmet, cap_dict = run_optimization(data, f_mult, demand_surge, ss_pen)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            st.markdown("### Executive KPIs")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Network OpEx", f"₹{t_cost:,.0f}")
            col2.metric("Overall Fill Rate", f"{f_rate:.2f}%")
            col3.metric("Lost Sales Volume", f"{t_unmet:,.0f} kL")

            df_prod_grouped = df_prod.groupby(['Plant', 'Line'])['Volume_kL'].sum().reset_index() if not df_prod.empty else pd.DataFrame()
            peak_util, peak_loc = 0, "Healthy"
            if not df_prod_grouped.empty:
                for _, row in df_prod_grouped.iterrows():
                    cap = cap_dict.get((row['Plant'], row['Line']), 1)
                    util = (row['Volume_kL'] / cap) * 100 if cap > 0 else 0
                    if util > peak_util:
                        peak_util = util
                        peak_loc = f"{row['Plant']} - {row['Line']}"
            col4.metric("Peak Line Strain", peak_loc, f"{peak_util:.1f}% Utilized", delta_color="off")

            st.divider()

            st.markdown("### Active Network Flow")
            if not df_route.empty:
                df_flow = df_route.groupby(['Origin', 'Dest'])['Volume_kL'].sum().reset_index()
                all_nodes = list(pd.concat([df_flow['Origin'], df_flow['Dest']]).unique())
                node_map = {node: i for i, node in enumerate(all_nodes)}

                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color="#2B2B2B"),
                    link=dict(
                        source=[node_map[src] for src in df_flow['Origin']],
                        target=[node_map[tgt] for tgt in df_flow['Dest']],
                        value=df_flow['Volume_kL'],
                        color="#00B140"
                    ))])
                fig_sankey.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=20))
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.warning("No routing volume generated.")

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### Line Capacity Utilization")
                if not df_prod.empty:
                    fig_bar = px.bar(df_prod_grouped, x='Plant', y='Volume_kL', color='Line',
                                     color_discrete_sequence=px.colors.qualitative.Prism)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No production volume.")

            with col_right:
                st.markdown("### CFA Shortfall Heatmap")
                if not df_short.empty:
                    df_heat = df_short.groupby(['CFA', 'Type'])['Volume_kL'].sum().reset_index()
                    df_heat_pivot = df_heat.pivot_table(index='Type', columns='CFA', values='Volume_kL', aggfunc='sum').fillna(0)
                    fig_heat = px.imshow(df_heat_pivot, text_auto=True, aspect="auto",
                                         color_continuous_scale=[[0, 'white'], [1, '#E31837']])
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.success("Zero Shortfalls Detected.")

            st.divider()

            st.markdown("### Extract Final Plan")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_prod.to_excel(writer, sheet_name='Production_Plan', index=False)
                df_route.to_excel(writer, sheet_name='Routing_Plan', index=False)
                df_short.to_excel(writer, sheet_name='Shortfall_Report', index=False)

            st.download_button(
                label="📥 Download Levisol January Plan (.xlsx)",
                data=output.getvalue(),
                file_name="Levisol_January_Plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
