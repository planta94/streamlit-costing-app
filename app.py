import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import json
from io import BytesIO

st.set_page_config(page_title="Costing App", layout="wide")
st.title("📦 Costing Table")

FREE_CURRENCY_API_BASE_URL = "https://api.currencyfreaks.com/v2.0/rates/latest"
FREE_CURRENCY_API_KEY = (
    "5118283d99dd4cdc8a2aa4d9451db566"  # Store your API key in Streamlit secrets
)
if not FREE_CURRENCY_API_KEY:
    st.warning(
        "Please add your FreeCurrencyAPI key to Streamlit secrets for live exchange rates."
    )
    USE_LIVE_RATES = False
else:
    USE_LIVE_RATES = True

AVAILABLE_CURRENCIES = ["", "USD", "EUR", "JPY", "GBP", "CNY", "MYR"]
HARDCODED_EXCHANGE_RATES = {  # Fallback if API fails or no key
    "USD": 1.0,
    "EUR": 1.1,
    "JPY": 0.007,
    "GBP": 1.25,
    "CNY": 0.14,
    "MYR": 0.21,
}
DEFAULT_TARGET_CURRENCY = "MYR"

filtered_currencies = [c for c in AVAILABLE_CURRENCIES if c != ""]
default_index = (
    filtered_currencies.index(DEFAULT_TARGET_CURRENCY)
    if DEFAULT_TARGET_CURRENCY in filtered_currencies
    else 0
)

target_currency = st.selectbox(
    f"Target Currency (for Total Cost (Converted) calculation) {'- using live rates' if USE_LIVE_RATES else 'using fallback rates'}",
    filtered_currencies,
    index=default_index,
    key="target_currency_selector",
)

if "bom_df" not in st.session_state:
    st.session_state.bom_df = pd.DataFrame(
        columns=[
            "Part Number",
            "Description",
            "Quantity",
            "Unit Cost",
            "Unit Cost Currency",
            "Total Cost (Original)",
            "Total Cost (Converted)",
        ]
    )

print(st.session_state, "mcb")


@st.cache_data(ttl=timedelta(hours=1))
def fetch_exchange_rates(rates):
    if USE_LIVE_RATES and FREE_CURRENCY_API_KEY:
        try:
            params = {"apikey": FREE_CURRENCY_API_KEY, "symbols": ",".join(rates)}
            response = requests.get(FREE_CURRENCY_API_BASE_URL, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching live exchange rates: {e}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON response: {e}")
            return None
    return None


def calculate_conversion(df, target_currency):
    interested_rates = set(target_currency)
    for index, row in df.iterrows():
        interested_rates.add(row["Unit Cost Currency"])
    live_rates = (
        fetch_exchange_rates(rates=list(interested_rates)) if USE_LIVE_RATES else None
    )
    all_rates = live_rates if live_rates else HARDCODED_EXCHANGE_RATES

    def _convert_row(row):
        from_cur = row["Unit Cost Currency"]
        unit_cost_conv = np.nan
        if from_cur and from_cur in all_rates and target_currency in all_rates:
            try:
                unit_cost = (
                    float(row["Unit Cost"]) if pd.notna(row["Unit Cost"]) else 0.0
                )
                qty = float(row["Quantity"]) if pd.notna(row["Quantity"]) else 0.0
                rate = all_rates[from_cur] / all_rates[target_currency]
                unit_cost_conv = unit_cost * rate
                return unit_cost_conv * qty
            except ValueError:
                return np.nan
        return np.nan

    df["Total Cost (Converted)"] = df.apply(_convert_row, axis=1)
    df["Total Cost (Original)"] = df["Quantity"].astype(float) * df["Unit Cost"].astype(
        float
    )
    return df


def _update_bom_df_from_editor(current_df, editor_state):
    updated_rows_dict = editor_state.get("edited_rows", {})
    for index, changes in updated_rows_dict.items():
        for col, value in changes.items():
            if index < len(current_df):
                current_df.loc[index, col] = value

    deleted_indices = sorted(editor_state.get("deleted_rows", []), reverse=True)
    current_df = current_df.drop(current_df.index[deleted_indices], errors="ignore")
    return current_df


# Initialize edited_df using st.session_state
edited_df = st.data_editor(
    st.session_state.bom_df,  # Use bom_df directly as the initial data
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Part Number": st.column_config.TextColumn("Part Number"),
        "Description": st.column_config.TextColumn("Description"),
        "Quantity": st.column_config.NumberColumn("Quantity", min_value=0),
        "Unit Cost": st.column_config.NumberColumn("Unit Cost", min_value=0),
        "Unit Cost Currency": st.column_config.SelectboxColumn(
            "Unit Cost Currency", options=AVAILABLE_CURRENCIES
        ),
        "Total Cost (Original)": st.column_config.NumberColumn(
            "Total Cost (Original)", disabled=True, format="%.2f"
        ),
        "Total Cost (Converted)": st.column_config.NumberColumn(
            "Total Cost (Converted)", disabled=True, format="%.2f"
        ),
    },
    key="bom_editor",
    on_change=lambda: st.session_state.update(
        {
            "bom_df": (
                pd.concat(
                    [
                        st.session_state.bom_df,
                        pd.DataFrame(
                            st.session_state.bom_editor.get("added_rows", [])
                        ).reindex(columns=st.session_state.bom_df.columns),
                    ],
                    ignore_index=True,
                )
                if st.session_state.bom_editor.get("added_rows")
                else (
                    pd.DataFrame(st.session_state.bom_editor)
                    if isinstance(st.session_state.bom_editor, list)
                    else (
                        _update_bom_df_from_editor(
                            st.session_state.bom_df.copy(), st.session_state.bom_editor
                        )
                        if st.session_state.bom_editor.get("edited_rows")
                        or st.session_state.bom_editor.get("deleted_rows")
                        else st.session_state.bom_df
                    )
                )
            )
        }
    ),
)

# Determine button state *after* data editor
disable_calculate = True
can_download = False
has_edit = False
if edited_df is not None:  # Check if edited_df has data
    if not edited_df.empty:
        if all(
            pd.notna(row["Quantity"])
            and pd.notna(row["Unit Cost"])
            and row["Unit Cost Currency"] in AVAILABLE_CURRENCIES
            for index, row in edited_df.iterrows()
        ):
            disable_calculate = False
            can_download = True

if st.button("Calculate Costs", disabled=disable_calculate):
    print(st.session_state, "mcb")
    updated_bom_df = calculate_conversion(
        edited_df.copy(),  # Use the directly edited data
        st.session_state.get(
            "target_currency_selector",
            filtered_currencies[default_index] if filtered_currencies else "",
        ),
    )
    st.session_state["bom_df"] = (
        updated_bom_df  # Update session state with calculated values
    )
    st.rerun()
    # No st.rerun() here

try:
    total_bom_cost_converted = (
        st.session_state["bom_df"]["Total Cost (Converted)"]
        .replace("", np.nan)
        .astype(float)
        .sum(skipna=True)
    )
except Exception:
    total_bom_cost_converted = 0

st.subheader(
    f"💰 Total BOM Cost (Converted) in {target_currency}: {total_bom_cost_converted:,.2f}"
)

st.subheader("Export Document")


def download_button(df, file_name, button_text, file_format):
    if file_format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            data = [df.columns.tolist()] + df.values.tolist()

            table = Table(data)
            style = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
            table.setStyle(style)
            elements = []
            elements.append(table)
            doc.build(elements)

            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}.pdf">{button_text}</a>'
        except ImportError:
            st.warning(
                "Please install the 'reportlab' library to export as PDF: `pip install reportlab`"
            )
            return

    st.markdown(href, unsafe_allow_html=True)


import base64

if not st.session_state.bom_df.empty and can_download:
    download_button(st.session_state.bom_df, "bom", "Export as PDF", "pdf")
else:
    st.info("The BOM table is empty. Please add data to export.")
