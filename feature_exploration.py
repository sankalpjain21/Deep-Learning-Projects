#!/usr/bin/env python3
import pandas as pd
import numpy as np
# from tabulate import tabulate # No longer needed for main table
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tkinter as tk
from tkinter import filedialog

# --- Helper function to clean column names (Robustly) ---
def clean_col_names(df_to_clean):
    original_cols = list(df_to_clean.columns)
    temp_cleaned_cols = []
    for col_original_name in original_cols:
        c = str(col_original_name).strip()
        if c.startswith('\ufeff'):
            c = c.replace('\ufeff', '')
        temp_cleaned_cols.append(c)

    final_new_cols = []
    counts = {}
    for c_name in temp_cleaned_cols:
        if c_name not in counts:
            final_new_cols.append(c_name)
            counts[c_name] = 1
        else:
            suffixed_name = f"{c_name}.{counts[c_name]}"
            while suffixed_name in final_new_cols:
                counts[c_name] += 1
                suffixed_name = f"{c_name}.{counts[c_name]}"
            final_new_cols.append(suffixed_name)
            counts[c_name] += 1
    df_to_clean.columns = final_new_cols
    return df_to_clean

# --- Helper function to clean numeric/monetary strings ---
def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().replace('$', '').replace(',', '')
        if value == '-' or value == '':
            return np.nan
        if value.startswith('(') and value.endswith(')'):
            return -float(value[1:-1])
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan

# === Safe Division Helper Function (closer to original script's behavior) ===
def safe_div(numerator, denominator):
    # Prepare denominator for calculation: replace 0s with NaN
    if isinstance(denominator, (int, float)):
        original_denom_was_zero = (denominator == 0)
        denom_for_calc = np.nan if original_denom_was_zero else denominator
    elif isinstance(denominator, pd.Series):
        original_denom_was_zero = (denominator == 0) # Boolean Series
        denom_for_calc = denominator.copy()
        denom_for_calc[original_denom_was_zero] = np.nan
    else: # Should not happen if inputs are numbers or Series
        original_denom_was_zero = False
        denom_for_calc = denominator

    # Perform division, suppressing warnings as we handle outcomes
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denom_for_calc

    # Apply logic: 0/X = 0, but 0/0 (original) = NaN
    if isinstance(numerator, pd.Series):
        num_is_zero_mask = (numerator == 0)
        # Initially set 0/anything to 0 (this makes 0/NaN_orig = 0 and 0/NaN_from_zero = 0)
        result[num_is_zero_mask] = 0.0
        # Correct 0/0 (where original denominator was 0) back to NaN
        if isinstance(original_denom_was_zero, pd.Series): # Denominator was Series
            result[num_is_zero_mask & original_denom_was_zero] = np.nan
        elif original_denom_was_zero: # Denominator was scalar 0
            result[num_is_zero_mask] = np.nan # All entries where numerator is 0 become NaN
    elif isinstance(numerator, (int, float)) and numerator == 0:
        if original_denom_was_zero: # Scalar 0 / Scalar 0
            return np.nan
        else: # Scalar 0 / (Non-zero or original NaN)
            return 0.0
            
    return result


# === MAIN SCRIPT EXECUTION STARTS HERE ===
def main():
    root = tk.Tk()
    root.withdraw()
    csv_file_path = filedialog.askopenfilename(
        title="Select your data.csv file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if not csv_file_path:
        print("No file selected. Exiting.")
        return
    print(f"Loading '{csv_file_path}'. Processing...")

    try:
        df_original = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"Failed to read with utf-8-sig ({e}), trying default encoding...")
        try:
            df_original = pd.read_csv(csv_file_path)
        except Exception as e_fallback:
            print(f"Failed to read CSV with default encoding as well: {e_fallback}")
            return

    df = df_original.copy()
    df = clean_col_names(df)

    expected_script_cols = {
        'Price per Share': 'Price per Share', 'Market Cap': 'Market Cap', 'Sales': None,
        'Net Profit': None, 'Current Assets': 'Current Assets', 'Current Liabilities': 'Current Liabilities',
        'Quick Assets': 'Quick Assets', 'Cash': 'Cash', 'Total Debt': 'Total Debt',
        'Shareholder Equity': 'Shareholder Equity', 'Total Assets': 'Total Assets',
        'Earnings Per Share (EPS)': 'Earnings Per Share (EPS)', 'Gross Profit': 'Gross Profit',
        'Operating Income': 'Operating Income', 'Interest Expense': 'Interest Expense',
        'Operating Cash Flow': 'Operating Cash Flow',
        'Depreciation & Amortization': 'Depreciation & Amortization',
        'Total Debt Service': 'Total Debt Service',
        'Earnings Growth Rate (%)': 'Earnings Growth Rate (%)'
    }

    sales_cols_found = [col for col in df.columns if 'sales' in col.lower()]
    if sales_cols_found:
        preferred_sales = [s for s in sales_cols_found if s.lower() == 'sales' or s.lower() == 'revenue']
        if preferred_sales: expected_script_cols['Sales'] = preferred_sales[0]
        elif len(sales_cols_found) > 1 and 'ttm' not in sales_cols_found[0].lower() and 'ttm' in sales_cols_found[1].lower():
            expected_script_cols['Sales'] = sales_cols_found[1]
        else: expected_script_cols['Sales'] = sales_cols_found[0]
    if expected_script_cols['Sales'] is None: expected_script_cols['Sales'] = 'Sales_MISSING'

    net_profit_cols_found = [col for col in df.columns if 'net profit' in col.lower() or 'net income' in col.lower()]
    if net_profit_cols_found:
        preferred_net_profit = [s for s in net_profit_cols_found if s.lower() == 'net profit' or s.lower() == 'net income']
        if preferred_net_profit: expected_script_cols['Net Profit'] = preferred_net_profit[0]
        elif len(net_profit_cols_found) > 1 and 'ttm' not in net_profit_cols_found[0].lower() and 'ttm' in net_profit_cols_found[1].lower():
            expected_script_cols['Net Profit'] = net_profit_cols_found[1]
        else: expected_script_cols['Net Profit'] = net_profit_cols_found[0]
    if expected_script_cols['Net Profit'] is None: expected_script_cols['Net Profit'] = 'Net Profit_MISSING'

    cols_to_clean_in_df = []
    for script_col, actual_col_name in expected_script_cols.items():
        col_to_use = actual_col_name if actual_col_name and not actual_col_name.endswith("_MISSING") else script_col
        if col_to_use in df.columns:
            cols_to_clean_in_df.append(col_to_use)
    cols_to_clean_in_df = list(set(cols_to_clean_in_df))

    for col_name in cols_to_clean_in_df:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(clean_numeric_value)

    df_processed = pd.DataFrame(index=df.index)
    for script_name, actual_df_col_name in expected_script_cols.items():
        final_col_name_to_use = actual_df_col_name if actual_df_col_name and not actual_df_col_name.endswith("_MISSING") else script_name
        if final_col_name_to_use in df.columns:
            df_processed[script_name] = df[final_col_name_to_use]
        else:
            df_processed[script_name] = np.nan
            if not final_col_name_to_use.endswith("_MISSING"):
                 print(f"Warning: Column '{final_col_name_to_use}' (for script's '{script_name}') not found. Filled with NaN.")

    calc_cols = ['Market Cap', 'Sales', 'Net Profit', 'Current Assets', 'Current Liabilities',
                 'Quick Assets', 'Cash', 'Total Debt', 'Shareholder Equity', 'Total Assets',
                 'Earnings Per Share (EPS)', 'Gross Profit', 'Operating Income', 'Interest Expense',
                 'Operating Cash Flow', 'Depreciation & Amortization', 'Total Debt Service',
                 'Earnings Growth Rate (%)', 'Price per Share']
    for c in calc_cols:
        if c not in df_processed:
            df_processed[c] = np.nan

    df_processed['Price to Sales Ratio'] = safe_div(df_processed['Market Cap'], df_processed['Sales'])
    df_processed['Profit Margin Ratio'] = safe_div(df_processed['Net Profit'], df_processed['Sales']) * 100
    df_processed['Current Ratio'] = safe_div(df_processed['Current Assets'], df_processed['Current Liabilities'])
    df_processed['Quick Ratio'] = safe_div(df_processed.get('Quick Assets'), df_processed['Current Liabilities'])
    df_processed['Cash Ratio'] = safe_div(df_processed['Cash'], df_processed['Current Liabilities'])
    df_processed['Debt to Equity Ratio'] = safe_div(df_processed['Total Debt'], df_processed['Shareholder Equity'])
    df_processed['Debt Ratio'] = safe_div(df_processed['Total Debt'], df_processed['Total Assets'])
    df_processed['D/C Ratio'] = safe_div(df_processed['Total Debt'], (df_processed['Total Debt'] + df_processed['Shareholder Equity']))
    df_processed['Total Asset Turnover'] = safe_div(df_processed['Sales'], df_processed['Total Assets'])
    df_processed['P/E Ratio'] = safe_div(df_processed['Price per Share'], df_processed['Earnings Per Share (EPS)'])
    df_processed['Gross Profit Margin'] = safe_div(df_processed['Gross Profit'], df_processed['Sales']) * 100
    df_processed['Operating Profit Margin'] = safe_div(df_processed['Operating Income'], df_processed['Sales']) * 100
    df_processed['Net Profit Margin'] = safe_div(df_processed['Net Profit'], df_processed['Sales']) * 100
    df_processed['ROI'] = safe_div(df_processed['Net Profit'], df_processed['Total Assets']) * 100
    df_processed['EBIT'] = df_processed['Operating Income']
    dep_amort_val = df_processed.get('Depreciation & Amortization', pd.Series(0.0, index=df_processed.index)).fillna(0.0)
    df_processed['EBITDA'] = df_processed['Operating Income'] + dep_amort_val
    df_processed['ROE'] = safe_div(df_processed['Net Profit'], df_processed['Shareholder Equity']) * 100
    df_processed['ROCE'] = safe_div(df_processed['EBIT'], (df_processed['Total Assets'] - df_processed['Current Liabilities'])) * 100
    df_processed['ROA'] = safe_div(df_processed['Net Profit'], df_processed['Total Assets']) * 100
    df_processed['Interest Coverage Ratio (ICR)'] = safe_div(df_processed['EBIT'], df_processed['Interest Expense'])
    total_debt_service_val = df_processed.get('Total Debt Service', pd.Series(np.nan, index=df_processed.index))
    df_processed['DSCR'] = safe_div(df_processed.get('Operating Cash Flow'), total_debt_service_val)
    earnings_growth_val = df_processed.get('Earnings Growth Rate (%)', pd.Series(5.0, index=df_processed.index)).fillna(5.0)
    df_processed['PEG Ratio'] = safe_div(df_processed['P/E Ratio'], earnings_growth_val)
    df_processed['PB Ratio'] = safe_div(df_processed['Market Cap'], df_processed['Shareholder Equity'])

    nifty_50_avg_ratios = {
        'Price to Sales Ratio': 4.0, 'Profit Margin Ratio': 10.0, 'Current Ratio': 1.5,
        'Quick Ratio': 1.0, 'Cash Ratio': 0.3, 'Debt to Equity Ratio': 1.0, 'Debt Ratio': 0.4,
        'D/C Ratio': 0.5, 'Total Asset Turnover': 0.8, 'P/E Ratio': 25.0,
        'Gross Profit Margin': 35.0, 'Operating Profit Margin': 15.0, 'Net Profit Margin': 10.0,
        'ROE': 15.0, 'ROA': 5.0, 'Interest Coverage Ratio (ICR)': 4.0, 'PB Ratio': 3.0,
        'ROI': 5.0, 'ROCE': 12.0,
    }
    comparison_ratios_list = [
        'Price to Sales Ratio', 'Profit Margin Ratio', 'Current Ratio', 'Quick Ratio',
        'Cash Ratio', 'Debt to Equity Ratio', 'Debt Ratio', 'D/C Ratio',
        'Total Asset Turnover', 'P/E Ratio', 'Gross Profit Margin', 'Operating Profit Margin',
        'Net Profit Margin', 'ROI', 'ROE', 'ROA', 'ROCE',
        'Interest Coverage Ratio (ICR)', 'DSCR', 'PEG Ratio', 'PB Ratio'
    ]
    
    if df_processed.empty:
        company_values_for_comparison = [np.nan] * len(comparison_ratios_list)
    else:
        company_values_for_comparison = []
        for r in comparison_ratios_list:
            if r in df_processed and not df_processed[r].empty:
                company_values_for_comparison.append(df_processed[r].iloc[0])
            else:
                company_values_for_comparison.append(np.nan)

    comparison_df = pd.DataFrame({'Ratio': comparison_ratios_list, 'Company Value': company_values_for_comparison})
    comparison_df['Nifty 50 Avg'] = comparison_df['Ratio'].map(nifty_50_avg_ratios)

    def classify_and_remark(row):
        ratio_name = row['Ratio']
        company_value = row['Company Value']
        avg_value = row['Nifty 50 Avg']
        remark = 'Neutral'
        if pd.isna(company_value): return 'Data Missing'
        if ratio_name == 'Current Ratio':
            if company_value > 2.0: remark = 'Positive (Very Liquid)'
            elif company_value >= 1.3: remark = 'Positive (Liquid)'
            else: remark = 'Negative (Less Liquid)'
        elif ratio_name == 'Quick Ratio':
            if company_value >= 1.0: remark = 'Positive'
            else: remark = 'Negative'
        elif ratio_name == 'Cash Ratio':
            if 0.2 <= company_value <= 1.0: remark = 'Positive (Sufficient Cash)'
            elif company_value > 1.0: remark = 'Neutral (High Cash, could be inefficient)'
            else: remark = 'Negative (Low Cash)'
        elif ratio_name == 'Debt to Equity Ratio':
            if company_value < 0.5: remark = 'Very Positive (Low Debt)'
            elif company_value < 1.0: remark = 'Positive (Manageable Debt)'
            elif not pd.isna(avg_value) and company_value < avg_value : remark = 'Positive (Below Avg D/E)'
            else: remark = 'Negative (High Debt)'
        elif ratio_name == 'Debt Ratio':
            if company_value < 0.4: remark = 'Positive (Low Debt Burden)'
            elif not pd.isna(avg_value) and company_value < avg_value : remark = 'Positive (Below Avg Debt Ratio)'
            else: remark = 'Negative (High Debt Burden)'
        elif ratio_name == 'D/C Ratio':
            if company_value < 0.33: remark = 'Positive (Low Leverage)'
            elif not pd.isna(avg_value) and company_value < avg_value : remark = 'Positive (Below Avg D/C)'
            else: remark = 'Negative (High Leverage)'
        elif ratio_name == 'Interest Coverage Ratio (ICR)':
            if company_value > 5: remark = 'Very Positive (Strong Coverage)'
            elif company_value > 2.5: remark = 'Positive (Good Coverage)'
            elif not pd.isna(avg_value) and company_value > avg_value : remark = 'Positive (Above Avg ICR)'
            elif company_value <=1.5 and company_value > 0 : remark = 'Negative (Weak Coverage, Risky)'
            elif company_value <=0 : remark = 'Negative (Cannot Cover Interest)'
            else: remark = 'Negative (Weak Coverage)'
        elif ratio_name == 'DSCR':
            if pd.isna(company_value): return 'Data Missing for DSCR'
            if company_value > 1.5 : remark = 'Positive (Good DSCR)'
            elif company_value > 1.2: remark = 'Neutral (Acceptable DSCR)'
            else: remark = 'Negative (Poor DSCR, <1.2 may be risky)'
        elif ratio_name in ['Profit Margin Ratio', 'Gross Profit Margin', 'Operating Profit Margin',
                            'Net Profit Margin', 'Total Asset Turnover', 'ROE', 'ROA', 'ROI', 'ROCE']:
            if pd.isna(avg_value): return 'No Benchmark for ' + ratio_name
            if company_value > avg_value * 1.15: remark = 'Very Positive (Significantly Above Avg)'
            elif company_value > avg_value: remark = 'Positive (Above Avg)'
            elif company_value == avg_value: remark = 'Neutral (At Avg)'
            elif company_value < avg_value * 0.85 : remark = 'Negative (Significantly Below Avg)'
            else: remark = 'Negative (Below Avg)'
        elif ratio_name == 'Price to Sales Ratio':
            if company_value <= 0: remark = "Neutral (Unusual P/S <=0)"
            elif company_value < 1.0: remark = 'Very Positive (Potentially Undervalued on Sales)'
            elif not pd.isna(avg_value):
                if company_value < avg_value * 0.8: remark = 'Positive (Below Avg P/S)'
                elif company_value > avg_value * 1.5: remark = 'Negative (Potentially Overvalued on Sales)'
        elif ratio_name == 'P/E Ratio':
            if pd.isna(avg_value): return 'No Benchmark for P/E'
            if company_value <= 0: remark = "Negative (Loss Making or No Earnings)"
            elif company_value < avg_value * 0.8 and company_value > 0: remark = 'Positive (Potentially Undervalued)'
            elif company_value > avg_value * 1.5: remark = 'Negative (Potentially Overvalued)'
            else: remark = 'Neutral (Around Avg P/E)'
        elif ratio_name == 'PB Ratio':
            if pd.isna(avg_value): return 'No Benchmark for P/B'
            if company_value <= 0: remark = "Negative (Negative or Zero Book Value)"
            elif company_value < avg_value * 0.8 and company_value > 0 : remark = 'Positive (Potentially Undervalued on Book)'
            elif company_value > avg_value * 1.5: remark = 'Negative (Potentially Overvalued on Book)'
            else: remark = 'Neutral (Around Avg P/B)'
        elif ratio_name == 'PEG Ratio':
            if company_value <= 0: remark = 'Neutral (Cannot assess PEG <=0)'
            elif company_value < 0.8: remark = 'Very Positive (Attractive Growth/Value)'
            elif company_value < 1.2: remark = 'Positive (Fairly Valued for Growth)'
            else: remark = 'Negative (Potentially Overvalued for Growth, >1.2)'
        return remark

    comparison_df['Remarks'] = comparison_df.apply(classify_and_remark, axis=1)

    positive_count = comparison_df['Remarks'].str.contains('Positive', case=False, na=False).sum()
    negative_count = comparison_df['Remarks'].str.contains('Negative', case=False, na=False).sum()

    print("\n📋 Company Ratios vs Nifty 50 Averages (Data for Chart):")
    # Use to_string for a simpler table output
    print(comparison_df.to_string(index=False, float_format="%.2f"))
    
    print(f"\n📊 Summary of Remarks:\n{comparison_df['Remarks'].value_counts().to_string()}")
    print(f"\n📈 Total Positive Remarks: {positive_count}")
    print(f"📉 Total Negative Remarks: {negative_count}")
    # Removed: Positive Remarks Percentage printout

    output_df = df_original.copy()
    for col in df_processed.columns:
        output_df[col] = df_processed[col]
    output_filename = 'financial_ratios_calculated_full.csv'
    try:
        output_df.to_csv(output_filename, index=False)
        print(f"\n✅ Full DataFrame with calculated ratios saved to '{os.path.join(os.getcwd(), output_filename)}'")
    except Exception as e:
        print(f"\n❌ Error saving CSV: {e}")

    # --- VISUALIZATIONS ---
    sns.set_theme(style="whitegrid")

    # Graph 1: Company vs Nifty 50 Ratios
    plot_df_for_viz = comparison_df.dropna(subset=['Company Value', 'Nifty 50 Avg']).copy()
    if not plot_df_for_viz.empty:
        high_value_ratios = ['P/E Ratio', 'Interest Coverage Ratio (ICR)', 'PEG Ratio']
        percentage_ratios = ['Profit Margin Ratio', 'Gross Profit Margin', 'Operating Profit Margin',
                             'Net Profit Margin', 'ROE', 'ROA', 'ROI', 'ROCE']
        other_ratios = [r for r in plot_df_for_viz['Ratio'].unique() 
                        if r not in high_value_ratios and r not in percentage_ratios]

        def create_bar_plot(data_subset, title_suffix, use_symlog_scale=False):
            if data_subset.empty:
                print(f"No data to plot for: {title_suffix}")
                return
            fig_width = max(12, len(data_subset) * 0.8 + 2)
            fig, ax = plt.subplots(figsize=(fig_width, 7))
            x = np.arange(len(data_subset['Ratio']))
            width = 0.35
            rects1 = ax.bar(x - width/2, data_subset['Company Value'], width, label='Company Value', color='skyblue')
            rects2 = ax.bar(x + width/2, data_subset['Nifty 50 Avg'], width, label='Nifty 50 Avg', color='lightcoral')
            ax.set_ylabel('Ratio Value')
            title = f'Company Ratios vs Nifty 50 Averages ({title_suffix})'
            if use_symlog_scale:
                ax.set_yscale('symlog', linthresh=0.1, linscale=0.5, subs=None)
                title += ' - Symlog Scale'
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(data_subset['Ratio'], rotation=65, ha='right', fontsize=9)
            ax.legend()
            fig.tight_layout()
            plt.grid(True, axis='y', linestyle='--')
            plt.show()

        plot_df_other_viz = plot_df_for_viz[plot_df_for_viz['Ratio'].isin(other_ratios)]
        plot_df_percentage_viz = plot_df_for_viz[plot_df_for_viz['Ratio'].isin(percentage_ratios)]
        plot_df_high_value_viz = plot_df_for_viz[plot_df_for_viz['Ratio'].isin(high_value_ratios)]

        if not plot_df_other_viz.empty: create_bar_plot(plot_df_other_viz, "General Ratios")
        if not plot_df_percentage_viz.empty: create_bar_plot(plot_df_percentage_viz, "Percentage-Based Ratios", use_symlog_scale=True)
        if not plot_df_high_value_viz.empty: create_bar_plot(plot_df_high_value_viz, "Potentially High-Value Ratios", use_symlog_scale=True)
    else:
        print("\nℹ Not enough comparable data (Company Value and Nifty 50 Avg) to plot ratio bar charts.")

    # Graph 2: Counts of Positive and Negative Remarks
    remarks_summary_data = {'Remark Type': ['Positive', 'Negative'], 'Count': [positive_count, negative_count]}
    remarks_summary_df = pd.DataFrame(remarks_summary_data)

    if positive_count > 0 or negative_count > 0: # Only plot if there's data
        plt.figure(figsize=(6, 5))
        sns.barplot(x='Remark Type', y='Count', data=remarks_summary_df, palette={'Positive':'green', 'Negative':'red'}, hue='Remark Type', legend=False)
        plt.title('Counts of Positive and Negative Remarks')
        plt.ylabel('Count')
        plt.xlabel('') # Remark Type is clear from x-ticks
        plt.tight_layout()
        plt.show()
    else:
        print("\nℹ No positive or negative remarks to display in summary count plot.")

    # Removed: Pie chart and full remarks countplot

if __name__ == "__main__":
    main()