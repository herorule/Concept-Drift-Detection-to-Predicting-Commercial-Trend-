import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from river import drift, linear_model, preprocessing, optim, compose
from river.drift import ADWIN, KSWIN, PageHinkley
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import ast
from pandas.tseries.offsets import DateOffset
import traceback
import threading

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
pd.options.mode.chained_assignment = None

class UILogger:
    def __init__(self, result_text_widget, status_label_widget=None, root_window=None, drift_whiteboards=None):
        self.result_text = result_text_widget
        self.status_label = status_label_widget
        self.root = root_window
        self.drift_whiteboards = drift_whiteboards if drift_whiteboards else {} # Store the whiteboards
        self._configure_tags()

    def _configure_tags(self):
        self.result_text.tag_config("header", font=("Arial", 11, "bold"), foreground="#003399", spacing1=5, spacing3=5)
        self.result_text.tag_config("subheader", font=("Arial", 10, "bold"), foreground="#0055AA", spacing1=3, spacing3=3)
        self.result_text.tag_config("error", foreground="red", font=("Arial", 9, "italic"))
        self.result_text.tag_config("warning", foreground="#E69138", font=("Arial", 9, "italic"))
        self.result_text.tag_config("info", font=("Arial", 9))
        self.result_text.tag_config("step", font=("Arial", 10, "bold"), spacing1=3)
        self.result_text.tag_config("calc", font=("Courier New", 9), foreground="navy")
        self.result_text.tag_config("param", font=("Courier New", 9), foreground="purple")

    def log(self, message, tag="info", status_update=None, see_end=True):
        self.result_text.insert(tk.END, f"  {message}\n", tag)
        if see_end: self.result_text.see(tk.END)
        if status_update and self.status_label: self.status_label.config(text=status_update)
        elif self.status_label and tag=="info" and len(message) < 80:
            self.status_label.config(text=message.split('\n')[0])
        if self.root: self.root.update_idletasks()

    def log_header(self, message, status_update=None): self.log(message, tag="header", status_update=status_update if status_update else message)
    def log_subheader(self, message, status_update=None): self.log(message, tag="subheader", status_update=status_update if status_update else message)
    def log_step(self, step_number, message_text, status_override=None):
        self.result_text.insert(tk.END, f"{step_number}. {message_text}...\n", "step")
        self.result_text.see(tk.END)
        if self.status_label: self.status_label.config(text=status_override if status_override else f"Step {step_number}: {message_text}...")
        if self.root: self.root.update_idletasks()
    def clear(self): self.result_text.delete("1.0", tk.END)
    def log_error(self, message, status_update="Error occurred."): self.log(message, tag="error", status_update=status_update)
    def log_warning(self, message, status_update="Warning."): self.log(message, tag="warning", status_update=status_update)
    def log_calculation(self, message, status_update=None): self.log(message, tag="calc", status_update=status_update)
    def log_parameters(self, params_dict):
        self.log("Using Drift Detection Parameters:", tag="subheader")
        for key, value in params_dict.items():
            self.log(f"  - {key}: {value}", tag="param")
        self.log("", see_end=False)

    def clear_drift_whiteboards(self):
        """Clears whiteboards before a new run."""
        if self.drift_whiteboards:
            for board in self.drift_whiteboards.values():
                board.delete("1.0", tk.END)
            self.log("Drift whiteboards cleared.", tag="info")

    def log_to_whiteboards(self, step_result):
        """Logs a drift event to the whiteboard."""
        if not self.drift_whiteboards:
            return

        metrics = step_result['metrics']
        msg = (
            f"Date: {step_result['index'].strftime('%Y-%m-%d')}\n"
            f"  - Actual:    {step_result['actual']:>10.2f}\n"
            f"  - Predicted: {step_result['predicted']:>10.2f}\n"
            f"  - Abs Error: {step_result['abs_error']:>10.2f}\n"
            f"--- Running Metrics ---\n"
            f"  - MAE:   {metrics['MAE']:.2f} ({metrics['%MAE']:.1f}%)\n"
            f"  - RMSE:  {metrics['RMSE']:.2f} ({metrics['%RMSE']:.1f}%)\n"
            f"-----------------------\n\n"
        )
        
        for detector_name, has_drifted in step_result['drift_status'].items():
            if has_drifted:
                if detector_name in self.drift_whiteboards:
                    board = self.drift_whiteboards[detector_name]
                    board.insert(tk.END, msg)
                    board.see(tk.END)
        
        if self.root:
            self.root.update_idletasks()

class DataHandler:
    def __init__(self, logger):
        self.logger = logger
        self.df_raw = None
        self.df_exploded = None

    def load_data_file(self, file_path):
        self.logger.log(f"Loading data from: {file_path.split('/')[-1]}", status_update="Loading data...")
        self.df_raw = pd.read_csv(file_path, dtype={'Product': str})
        if self.df_raw.empty: raise ValueError("The selected CSV file is empty.")
        raw_rows, raw_cols = self.df_raw.shape
        raw_total_cost = self.df_raw['Total_Cost'].sum() if 'Total_Cost' in self.df_raw.columns else 0
        self.logger.log(f"Raw data loaded: {raw_rows} rows, {raw_cols} columns. Grand Total_Cost: ${raw_total_cost:,.2f}")
        required_cols = ['Date', 'Total_Cost', 'Product']
        for col in required_cols:
            if col not in self.df_raw.columns: raise ValueError(f"Essential column '{col}' not found.")
        try:
            self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'])
            self.df_raw.set_index('Date', inplace=True); self.df_raw.sort_index(inplace=True)
            self.logger.log("'Date' column processed and set as index.")
        except Exception as e: raise ValueError(f"Error parsing 'Date' column or setting index: {e}")
        return self.df_raw

    def explode_products(self, df_input):
        self.logger.log("Preprocessing 'Product' column...")
        processed_rows = []
        invalid_formats = 0
        for index, row in df_input.iterrows():
            products_str = row['Product']
            if pd.isna(products_str): invalid_formats += 1; continue
            if not isinstance(products_str, str) or not products_str.strip().startswith('['):
                invalid_formats += 1; continue
            try:
                products = ast.literal_eval(products_str.strip())
                if isinstance(products, list):
                    if not products: invalid_formats +=1; continue
                    for product in products:
                        new_row = row.drop('Product').to_dict()
                        new_row['Product_Name'] = str(product).strip()
                        new_row['Date'] = index
                        processed_rows.append(new_row)
                else: invalid_formats += 1
            except (ValueError, SyntaxError): invalid_formats += 1
            except Exception: invalid_formats += 1
        if invalid_formats > 0: self.logger.log_warning(f"Skipped {invalid_formats} rows in 'Product' processing (NaN, empty, or malformed).")
        if not processed_rows: raise ValueError("No valid product data from explosion. All rows were skipped or resulted in no products.")
        self.df_exploded = pd.DataFrame(processed_rows)
        if 'Date' not in self.df_exploded.columns: raise ValueError("Date column lost after product processing.")
        try:
            self.df_exploded['Date'] = pd.to_datetime(self.df_exploded['Date'])
            self.df_exploded.set_index('Date', inplace=True); self.df_exploded.sort_index(inplace=True)
        except Exception as e: raise ValueError(f"Error setting index for exploded data: {e}")
        self.logger.log(f"Exploded product data: {len(self.df_exploded)} itemized product rows.")
        return self.df_exploded

class CostModeler:
    MIN_SAMPLES_FOR_INITIAL_TRAIN = 10 

    def __init__(self, logger):
        self.logger = logger
        self.features_for_model_template = ['Year', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Total_Cost_Lag1', 'drift_flag']
        self.target_col = 'Total_Cost'
        self.actual_features_used = []
        self.drift_manager = DriftDetectorManager(self.logger)
        # Initialize the model pipeline
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))
        )

    def aggregate_cost(self, df_raw, frequency):
        self.logger.log(f"Aggregating '{self.target_col}' by {frequency}...")
        resample_freq_map = {'Daily': 'D', 'Weekly': 'W-SUN', 'Monthly': 'MS'} 
        resample_freq = resample_freq_map.get(frequency)
        if not resample_freq: raise ValueError(f"Invalid frequency: {frequency}")
        aggregated_cost_data = df_raw[self.target_col].resample(resample_freq).sum().fillna(0).to_frame()
        num_periods_aggregated = len(aggregated_cost_data)
        total_cost_aggregated = aggregated_cost_data[self.target_col].sum() if not aggregated_cost_data.empty else 0
        self.logger.log(f"'{self.target_col}' aggregated: {num_periods_aggregated} periods. Total cost for these periods: ${total_cost_aggregated:,.2f}")
        if len(aggregated_cost_data) < self.MIN_SAMPLES_FOR_INITIAL_TRAIN : self.logger.log_warning("Few data points after aggregation for robust modeling.")
        return aggregated_cost_data

    def engineer_features(self, aggregated_cost_data):
        self.logger.log("Engineering features for the cost model...")
        cost_feature_data = aggregated_cost_data.copy()
        if cost_feature_data.empty: raise ValueError("Cannot engineer features on empty aggregated data.")
        cost_feature_data['Year'] = cost_feature_data.index.year
        cost_feature_data['Month'] = cost_feature_data.index.month
        cost_feature_data['DayOfWeek'] = cost_feature_data.index.dayofweek
        cost_feature_data['DayOfYear'] = cost_feature_data.index.dayofyear
        cost_feature_data['WeekOfYear'] = cost_feature_data.index.isocalendar().week.astype(int)
        if len(cost_feature_data) > 1:
            cost_feature_data[self.target_col + '_Lag1'] = cost_feature_data[self.target_col].shift(1)
            cost_feature_data.dropna(inplace=True)
        else: self.logger.log_warning("Not enough data for lagged features.")
        
        # Initialize the drift_flag feature with 0
        cost_feature_data['drift_flag'] = 0
        
        rows_for_modeling = len(cost_feature_data)
        self.logger.log(f"Cost feature engineering complete. Rows available for modeling (after lag): {rows_for_modeling}.")
        if rows_for_modeling < 2: raise ValueError("Insufficient data after feature engineering for model training (need at least 2 rows).")
        return cost_feature_data

    def split_data(self, cost_feature_data, split_ratio=0.6):
        self.logger.log(f"Splitting data ({int(split_ratio*100)}% train, {int((1-split_ratio)*100)}% test)...")
        min_data_for_meaningful_split = 3
        if len(cost_feature_data) < min_data_for_meaningful_split:
            raise ValueError(f"Not enough data ({len(cost_feature_data)}) for train/test split. Need at least {min_data_for_meaningful_split}.")
        min_data_for_ratio_split = 5 
        if len(cost_feature_data) < min_data_for_ratio_split:
            split_index = len(cost_feature_data) - 1 
            if split_index < 1: split_index = 1 
            self.logger.log_warning(f"Few data points ({len(cost_feature_data)}). Adjusting split: {split_index} train, {len(cost_feature_data)-split_index} test.")
        else:
            split_index = int(len(cost_feature_data) * split_ratio)
            if split_index >= len(cost_feature_data): split_index = len(cost_feature_data) - 1 
            if split_index == 0 and len(cost_feature_data) > 1 : split_index = 1 
            elif split_index == 0 and len(cost_feature_data) <=1 : raise ValueError("Cannot split data with 1 or 0 rows.")
        train_data = cost_feature_data.iloc[:split_index]; test_data = cost_feature_data.iloc[split_index:]
        if train_data.empty or test_data.empty:
                 raise ValueError(f"Train ({len(train_data)}) or Test ({len(test_data)}) set empty after split.")
        
        # Update feature list logic to include the new drift_flag
        self.actual_features_used = [f for f in self.features_for_model_template if f in train_data.columns and f in test_data.columns]
        lag_feature_name = self.target_col + '_Lag1'
        if lag_feature_name not in self.actual_features_used and lag_feature_name in train_data.columns and lag_feature_name in test_data.columns :
                 self.actual_features_used.append(lag_feature_name)
        self.actual_features_used = list(set(self.actual_features_used))
        if 'drift_flag' not in self.actual_features_used:
             self.actual_features_used.append('drift_flag')

        if not self.actual_features_used or lag_feature_name not in self.actual_features_used :
            raise ValueError(f"Not enough valid features (especially '{lag_feature_name}') for training after split.")
        total_cost_train = train_data[self.target_col].sum()
        avg_cost_train = train_data[self.target_col].mean() if not train_data.empty else 0
        total_cost_test = test_data[self.target_col].sum()
        avg_cost_test = test_data[self.target_col].mean() if not test_data.empty else 0
        self.logger.log(f"Data split complete. Train: {len(train_data)} rows (Total Cost: ${total_cost_train:,.2f}, Avg Cost: ${avg_cost_train:,.2f}), "
                        f"Test: {len(test_data)} rows (Total Cost: ${total_cost_test:,.2f}, Avg Cost: ${avg_cost_test:,.2f}). "
                        f"Features: {self.actual_features_used}")
        return train_data, test_data

    def train_model_initial(self, train_data):
        self.logger.log(f"Initial training for Online Model ({type(self.model).__name__})...")
        if train_data.empty:
                 self.logger.log_warning(f"Training data is empty. Online model will start untrained.")
                 return self.model
        if len(train_data) < self.MIN_SAMPLES_FOR_INITIAL_TRAIN:
                 self.logger.log_warning(f"Training data has only {len(train_data)} samples (less than {self.MIN_SAMPLES_FOR_INITIAL_TRAIN}). Model might be weak initially.")
        
        X_train_dict = train_data[self.actual_features_used].to_dict(orient='records')
        y_train_series = train_data[self.target_col]
        
        for x, y in zip(X_train_dict, y_train_series):
            try: self.model.learn_one(x, y)
            except Exception as e: self.logger.log_error(f"Error during online model learn_one (initial train): {e} with features {x}")
        self.logger.log("Online model initial training/burn-in complete.")
        return self.model


class DriftDetectorManager:
    def __init__(self, logger):
        self.logger = logger
        self.adwin = None
        self.kswin = None
        self.ph = None
        self.detectors = {}

    def initialize_detectors(self, params):
        """Initializes or re-initializes detectors for a new run."""
        self.adwin = ADWIN(delta=params['adwin_delta'])
        self.kswin = KSWIN(alpha=params['kswin_alpha'], window_size=params['kswin_window_size'], stat_size=params['kswin_stat_size'])
        self.ph = PageHinkley(min_instances=params['ph_min_instances'], delta=params['ph_delta'], threshold=params['ph_threshold'])
        self.detectors = {"ADWIN": self.adwin, "KSWIN": self.kswin, "PageHinkley": self.ph}
        self.logger.log("Drift detectors have been initialized.", tag="info")

    def update_and_check_all(self, error_value):
        """Updates all detectors with a new error value and returns their drift status."""
        if pd.isna(error_value):
            return {name: False for name in self.detectors}
            
        drift_status = {}
        for name, detector in self.detectors.items():
            try:
                detector.update(error_value)
                drift_status[name] = detector.drift_detected
            except Exception as e:
                self.logger.log_warning(f"Error in {name} detector: {e}")
                drift_status[name] = False
        return drift_status
    
class TrendAnalyzer:
    def __init__(self, logger):
        self.logger = logger

    def analyze_product_trends(self, df_exploded, frequency, top_n=5, detected_drift_points=None, title_prefix=""):
        adjusted_period = False
        if df_exploded is None or df_exploded.empty: 
            self.logger.log_warning(f"{title_prefix}No data provided for trend analysis. Skipping.")
            return pd.DataFrame(), pd.DataFrame(), adjusted_period
        end_date = df_exploded.index.max()
        offset, period_unit, period_current_start = None, None, None
        if frequency == 'Daily':
            offset = pd.Timedelta(days=7); period_unit = pd.Timedelta(days=1)
            period_current_start = end_date - offset + period_unit
        elif frequency == 'Weekly':
            offset = DateOffset(weeks=1); period_unit = pd.Timedelta(days=7)
            period_current_start = (end_date - pd.offsets.Week(weekday=6)).normalize()
            if period_current_start > end_date : period_current_start = period_current_start - pd.Timedelta(days=7)
        elif frequency == 'Monthly':
            offset = DateOffset(months=1); period_unit = DateOffset(months=1)
            period_current_start = end_date.normalize().replace(day=1)
        else:
            self.logger.log_error(f"{title_prefix}Invalid frequency '{frequency}' for trend analysis.")
            return pd.DataFrame(), pd.DataFrame(), adjusted_period
        if period_current_start > end_date:
            if frequency == 'Daily': period_current_start = end_date - pd.Timedelta(days=6)
            elif frequency == 'Weekly': period_current_start = (end_date - pd.offsets.Week(weekday=6)).normalize()
        period_current_end = end_date
        last_relevant_drift_point = None
        if detected_drift_points:
            relevant_drifts = [pd.to_datetime(p) for p in detected_drift_points if pd.to_datetime(p) < period_current_start]
            if relevant_drifts: last_relevant_drift_point = max(relevant_drifts)
        period_previous_start, period_previous_end = None, None
        if last_relevant_drift_point:
            if isinstance(period_unit, pd.Timedelta): period_previous_start = last_relevant_drift_point + period_unit
            elif isinstance(period_unit, DateOffset):
                temp_start = last_relevant_drift_point + period_unit
                if frequency == 'Monthly': period_previous_start = temp_start.replace(day=1)
                else: period_previous_start = temp_start
            else: period_previous_start = last_relevant_drift_point + pd.Timedelta(days=1)
            period_previous_end = period_current_start - pd.Timedelta(days=1)
            if period_previous_start <= period_previous_end:
                adjusted_period = True; self.logger.log(f"{title_prefix}Previous period adjusted by drift at {last_relevant_drift_point.date()}.", tag="info")
            else:
                adjusted_period = False; self.logger.log(f"{title_prefix}Drift too recent. Using standard previous period.", tag="info")
        if not adjusted_period:
            period_previous_end = period_current_start - pd.Timedelta(days=1)
            period_previous_start = period_current_start - offset
            if frequency == 'Weekly': period_previous_start = (period_previous_end - pd.offsets.Week(weekday=6)).normalize()
            elif frequency == 'Monthly':
                temp_prev_start = (period_current_start - DateOffset(months=1)).replace(day=1)
                if temp_prev_start > period_previous_end: temp_prev_start = (period_current_start - DateOffset(months=2)).replace(day=1)
                period_previous_start = temp_prev_start
        if not (period_current_start > period_current_end or period_previous_start > period_previous_end):
             self.logger.log(f"{title_prefix}Comparing: Current ({period_current_start.date()} to {period_current_end.date()}) vs. "
                             f"Previous ({period_previous_start.date()} to {period_previous_end.date()}).", tag="info")
        current_period_data = df_exploded[(df_exploded.index >= period_current_start) & (df_exploded.index <= period_current_end)]
        previous_period_data = df_exploded[(df_exploded.index >= period_previous_start) & (df_exploded.index <= period_previous_end)]
        if current_period_data.empty: self.logger.log_warning(f"{title_prefix}No data in current period for trend analysis.")
        if previous_period_data.empty: self.logger.log_warning(f"{title_prefix}No data in previous period for trend analysis.")
        current_counts = current_period_data.groupby('Product_Name').size() if not current_period_data.empty else pd.Series(dtype=int)
        previous_counts = previous_period_data.groupby('Product_Name').size() if not previous_period_data.empty else pd.Series(dtype=int)
        trend_df = pd.DataFrame({'Current_Quantity': current_counts, 'Previous_Quantity': previous_counts}).fillna(0).astype(int)
        trend_df['Quantity_Change'] = trend_df['Current_Quantity'] - trend_df['Previous_Quantity']
        top_current_selling = trend_df.sort_values('Current_Quantity', ascending=False).head(top_n)
        top_growth = trend_df[(trend_df['Quantity_Change'] > 0) & (trend_df['Current_Quantity'] > 0)].sort_values('Quantity_Change', ascending=False).head(top_n)
        self.logger.log_subheader(f"{title_prefix}Trending Products (Top {top_n})")
        log_output_trends = f"{title_prefix}  >> Top Selling (Recent Period):\n"
        if not top_current_selling.empty:
            for name, row_data in top_current_selling.iterrows(): log_output_trends += f"{title_prefix}      - {name}: {int(row_data['Current_Quantity'])} units\n"
        else: log_output_trends += f"{title_prefix}      (N/A)\n"
        log_output_trends += f"\n{title_prefix}  >> Top Growth (vs Previous Period {'Adjusted' if adjusted_period else 'Standard'}):\n"
        if not top_growth.empty:
            for name, row_data in top_growth.iterrows(): log_output_trends += f"{title_prefix}      - {name}: Change {int(row_data['Quantity_Change']):+} (Now: {int(row_data['Current_Quantity'])})\n"
        else: log_output_trends += f"{title_prefix}      (N/A)\n"
        self.logger.log(log_output_trends.strip(), tag="info", see_end=False)
        self.logger.log(f"{title_prefix}Product trend analysis complete.", tag="info")
        return top_current_selling, top_growth, adjusted_period

    def analyze_trends_around_drift(self, df_exploded, combined_drift_points, frequency, top_n=3):
        if df_exploded is None or df_exploded.empty: self.logger.log_warning("Skipping analysis around drift: Missing product data."); return
        if not combined_drift_points: self.logger.log("No confirmed drift points for around-drift analysis.", tag="info"); return
        self.logger.log_header(f"Analyzing Trends Around Drift Points (Top {top_n} Changes)")
        window_offset = None
        if frequency == 'Daily': window_offset = pd.Timedelta(days=7)
        elif frequency == 'Weekly': window_offset = DateOffset(weeks=1)
        elif frequency == 'Monthly': window_offset = DateOffset(months=1)
        else: self.logger.log_error(f"Invalid or unsupported frequency '{frequency}' for analysis around drift."); return
        processed_drifts_count = 0
        for i, drift_point_ts in enumerate(combined_drift_points):
            drift_point = pd.to_datetime(drift_point_ts)
            self.logger.log_subheader(f"Analysis around Drift Point {i+1} ({drift_point.date()})")
            before_end = drift_point - pd.Timedelta(days=1); before_start = before_end - window_offset + pd.Timedelta(days=1)
            after_start = drift_point; after_end = after_start + window_offset - pd.Timedelta(days=1)
            if isinstance(window_offset, DateOffset):
                if frequency == 'Weekly':
                    before_end_normalized = (drift_point - pd.offsets.Week(weekday=6)).normalize() - pd.Timedelta(days=1)
                    if before_end_normalized >= drift_point : before_end_normalized -= pd.Timedelta(days=7)
                    before_start_normalized = before_end_normalized - window_offset + pd.Timedelta(days=1)
                    after_start_normalized = (drift_point - pd.offsets.Week(weekday=6)).normalize()
                    if after_start_normalized < drift_point: after_start_normalized += pd.Timedelta(days=7)
                    after_end_normalized = after_start_normalized + window_offset - pd.Timedelta(days=1)
                    if before_start_normalized <= before_end_normalized and after_start_normalized <= after_end_normalized:
                        before_start, before_end = before_start_normalized, before_end_normalized
                        after_start, after_end = after_start_normalized, after_end_normalized
                elif frequency == 'Monthly':
                    before_end_normalized = drift_point.replace(day=1) - pd.Timedelta(days=1)
                    before_start_normalized = (before_end_normalized - window_offset + pd.Timedelta(days=1)).replace(day=1)
                    after_start_normalized = drift_point.replace(day=1)
                    if after_start_normalized < drift_point : after_start_normalized = (after_start_normalized + DateOffset(months=1)).replace(day=1)
                    after_end_normalized = (after_start_normalized + window_offset).replace(day=1) - pd.Timedelta(days=1)
                    if before_start_normalized <= before_end_normalized and after_start_normalized <= after_end_normalized:
                        before_start, before_end = before_start_normalized, before_end_normalized
                        after_start, after_end = after_start_normalized, after_end_normalized
            before_data = df_exploded[(df_exploded.index >= before_start) & (df_exploded.index <= before_end)]
            after_data = df_exploded[(df_exploded.index >= after_start) & (df_exploded.index <= after_end)]
            if before_data.empty or after_data.empty: self.logger.log_warning(f"Not enough data for drift at {drift_point.date()}. Skipping."); continue
            processed_drifts_count +=1
            before_counts = before_data.groupby('Product_Name').size(); after_counts = after_data.groupby('Product_Name').size()
            compare_df = pd.DataFrame({'Before_Drift': before_counts, 'After_Drift': after_counts}).fillna(0).astype(int)
            compare_df['Change'] = compare_df['After_Drift'] - compare_df['Before_Drift']
            top_increase = compare_df[compare_df['Change'] > 0].sort_values('Change', ascending=False).head(top_n)
            top_decrease = compare_df[compare_df['Change'] < 0].sort_values('Change', ascending=True).head(top_n)
            self.logger.log(f"Comparing window for drift at {drift_point.date()}: "
                            f"[{before_start.date()} to {before_end.date()}] vs [{after_start.date()} to {after_end.date()}]", tag="info")
            log_output = ""
            if not top_increase.empty:
                log_output += "  Top Increases:\n"
                for name, row_data in top_increase.iterrows(): log_output += f"    - {name}: +{int(row_data['Change'])} (B: {int(row_data['Before_Drift'])}, A: {int(row_data['After_Drift'])})\n"
            else: log_output += "  No significant increases.\n"
            if not top_decrease.empty:
                log_output += "  Top Decreases:\n"
                for name, row_data in top_decrease.iterrows(): log_output += f"    - {name}: {int(row_data['Change'])} (B: {int(row_data['Before_Drift'])}, A: {int(row_data['After_Drift'])})\n"
            else: log_output += "  No significant decreases.\n"
            self.logger.log(log_output.strip(), tag="info")
        if processed_drifts_count == 0 and combined_drift_points: self.logger.log_warning("Could not analyze around any drift points.")
        elif processed_drifts_count > 0: self.logger.log("--- End of Analysis Around Drift Points ---", tag="info", see_end=True)

class AnalysisController:
    """Orchestrates the entire analysis workflow."""
    def __init__(self, app_ref, logger):
        self.app_ref = app_ref; self.logger = logger
        self.data_handler = DataHandler(logger)
        self.cost_modeler = CostModeler(logger)
        self.trend_analyzer = TrendAnalyzer(logger)
        self._reset_state()

    def _reset_state(self):
        """Resets internal state and re-initializes modeler for a fresh run."""
        self.df_raw = None; self.df_exploded = None; self.aggregated_cost_data = None; self.cost_feature_data = None
        self.train_data = None; self.test_data = None; self.y_train = pd.Series(dtype=float); self.y_test = pd.Series(dtype=float)
        self.test_pred_series = pd.Series(dtype=float); self.mae = 0.0; self.rmse = 0.0; self.combined_drift_points = []
        self.top_selling_initial = pd.DataFrame(); self.top_growth_initial = pd.DataFrame()

    def run_analysis_workflow(self, file_path, frequency, drift_params):
        """Executes the full analysis pipeline."""
        self._reset_state()
        self.cost_modeler = CostModeler(self.logger)
        self.logger.clear()
        self.logger.log_header("Starting Full Analysis Workflow (Online Learning Model)", status_update="Analysis in progress...")
        
        try:
            if not file_path: raise ValueError("No data file selected.")
            self.logger.log(f"File: {file_path.split('/')[-1]} | Frequency: {frequency}", tag="info")
            self.logger.log_parameters(drift_params)

            self.logger.log_step(1, "Loading and Initial Preprocessing", status_override="Loading data...")
            self.df_raw = self.data_handler.load_data_file(file_path)
            
            self.logger.log_step(2, "Exploding Product Data", status_override="Processing products...")
            self.df_exploded = self.data_handler.explode_products(self.df_raw.copy())

            self.logger.log_step(3, "Aggregating Total Cost", status_override="Aggregating cost...")
            self.aggregated_cost_data = self.cost_modeler.aggregate_cost(self.df_raw, frequency)

            self.logger.log_step(4, "Feature Engineering", status_override="Engineering features...")
            self.cost_feature_data = self.cost_modeler.engineer_features(self.aggregated_cost_data)

            self.logger.log_step(5, "Splitting Data (70/30)", status_override="Splitting data...")
            self.train_data, self.test_data = self.cost_modeler.split_data(self.cost_feature_data)
            self.y_train = self.train_data[self.cost_modeler.target_col] 
            
            self.logger.log_step(6, "Initial Training of Online Model", status_override="Initial model training...")
            self.cost_modeler.train_model_initial(self.train_data)

            self.logger.log_step(7, "Prequential Prediction, Evaluation & Live Drift Detection", status_override="Prequential evaluation...")
            self.logger.clear_drift_whiteboards()
            self.combined_drift_points = []
            y_test_actual_list, test_pred_series_list, test_indices_list = [], [], []
            y_true_history, y_pred_history = [], []
            min_votes = drift_params['min_votes']
            
            # Initialize the drift flag state
            drift_flag_value = 0
            self.cost_modeler.drift_manager.initialize_detectors(drift_params)

            X_test_dict = self.test_data[self.cost_modeler.actual_features_used].to_dict(orient='records')
            y_test_series_actual = self.test_data[self.cost_modeler.target_col]

            for i in range(len(X_test_dict)):
                x_instance = X_test_dict[i]
                y_actual_instance = y_test_series_actual.iloc[i]
                instance_index = y_test_series_actual.index[i]

                # Add the current drift flag state to the instance features
                x_instance['drift_flag'] = drift_flag_value

                # 1. PREDICT
                y_pred_instance = self.cost_modeler.model.predict_one(x_instance) or 0
                
                # Update history for metrics and plotting
                y_true_history.append(y_actual_instance)
                y_pred_history.append(y_pred_instance)
                y_test_actual_list.append(y_actual_instance)
                test_pred_series_list.append(y_pred_instance)
                test_indices_list.append(instance_index)

                # 2. CALCULATE METRICS
                mae = mean_absolute_error(y_true_history, y_pred_history)
                rmse = np.sqrt(mean_squared_error(y_true_history, y_pred_history))
                mean_actual = np.mean(y_true_history)
                percent_mae = (mae / mean_actual * 100) if mean_actual != 0 else 0
                percent_rmse = (rmse / mean_actual * 100) if mean_actual != 0 else 0

                # 3. CHECK FOR DRIFT
                current_abs_error = abs(y_actual_instance - y_pred_instance)
                drift_statuses = self.cost_modeler.drift_manager.update_and_check_all(current_abs_error)
                
                vote_count = sum(1 for detected in drift_statuses.values() if detected)
                voted_drift = vote_count >= min_votes

                if voted_drift:
                    if instance_index not in self.combined_drift_points:
                        self.combined_drift_points.append(instance_index)
                    drift_statuses['Voting System'] = True
                    # Set the flag to 1 for all subsequent steps
                    if drift_flag_value == 0:
                        self.logger.log(f"Voted drift confirmed at {instance_index.strftime('%Y-%m-%d')}. Activating drift flag.", tag="warning")
                        drift_flag_value = 1
                else:
                    drift_statuses['Voting System'] = False

                # 4. LOG TO WHITEBOARDS
                if any(drift_statuses.values()):
                    step_result = {
                        "index": instance_index, "actual": y_actual_instance, "predicted": y_pred_instance,
                        "abs_error": current_abs_error, "drift_status": drift_statuses,
                        "metrics": {"MAE": mae, "RMSE": rmse, "%MAE": percent_mae, "%RMSE": percent_rmse}
                    }
                    self.logger.log_to_whiteboards(step_result)

                # 5. LEARN
                self.cost_modeler.model.learn_one(x_instance, y_actual_instance)

            self.y_test = pd.Series(y_test_actual_list, index=test_indices_list, name='Actual_Cost')
            self.test_pred_series = pd.Series(test_pred_series_list, index=test_indices_list, name='Predicted_Cost_Online')

            if not self.y_test.empty:
                self.mae = mean_absolute_error(self.y_test, self.test_pred_series)
                self.rmse = np.sqrt(mean_squared_error(self.y_test, self.test_pred_series))
                self.logger.log_subheader(f"\nFinal Test Set Metrics: MAE={self.mae:.2f}, RMSE={self.rmse:.2f}")

            self.logger.log_step(8, "Initial Global Product Trends", status_override="Analyzing initial trends...")
            if self.df_exploded is not None and not self.df_exploded.empty:
                self.logger.log_subheader("Initial Global Trends")
                self.top_selling_initial, self.top_growth_initial, _ = self.trend_analyzer.analyze_product_trends(
                    self.df_exploded, frequency, top_n=5, detected_drift_points=None, title_prefix="  [Global]" )
            else:
                self.logger.log_warning("Skipping initial product trend analysis: No exploded product data.")
            
            self.logger.log_step(9, "Drift-Adjusted Global Trends", status_override="Analyzing adjusted trends...")
            if self.df_exploded is not None and not self.df_exploded.empty:
                if not self.combined_drift_points: self.logger.log("No drifts to adjust global trends. Using initial results.", tag="info")
                else:
                    self.trend_analyzer.analyze_product_trends(self.df_exploded, frequency, top_n=5, detected_drift_points=self.combined_drift_points, title_prefix="  [Global - Drift-Adjusted]")
            else:
                self.logger.log_warning("Skipping drift-adjusted global trends: No exploded data.")
            
            self.logger.log_step(10, "Trends Around Specific Drift Points", status_override="Analyzing trends around drifts...")
            if self.df_exploded is not None and not self.df_exploded.empty:
                    self.trend_analyzer.analyze_trends_around_drift(self.df_exploded, self.combined_drift_points, frequency, top_n=3)
            else:
                self.logger.log_warning("Skipping analysis around drifts: No exploded data.")
            
            self.logger.log_step(11, "Plotting Results", status_override="Plotting...")
            if self.y_train.empty or self.y_test.empty or self.test_pred_series.empty:
                    self.logger.log_warning("Skipping plot: Insufficient data.")
            else:
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(self.train_data.index, self.y_train, label='Actual Cost (Train)', color='blue', lw=1.2, alpha=0.8) 
                ax.plot(self.y_test.index, self.y_test, label='Actual Cost (Test)', color='darkorange', lw=1.5)
                ax.plot(self.test_pred_series.index, self.test_pred_series, label='Predicted Cost (Test)', color='green', ls='--', lw=1.5)
                if self.combined_drift_points:
                    first_drift_line = True
                    for dp_timestamp in self.combined_drift_points:
                        ax.axvline(dp_timestamp, color='red', ls=':', lw=1.8, label='Voted Drift Point' if first_drift_line else ""); first_drift_line = False
                ax.set_xlabel('Time', fontsize=12); ax.set_ylabel('Total Sales ($)', fontsize=12)
                ax.set_title(f'Total Cost Online Prediction ({frequency.capitalize()}) & Voted Concept Drift Points', fontsize=14)
                ax.legend(fontsize=10); ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45); plt.tight_layout()
                for widget in self.app_ref.plot_frame_container.winfo_children(): widget.destroy()
                canvas = FigureCanvasTkAgg(fig, master=self.app_ref.plot_frame_container)
                canvas_widget = canvas.get_tk_widget(); canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                toolbar_frame = ttk.Frame(self.app_ref.plot_frame_container); toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame); toolbar.update()
                canvas.draw()
                self.logger.log("Plot generated successfully.", tag="info")
            
            self.logger.log_header("Analysis Workflow Complete", status_update="Analysis complete. Ready.")
        except ValueError as ve:
            self.logger.log_error(f"ValueError: {str(ve)}", status_update="Data/Logic Error!")
            messagebox.showerror("ValueError", f"A data or logical error occurred:\n{str(ve)}")
        except Exception as e:
            detailed_traceback = traceback.format_exc()
            self.logger.log_error(f"Unexpected error: {type(e).__name__} - {str(e)}\nTraceback:\n{detailed_traceback}")
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {type(e).__name__} - {str(e)}")


class App:
    def __init__(self, master_root):
        self.root = master_root
        self.root.title("Concept Drift & Product Trend Analysis Application")
        self.root.attributes('-fullscreen', True); self.root.resizable(False, False)
        
        self.current_file_path = None
        self._define_parameter_presets()
        self.drift_params = self.PRESET_PACKS[1].copy() 
        
        self._create_widgets()
        
        self.logger = UILogger(self.result_text, self.status_label, self.root, self.drift_whiteboards)
        self.analysis_controller = AnalysisController(self, self.logger)
        
        self.browse_button.config(command=self._browse_file)
        self.run_button_full.config(command=self._start_analysis_thread)
        self.settings_button.config(command=self._open_settings_window)

    def _define_parameter_presets(self):
        """Defines 4 preset parameter packs for the settings slider."""
        self.PRESET_NAMES = ["Custom", "Baseline", "Sensitive", "Conservative", "Fast-React"]
        self.PRESET_PACKS = [
            {},
            { 
                "adwin_delta": 0.002, "kswin_alpha": 0.005,
                "kswin_window_size": 100, "kswin_stat_size": 30,
                "ph_min_instances": 30, "ph_delta": 0.005,
                "ph_threshold": 50, "min_votes": 2
            },
            { 
                "adwin_delta": 0.0005, "kswin_alpha": 0.01,
                "kswin_window_size": 100, "kswin_stat_size": 30,
                "ph_min_instances": 25, "ph_delta": 0.001,
                "ph_threshold": 30, "min_votes": 2
            },
            {
                "adwin_delta": 0.01, "kswin_alpha": 0.001,
                "kswin_window_size": 150, "kswin_stat_size": 45,
                "ph_min_instances": 30, "ph_delta": 0.01,
                "ph_threshold": 75, "min_votes": 2
            },
            { 
                "adwin_delta": 0.001, "kswin_alpha": 0.01,
                "kswin_window_size": 50, "kswin_stat_size": 15,
                "ph_min_instances": 20, "ph_delta": 0.005,
                "ph_threshold": 40, "min_votes": 1
            }
        ]

    def _create_widgets(self):
        title_label = ttk.Label(self.root, text="Concept Drift & Product Trend Analysis", font=("Arial", 18, "bold")); title_label.pack(pady=10, side=tk.TOP)
        
        top_controls_frame = ttk.Frame(self.root); top_controls_frame.pack(padx=10, pady=5, fill="x", side=tk.TOP)
        data_frame = ttk.LabelFrame(top_controls_frame, text="1. Data Input & Aggregation"); data_frame.pack(pady=5, fill="x", expand=True) 
        
        file_path_label = ttk.Label(data_frame, text="Data File Path (CSV):"); file_path_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_entry = ttk.Entry(data_frame, width=60); self.file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        data_frame.grid_columnconfigure(1, weight=1)
        self.browse_button = ttk.Button(data_frame, text="Browse..."); self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        frequency_label = ttk.Label(data_frame, text="Aggregation Freq:"); frequency_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.frequency_combobox = ttk.Combobox(data_frame, values=["Daily", "Weekly", "Monthly"], state="readonly", width=15)
        self.frequency_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="w"); self.frequency_combobox.set("Weekly")
        
        action_buttons_frame = ttk.Frame(self.root); action_buttons_frame.pack(pady=10, fill="x")
        
        self.settings_button = ttk.Button(action_buttons_frame, text="Drift Detection Settings...")
        self.settings_button.pack(side=tk.LEFT, padx=(20, 10), ipady=5)
        
        self.run_button_full = ttk.Button(action_buttons_frame, text="Run Full Analysis"); 
        self.run_button_full.pack(side=tk.LEFT, padx=10, ipady=5, expand=True)
        
        exit_button = ttk.Button(action_buttons_frame, text="Exit Application", command=self.root.destroy); 
        exit_button.pack(side=tk.RIGHT, padx=20, ipady=5, expand=True)
        
        main_content_frame = ttk.Frame(self.root)
        main_content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        main_paned_window = ttk.PanedWindow(main_content_frame, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill="both", expand=True)

        # Left side: Plot
        self.plot_frame_container = ttk.LabelFrame(main_paned_window, text="'Total_Cost' Prediction & Drift Visualization")
        plot_placeholder_label = ttk.Label(self.plot_frame_container, text="Plot will appear here after analysis.", font=("Arial", 11, "italic"))
        plot_placeholder_label.pack(padx=20, pady=20, expand=True, anchor="center")
        main_paned_window.add(self.plot_frame_container, weight=3)

        # Right side: Another paned window for logs
        right_pane = ttk.PanedWindow(main_paned_window, orient=tk.VERTICAL)
        main_paned_window.add(right_pane, weight=2)

        # Top-Right: The analysis log
        result_frame = ttk.LabelFrame(right_pane, text="Analysis Log")
        result_text_scrollbar_y = ttk.Scrollbar(result_frame, orient="vertical")
        self.result_text = tk.Text(result_frame, wrap="word", yscrollcommand=result_text_scrollbar_y.set, font=("Courier New", 9))
        result_text_scrollbar_y.config(command=self.result_text.yview)
        result_text_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill="both", expand=True)
        right_pane.add(result_frame, weight=1) # Log gets less space

        # Bottom-Right: Frame for the four new whiteboards
        drift_events_frame = ttk.LabelFrame(right_pane, text="Drift Event Whiteboards")
        right_pane.add(drift_events_frame, weight=2) # Whiteboards get more space

        # Create the 4 whiteboards in a 2x2 grid
        self.drift_whiteboards = {}
        whiteboard_names = {
            "ADWIN": (0, 0), "KSWIN": (0, 1),
            "PageHinkley": (1, 0), "Voting System": (1, 1)
        }

        for name, (r, c) in whiteboard_names.items():
            board_frame = ttk.LabelFrame(drift_events_frame, text=f"{name} Events")
            board_frame.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            drift_events_frame.grid_rowconfigure(r, weight=1)
            drift_events_frame.grid_columnconfigure(c, weight=1)
            
            scrollbar = ttk.Scrollbar(board_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            text_widget = tk.Text(board_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier New", 9), bg="#FFFFFF")
            text_widget.pack(side=tk.LEFT, fill="both", expand=True)
            scrollbar.config(command=text_widget.yview)
            
            self.drift_whiteboards[name] = text_widget
        
        status_bar_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2); status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(status_bar_frame, text="Ready", anchor=tk.W); self.status_label.pack(fill=tk.X, padx=5)

    def _browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            self.file_path_entry.delete(0, tk.END); self.file_path_entry.insert(0, file_path)
            self.current_file_path = file_path
            if hasattr(self, 'logger') and self.logger: self.logger.log(f"File selected: {file_path.split('/')[-1]}", status_update=f"Selected: {file_path.split('/')[-1]}")
        else: self.current_file_path = None

    def _open_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Drift Detection Settings")
        settings_window.transient(self.root); settings_window.grab_set()
        
        slider_frame = ttk.Frame(settings_window, padding=10); slider_frame.pack(fill="x")
        
        self.preset_name_label = ttk.Label(slider_frame, text="", font=("Arial", 10, "bold"))
        self.preset_name_label.pack()
        
        current_preset_index = self.PRESET_NAMES.index("Custom") # Default to custom unless a preset matches
        for i, preset in enumerate(self.PRESET_PACKS):
            if i > 0 and preset == self.drift_params: current_preset_index = i; break
        
        self.slider = ttk.Scale(settings_window, from_=0, to=4, orient="horizontal", command=self._on_slider_change)
        self.slider.set(current_preset_index)
        self.slider.pack(fill="x", expand=True, pady=5, padx=10)
        self.preset_name_label.config(text=f"Preset: {self.PRESET_NAMES[current_preset_index]}")

        params_frame = ttk.LabelFrame(settings_window, text="Parameters", padding=10)
        params_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.param_entries = {}
        param_layout = {
            "ADWIN Delta:": "adwin_delta", "Min Votes:": "min_votes",
            "KSWIN Alpha:": "kswin_alpha", "KSWIN Window:": "kswin_window_size", "KSWIN Stat Size:": "kswin_stat_size",
            "PH Min Inst:": "ph_min_instances", "PH Delta:": "ph_delta", "PH Threshold:": "ph_threshold"
        }
        row_num = 0; col_num = 0
        for label_text, key in param_layout.items():
            ttk.Label(params_frame, text=label_text).grid(row=row_num, column=col_num, padx=5, pady=3, sticky="w")
            self.param_entries[key] = ttk.Entry(params_frame, width=12); self.param_entries[key].grid(row=row_num, column=col_num+1, padx=5, pady=3, sticky="w")
            self.param_entries[key].bind("<KeyRelease>", self._on_manual_param_change)
            col_num += 2
            if col_num >= 6: row_num += 1; col_num = 0
        
        self._populate_settings_fields(self.drift_params)

        button_bar = ttk.Frame(settings_window); button_bar.pack(pady=10)
        ttk.Button(button_bar, text="Apply & Close", command=lambda: self._apply_settings(settings_window)).pack(side="left", padx=10)
        ttk.Button(button_bar, text="Cancel", command=settings_window.destroy).pack(side="left", padx=10)
        
    def _on_slider_change(self, value_str):
        preset_index = round(float(value_str))
        self.preset_name_label.config(text=f"Preset: {self.PRESET_NAMES[preset_index]}")
        if preset_index > 0:
            preset_params = self.PRESET_PACKS[preset_index]
            self._populate_settings_fields(preset_params)
    
    def _on_manual_param_change(self, event=None):
        self.slider.set(0)
        self.preset_name_label.config(text=f"Preset: {self.PRESET_NAMES[0]}")

    def _populate_settings_fields(self, params_dict):
        for key, entry_widget in self.param_entries.items():
            entry_widget.delete(0, tk.END)
            if key in params_dict: entry_widget.insert(0, str(params_dict[key]))

    def _apply_settings(self, settings_window):
        try:
            new_params = {
                "adwin_delta": float(self.param_entries['adwin_delta'].get()),
                "kswin_alpha": float(self.param_entries['kswin_alpha'].get()),
                "kswin_window_size": int(self.param_entries['kswin_window_size'].get()),
                "kswin_stat_size": int(self.param_entries['kswin_stat_size'].get()),
                "ph_min_instances": int(self.param_entries['ph_min_instances'].get()),
                "ph_delta": float(self.param_entries['ph_delta'].get()),
                "ph_threshold": int(self.param_entries['ph_threshold'].get()),
                "min_votes": int(self.param_entries['min_votes'].get())
            }
            self.drift_params = new_params
            self.logger.log_header("Drift detection parameters updated.", status_update="Settings updated.")
            settings_window.destroy()
        except ValueError: messagebox.showerror("Input Error", "Invalid number format for one or more parameters.", parent=settings_window)
        except Exception as e: messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=settings_window)

    def _start_analysis_thread(self):
        file_path = self.current_file_path if hasattr(self, 'current_file_path') and self.current_file_path else self.file_path_entry.get()
        frequency = self.frequency_combobox.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a data file first.")
            if hasattr(self, 'logger') and self.logger: self.logger.log_error("Analysis not started: No file selected.")
            return

        self.run_button_full.config(state="disabled")
        self.settings_button.config(state="disabled")
        
        def analysis_task():
            try:
                self.analysis_controller.run_analysis_workflow(file_path, frequency, self.drift_params)
            finally:
                self.root.after(0, self._finalize_analysis_ui)

        analysis_thread = threading.Thread(target=analysis_task)
        analysis_thread.daemon = True; analysis_thread.start()
        
    def _finalize_analysis_ui(self):
        """Updates UI after the analysis thread is complete. Runs in the main thread."""
        self.run_button_full.config(state="normal")
        self.settings_button.config(state="normal")

if __name__ == '__main__':
    try:
        from ttkthemes import ThemedTk
        root_tk_instance = ThemedTk(theme="radiance")
    except ImportError:
        print("ttkthemes library not found. Using default Tkinter theme.")
        root_tk_instance = tk.Tk()
    
    app = App(root_tk_instance)
    root_tk_instance.mainloop()