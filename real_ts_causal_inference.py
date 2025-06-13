"""
真实TabPFN-TS和Do-PFN模型的时间序列因果推断实现
需要确保项目中有相关的模型文件和依赖
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('tabpfn-time-series-main')
sys.path.append('Do-PFN-main')

try:
    # TabPFN-TS 导入
    from tabpfn_time_series.features import (
        RunningIndexFeature, 
        CalendarFeature, 
        AutoSeasonalFeature,
        PeriodicSinCosineFeature,
        FeatureTransformer
    )
    from tabpfn_time_series.predictor import TabPFNTimeSeriesPredictor, TabPFNMode
    from autogluon.timeseries import TimeSeriesDataFrame
    
    # Do-PFN 导入
    from scripts.transformer_prediction_interface.base import DoPFNRegressor
    
    REAL_MODELS_AVAILABLE = True
    print("✅ 真实模型依赖加载成功")
    
except ImportError as e:
    print(f"⚠️ 真实模型依赖加载失败: {e}")
    print("将使用模拟版本进行演示")
    REAL_MODELS_AVAILABLE = False

class RealTimeSeriesCausalInference:
    """使用真实TabPFN-TS和Do-PFN的时间序列因果推断类"""
    
    def __init__(self, use_real_models: bool = True):
        self.use_real_models = use_real_models and REAL_MODELS_AVAILABLE
        
        if self.use_real_models:
            # 初始化真实模型
            self._init_real_models()
        else:
            # 使用模拟模型
            self._init_mock_models()
    
    def _init_real_models(self):
        """初始化真实模型"""
        try:
            # 初始化TabPFN-TS特征转换器
            self.feature_transformer = FeatureTransformer([
                RunningIndexFeature(),
                CalendarFeature(components=['year', 'month', 'day']),
                AutoSeasonalFeature(),
            ])
            
            # 初始化Do-PFN回归器
            self.dopfn = DoPFNRegressor()
            
            print("✅ 真实模型初始化成功")
            
        except Exception as e:
            print(f"❌ 真实模型初始化失败: {e}")
            print("切换到模拟模型")
            self.use_real_models = False
            self._init_mock_models()
    
    def _init_mock_models(self):
        """初始化模拟模型（当真实模型不可用时）"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        self.feature_transformer = MockFeatureTransformer()
        self.dopfn = MockDoPFNWrapper()
        print("🔄 使用模拟模型")
    
    def generate_realistic_timeseries(self, 
                                    n_days: int = 365,
                                    intervention_probability: float = 0.2,
                                    base_intervention_effect: float = 5.0,
                                    effect_decay: float = 0.8,
                                    seasonal_components: List[int] = [7, 30, 365],
                                    noise_level: float = 1.0) -> pd.DataFrame:
        """生成更真实的时间序列数据"""
        
        # 创建时间索引
        start_date = pd.Timestamp('2023-01-01')
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        # 基础趋势（非线性）
        t = np.arange(n_days)
        trend = 10 + 0.01 * t + 0.0001 * t**2 - 0.000001 * t**3
        
        # 多重季节性
        seasonal = np.zeros(n_days)
        for period in seasonal_components:
            amplitude = 5.0 / period * 100  # 调整振幅
            seasonal += amplitude * np.sin(2 * np.pi * t / period)
        
        # 生成干预（考虑时间依赖性）
        treatment = np.zeros(n_days)
        for i in range(n_days):
            # 周末更容易有干预
            base_prob = intervention_probability
            if dates[i].weekday() >= 5:  # 周末
                base_prob *= 1.5
            
            # 月初更容易有干预
            if dates[i].day <= 5:
                base_prob *= 1.3
                
            treatment[i] = np.random.binomial(1, min(base_prob, 0.8))
        
        # 协变量
        day_of_week = np.array([d.weekday() for d in dates])
        month = np.array([d.month for d in dates])
        is_weekend = (day_of_week >= 5).astype(int)
        is_month_start = np.array([d.day <= 5 for d in dates]).astype(int)
        
        # 复杂的混杂关系
        confounder_effect = (
            2 * is_weekend + 
            1.5 * is_month_start + 
            0.5 * np.sin(2 * np.pi * month / 12)
        )
        
        # 目标变量基础值
        base_outcome = trend + seasonal + confounder_effect
        
        # 添加时变的干预效应
        intervention_effect = np.zeros(n_days)
        for i in range(n_days):
            # 当前和历史干预的累积效应
            for lag in range(min(i + 1, 7)):  # 考虑最近7天的效应
                if i - lag >= 0 and treatment[i - lag] == 1:
                    decay_factor = effect_decay ** lag
                    seasonal_modifier = 1 + 0.3 * np.sin(2 * np.pi * (i % 365) / 365)
                    intervention_effect[i] += base_intervention_effect * decay_factor * seasonal_modifier
        
        # 最终目标变量
        target = base_outcome + intervention_effect + np.random.normal(0, noise_level, n_days)
        
        # 添加一些外生变量
        weather_effect = 2 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, n_days)
        competitor_activity = np.random.binomial(1, 0.15, n_days)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'timestamp': dates,
            'target': target,
            'treatment': treatment.astype(int),
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'is_month_start': is_month_start,
            'weather_effect': weather_effect,
            'competitor_activity': competitor_activity,
            'true_intervention_effect': intervention_effect  # 用于验证
        })
        
        return data
    
    def prepare_timeseries_for_tabpfn(self, 
                                    data: pd.DataFrame,
                                    target_col: str = 'target') -> TimeSeriesDataFrame:
        """准备TabPFN-TS格式的数据"""
        
        if not self.use_real_models:
            return data  # 模拟模式直接返回
        
        # 创建TimeSeriesDataFrame格式
        df_for_ts = data[['timestamp', target_col]].copy()
        df_for_ts = df_for_ts.set_index('timestamp')
        
        # 添加item_id用于TimeSeriesDataFrame
        df_for_ts['item_id'] = 0
        df_for_ts = df_for_ts.set_index('item_id', append=True)
        df_for_ts = df_for_ts.reorder_levels(['item_id', 'timestamp'])
        
        return TimeSeriesDataFrame(df_for_ts)
    
    def extract_tabpfn_features(self, 
                               data: pd.DataFrame,
                               target_col: str = 'target') -> pd.DataFrame:
        """使用TabPFN-TS提取时间序列特征"""
        
        if not self.use_real_models:
            return self._extract_mock_features(data, target_col)
        
        try:
            # 准备TimeSeriesDataFrame
            ts_data = self.prepare_timeseries_for_tabpfn(data, target_col)
            
            # 应用特征转换
            featured_data = self.feature_transformer.transform(ts_data)
            
            # 转换回普通DataFrame并添加其他变量
            result_df = featured_data.reset_index()
            
            # 合并原始数据中的其他列
            merge_cols = [col for col in data.columns 
                         if col not in ['timestamp', target_col]]
            
            if merge_cols:
                original_with_idx = data.reset_index()
                for col in merge_cols:
                    if len(original_with_idx) == len(result_df):
                        result_df[col] = original_with_idx[col].values
            
            return result_df
            
        except Exception as e:
            print(f"TabPFN-TS特征提取失败: {e}")
            print("使用备用特征提取方法")
            return self._extract_mock_features(data, target_col)
    
    def _extract_mock_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """备用的特征提取方法"""
        df = data.copy()
        
        # 趋势特征
        df['running_index'] = range(len(df))
        df['trend'] = df['running_index'] / len(df)
        
        # 日历特征
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofyear / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofyear / 365)
        
        # 滞后特征
        for lag in [1, 3, 7, 14]:
            if lag < len(df):
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # 滑动窗口特征
        for window in [3, 7, 14, 30]:
            if window < len(df):
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
                df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
                df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # 周期性特征（基于FFT的自动发现）
        if len(df) > 20:
            target_clean = df[target_col].fillna(method='ffill').fillna(method='bfill')
            
            # 简单的FFT分析
            fft_vals = np.fft.fft(target_clean - target_clean.mean())
            freqs = np.fft.fftfreq(len(target_clean))
            
            # 找到最强的几个频率分量
            power = np.abs(fft_vals)
            # 排除直流分量和负频率
            valid_idx = (freqs > 0) & (freqs < 0.5)
            if np.any(valid_idx):
                valid_freqs = freqs[valid_idx]
                valid_power = power[valid_idx]
                
                # 选择最强的3个频率
                top_freq_idx = np.argsort(valid_power)[-3:]
                top_freqs = valid_freqs[top_freq_idx]
                
                for i, freq in enumerate(top_freqs):
                    if freq > 0:
                        period = 1 / freq
                        df[f'auto_sin_{i}'] = np.sin(2 * np.pi * df['running_index'] / period)
                        df[f'auto_cos_{i}'] = np.cos(2 * np.pi * df['running_index'] / period)
        
        # 填充缺失值
        df = df.fillna(0)
        
        return df
    
    def estimate_causal_effect_dopfn(self,
                                   featured_data: pd.DataFrame,
                                   intervention_col: str = 'treatment',
                                   target_col: str = 'target',
                                   query_idx: int = None,
                                   context_window: int = None) -> Dict[str, Any]:
        """使用Do-PFN估计因果效应"""
        
        if query_idx is None:
            query_idx = len(featured_data) - 1
        
        if context_window is None:
            context_window = min(query_idx, 100)  # 最多使用100个历史点
        
        # 确定上下文范围
        context_start = max(0, query_idx - context_window)
        context_data = featured_data.iloc[context_start:query_idx].copy()
        query_data = featured_data.iloc[[query_idx]].copy()
        
        if len(context_data) < 10:
            raise ValueError(f"上下文数据太少: {len(context_data)} < 10")
        
        # 准备特征
        feature_cols = [col for col in featured_data.columns 
                       if col not in [target_col, 'timestamp', 'item_id']]
        
        # 确保所有特征列都存在且是数值型
        for col in feature_cols:
            if col not in context_data.columns:
                context_data[col] = 0
                query_data[col] = 0
            
            # 转换为数值型
            context_data[col] = pd.to_numeric(context_data[col], errors='coerce').fillna(0)
            query_data[col] = pd.to_numeric(query_data[col], errors='coerce').fillna(0)
        
        X_context = context_data[feature_cols].values.astype(np.float32)
        y_context = context_data[target_col].values.astype(np.float32)
        
        import torch

        # 将numpy数组转换为torch张量
        X_context_tensor = torch.from_numpy(X_context).float()
        y_context_tensor = torch.from_numpy(y_context).float()

        # 拟合模型
        self.dopfn.fit(X_context_tensor, y_context_tensor)

        # 拟合Do-PFN模型
        # try:
        #     self.dopfn.fit(X_context, y_context)
        # except Exception as e:
        #     print(f"Do-PFN拟合失败: {e}")
        #     raise
        
        # 准备反事实查询
        X_query_base = query_data[feature_cols].values.astype(np.float32)
        
        # 创建干预和控制场景
        X_query_treatment = X_query_base.copy()
        X_query_control = X_query_base.copy()

        # 预测时也要转换
        X_query_treatment_tensor = torch.from_numpy(X_query_treatment).float()
        X_query_control_tensor = torch.from_numpy(X_query_control).float()

        pred_treatment = self.dopfn.predict_full(X_query_treatment_tensor)
        pred_control = self.dopfn.predict_full(X_query_control_tensor)
        
        # 找到treatment列的索引
        if intervention_col in feature_cols:
            treatment_idx = feature_cols.index(intervention_col)
            X_query_treatment[:, treatment_idx] = 1.0
            X_query_control[:, treatment_idx] = 0.0
        else:
            raise ValueError(f"干预列 '{intervention_col}' 不在特征列中")
        
        # 预测反事实结果
        try:
            if hasattr(self.dopfn, 'predict_full'):
                pred_treatment = self.dopfn.predict_full(X_query_treatment)
                pred_control = self.dopfn.predict_full(X_query_control)
                
                # 提取预测结果
                # if isinstance(pred_treatment, dict):
                #     treatment_mean = pred_treatment.get('mean', pred_treatment.get('median', 0))[0]
                #     control_mean = pred_control.get('mean', pred_control.get('median', 0))[0]
                # 提取预测结果
                if isinstance(pred_treatment, dict):
                    treatment_mean_raw = pred_treatment.get('mean', pred_treatment.get('median', 0))
                    control_mean_raw = pred_control.get('mean', pred_control.get('median', 0))
                    
                    # 处理不同的数据类型
                    if hasattr(treatment_mean_raw, 'cpu'):
                        treatment_mean = float(treatment_mean_raw.cpu().numpy()[0])
                        control_mean = float(control_mean_raw.cpu().numpy()[0])
                    elif isinstance(treatment_mean_raw, np.ndarray):
                        treatment_mean = float(treatment_mean_raw[0])
                        control_mean = float(control_mean_raw[0])
                    else:
                        treatment_mean = float(treatment_mean_raw)
                        control_mean = float(control_mean_raw)    
                        
                    # 估计不确定性
                    treatment_std = pred_treatment.get('std', np.array([0.5]))[0]
                    control_std = pred_control.get('std', np.array([0.5]))[0]
                else:
                    treatment_mean = float(pred_treatment[0])
                    control_mean = float(pred_control[0])
                    treatment_std = 0.5
                    control_std = 0.5
            else:
                # 备用预测方法
                treatment_mean = float(self.dopfn.predict(X_query_treatment)[0])
                control_mean = float(self.dopfn.predict(X_query_control)[0])
                treatment_std = 0.5
                control_std = 0.5
                
        except Exception as e:
            print(f"Do-PFN预测失败: {e}")
            raise
        
        # 计算因果效应
        causal_effect = {
            'ate': treatment_mean - control_mean,
            'treatment_outcome': treatment_mean,
            'control_outcome': control_mean,
            'uncertainty': np.sqrt(treatment_std**2 + control_std**2),
            'query_idx': query_idx,
            'context_length': len(context_data)
        }
        
        return causal_effect
    
    def run_time_varying_causal_analysis(self,
                                       data: pd.DataFrame,
                                       intervention_col: str = 'treatment',
                                       target_col: str = 'target',
                                       start_idx: int = None,
                                       end_idx: int = None,
                                       step_size: int = 1,
                                       min_context: int = 30) -> pd.DataFrame:
        """运行时变因果分析"""
        
        print("开始时变因果分析...")
        
        # 1. 特征工程
        print("1. 执行特征工程...")
        featured_data = self.extract_tabpfn_features(data, target_col)
        print(f"   特征维度: {featured_data.shape}")
        
        # 2. 设置分析范围
        if start_idx is None:
            start_idx = min_context
        if end_idx is None:
            end_idx = len(data) - 1
        
        query_indices = range(start_idx, end_idx + 1, step_size)
        print(f"   分析范围: {start_idx} 到 {end_idx}, 步长: {step_size}")
        print(f"   总查询点: {len(query_indices)}")
        
        # 3. 逐点估计因果效应
        results = []
        successful_estimates = 0
        
        for i, query_idx in enumerate(query_indices):
            try:
                print(f"   处理查询点 {query_idx} ({i+1}/{len(query_indices)})", end='')
                
                effect = self.estimate_causal_effect_dopfn(
                    featured_data,
                    intervention_col=intervention_col,
                    target_col=target_col,
                    query_idx=query_idx
                )
                
                # 添加原始数据信息
                original_row = data.iloc[query_idx]
                result_row = {
                    'query_idx': query_idx,
                    'timestamp': original_row.get('timestamp', query_idx),
                    'ate': effect['ate'],
                    'treatment_outcome': effect['treatment_outcome'],
                    'control_outcome': effect['control_outcome'],
                    'uncertainty': effect['uncertainty'],
                    'context_length': effect['context_length'],
                    'actual_treatment': original_row[intervention_col],
                    'actual_outcome': original_row[target_col],
                }
                
                # 如果有真实效应，添加进来
                if 'true_intervention_effect' in original_row:
                    result_row['true_effect'] = original_row['true_intervention_effect']
                
                results.append(result_row)
                successful_estimates += 1
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ (错误: {str(e)[:50]})")
                continue
        
        print(f"\n   成功估计: {successful_estimates}/{len(query_indices)} 个时间点")
        
        if not results:
            raise ValueError("没有成功的因果效应估计")
        
        return pd.DataFrame(results)
    
    def plot_comprehensive_analysis(self,
                                  original_data: pd.DataFrame,
                                  causal_results: pd.DataFrame,
                                  intervention_col: str = 'treatment',
                                  target_col: str = 'target'):
        """绘制全面的因果分析结果"""
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # 1. 原始时间序列和干预
        ax1 = axes[0]
        ax1.plot(original_data['timestamp'], original_data[target_col], 
                label='目标变量', color='blue', alpha=0.7, linewidth=1)
        
        # 标记干预点
        intervention_points = original_data[original_data[intervention_col] == 1]
        if len(intervention_points) > 0:
            ax1.scatter(intervention_points['timestamp'], intervention_points[target_col],
                       color='red', s=30, alpha=0.8, label='干预点', zorder=5)
        
        ax1.set_title('原始时间序列数据', fontsize=14, fontweight='bold')
        ax1.set_ylabel('目标变量值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 估计的因果效应
        ax2 = axes[1]
        ax2.plot(causal_results['timestamp'], causal_results['ate'], 
                color='green', linewidth=2, label='估计的ATE')
        
        # 不确定性区间
        ax2.fill_between(causal_results['timestamp'],
                        causal_results['ate'] - causal_results['uncertainty'],
                        causal_results['ate'] + causal_results['uncertainty'],
                        alpha=0.3, color='green', label='95%置信区间')
        
        # 如果有真实效应，显示对比
        if 'true_effect' in causal_results.columns:
            # 计算每个时间点的真实瞬时效应变化
            true_effects = []
            for idx in causal_results['query_idx']:
                if idx > 0:
                    current_true = original_data.iloc[idx]['true_intervention_effect']
                    prev_true = original_data.iloc[idx-1]['true_intervention_effect']
                    true_effects.append(current_true - prev_true)
                else:
                    true_effects.append(original_data.iloc[idx]['true_intervention_effect'])
            
            ax2.plot(causal_results['timestamp'], true_effects,
                    color='orange', linewidth=2, linestyle='--', label='真实效应', alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('因果效应估计 vs 真实效应', fontsize=14, fontweight='bold')
        ax2.set_ylabel('因果效应')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 反事实预测对比
        ax3 = axes[2]
        ax3.plot(causal_results['timestamp'], causal_results['treatment_outcome'],
                color='red', label='干预情景预测', linewidth=2)
        ax3.plot(causal_results['timestamp'], causal_results['control_outcome'],
                color='blue', label='控制情景预测', linewidth=2)
        ax3.plot(causal_results['timestamp'], causal_results['actual_outcome'],
                color='black', alpha=0.7, linestyle=':', linewidth=2, label='实际观测值')
        
        ax3.set_title('反事实预测对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('预测结果')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差分析
        ax4 = axes[3]
        if 'true_effect' in causal_results.columns:
            true_effects_instant = []
            for idx in causal_results['query_idx']:
                if idx > 0:
                    current_true = original_data.iloc[idx]['true_intervention_effect']
                    prev_true = original_data.iloc[idx-1]['true_intervention_effect']
                    true_effects_instant.append(current_true - prev_true)
                else:
                    true_effects_instant.append(original_data.iloc[idx]['true_intervention_effect'])
            
            estimation_error = causal_results['ate'] - true_effects_instant
            ax4.plot(causal_results['timestamp'], estimation_error,
                    color='purple', linewidth=2, label='估计误差')
            ax4.fill_between(causal_results['timestamp'],
                            -causal_results['uncertainty'],
                            causal_results['uncertainty'],
                            alpha=0.3, color='gray', label='不确定性区间')
            
            rmse = np.sqrt(np.mean(estimation_error**2))
            mae = np.mean(np.abs(estimation_error))
            ax4.set_title(f'估计误差分析 (RMSE: {rmse:.3f}, MAE: {mae:.3f})', 
                         fontsize=14, fontweight='bold')
        else:
            # 如果没有真实效应，显示预测残差
            prediction_residuals = causal_results['actual_outcome'] - causal_results['treatment_outcome']
            ax4.plot(causal_results['timestamp'], prediction_residuals,
                    color='purple', linewidth=2, label='预测残差')
            ax4.set_title('预测残差分析', fontsize=14, fontweight='bold')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('时间')
        ax4.set_ylabel('误差')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig('causal_analysis_report.png', dpi=300)
        
        # 打印统计汇总
        self._print_analysis_summary(causal_results, original_data)
    
    def _print_analysis_summary(self, causal_results: pd.DataFrame, original_data: pd.DataFrame):
        """打印分析汇总统计"""
        print("\n" + "="*60)
        print("📊 时间序列因果分析汇总报告")
        print("="*60)
        
        # 基本统计
        print(f"📈 分析时间范围: {len(causal_results)} 个时间点")
        print(f"📈 平均估计因果效应: {causal_results['ate'].mean():.4f}")
        print(f"📈 因果效应标准差: {causal_results['ate'].std():.4f}")
        print(f"📈 平均不确定性: {causal_results['uncertainty'].mean():.4f}")
        
        # 显著性分析
        significant_positive = (causal_results['ate'] > causal_results['uncertainty']).sum()
        significant_negative = (causal_results['ate'] < -causal_results['uncertainty']).sum()
        
        print(f"\n🔍 显著性分析:")
        print(f"   显著正效应时间点: {significant_positive} ({significant_positive/len(causal_results)*100:.1f}%)")
        print(f"   显著负效应时间点: {significant_negative} ({significant_negative/len(causal_results)*100:.1f}%)")
        print(f"   不显著时间点: {len(causal_results) - significant_positive - significant_negative}")
        
        # 真实效应对比（如果可用）
        if 'true_effect' in causal_results.columns:
            true_effects = []
            for idx in causal_results['query_idx']:
                if idx > 0:
                    current_true = original_data.iloc[idx]['true_intervention_effect']
                    prev_true = original_data.iloc[idx-1]['true_intervention_effect']
                    true_effects.append(current_true - prev_true)
                else:
                    true_effects.append(original_data.iloc[idx]['true_intervention_effect'])
            
            estimation_error = causal_results['ate'] - true_effects
            rmse = np.sqrt(np.mean(estimation_error**2))
            mae = np.mean(np.abs(estimation_error))
            correlation = np.corrcoef(causal_results['ate'], true_effects)[0, 1]
            
            print(f"\n✅ 估计精度评估:")
            print(f"   均方根误差(RMSE): {rmse:.4f}")
            print(f"   平均绝对误差(MAE): {mae:.4f}")
            print(f"   相关系数: {correlation:.4f}")
        
        # 干预分析
        total_interventions = original_data[original_data['treatment'] == 1].shape[0]
        intervention_rate = total_interventions / len(original_data)
        
        print(f"\n💊 干预统计:")
        print(f"   总干预次数: {total_interventions}")
        print(f"   干预率: {intervention_rate:.3f}")
        
        # 模型性能
        avg_context_length = causal_results['context_length'].mean()
        print(f"\n🔧 模型设置:")
        print(f"   平均上下文长度: {avg_context_length:.1f}")
        print(f"   使用真实模型: {'是' if self.use_real_models else '否'}")


# 模拟类定义（当真实模型不可用时）
class MockFeatureTransformer:
    def transform(self, data):
        return data

class MockDoPFNWrapper:
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_fitted = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict_full(self, X):
        if not self.is_fitted:
            raise ValueError("模型未拟合")
        
        pred = self.model.predict(X)
        std_est = np.abs(pred) * 0.1 + 0.5  # 简单的不确定性估计
        
        return {
            'mean': pred,
            'std': std_est,
            'median': pred
        }
    
    def predict(self, X):
        return self.model.predict(X)


def main_comprehensive_demo():
    """完整的时间序列因果推断演示"""
    
    print("🚀 启动时间序列因果推断系统")
    print("="*60)
    
    # 1. 初始化系统
    causal_system = RealTimeSeriesCausalInference(use_real_models=True)
    
    # 2. 生成更真实的数据
    print("\n📊 生成真实感时间序列数据...")
    data = causal_system.generate_realistic_timeseries(
        n_days=200,
        intervention_probability=0.25,
        base_intervention_effect=4.0,
        effect_decay=0.7,
        seasonal_components=[7, 30, 90],
        noise_level=1.2
    )
    
    print(f"   数据维度: {data.shape}")
    print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    print(f"   干预总数: {data['treatment'].sum()}")
    print(f"   目标变量范围: {data['target'].min():.2f} 到 {data['target'].max():.2f}")
    
    # 3. 运行时变因果分析
    print("\n🔍 执行时变因果分析...")
    causal_results = causal_system.run_time_varying_causal_analysis(
        data,
        intervention_col='treatment',
        target_col='target',
        start_idx=50,  # 从第50天开始分析
        end_idx=180,   # 到第180天结束
        step_size=2,   # 每2天分析一次
        min_context=40  # 最少40天的历史数据
    )
    
    # 4. 可视化和分析
    print("\n📈 生成可视化结果...")
    causal_system.plot_comprehensive_analysis(
        data, causal_results,
        intervention_col='treatment',
        target_col='target'
    )
    
    return {
        'system': causal_system,
        'data': data,
        'results': causal_results
    }

if __name__ == "__main__":
    results = main_comprehensive_demo()