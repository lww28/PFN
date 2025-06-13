"""
时间序列因果推断配置和运行脚本
提供简单的接口来配置和运行因果推断分析
"""

import os
import yaml
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class CausalInferenceConfig:
    """因果推断配置类"""
    
    # 数据生成配置
    n_days: int = 180
    intervention_probability: float = 0.25
    base_intervention_effect: float = 4.0
    effect_decay: float = 0.7
    seasonal_components: List[int] = None
    noise_level: float = 1.0
    
    # 分析配置
    intervention_col: str = 'treatment'
    target_col: str = 'target'
    covariates: List[str] = None
    
    # 时变分析配置
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    step_size: int = 1
    min_context: int = 30
    context_window: Optional[int] = None
    
    # 模型配置
    use_real_models: bool = True
    
    # 输出配置
    save_results: bool = True
    output_dir: str = "causal_analysis_results"
    plot_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.seasonal_components is None:
            self.seasonal_components = [7, 30, 90]
        if self.covariates is None:
            self.covariates = ['is_weekend', 'is_month_start']

class CausalAnalysisRunner:
    """因果分析运行器"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self.system = None
        self.data = None
        self.results = None
        
        # 创建输出目录
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def load_custom_data(self, data_path: str, 
                        timestamp_col: str = 'timestamp',
                        target_col: str = 'target',
                        intervention_col: str = 'treatment') -> pd.DataFrame:
        """加载自定义数据"""
        
        print(f"📂 加载数据: {data_path}")
        
        # 根据文件扩展名选择加载方法
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        # 验证必要列存在
        required_cols = [timestamp_col, target_col, intervention_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"数据中缺少必要列: {missing_cols}")
        
        # 标准化列名
        data = data.rename(columns={
            timestamp_col: 'timestamp',
            target_col: 'target',
            intervention_col: 'treatment'
        })
        
        # 确保时间戳格式正确
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # 确保干预列是二元的
        data['treatment'] = data['treatment'].astype(int)
        
        print(f"   ✅ 数据加载成功: {data.shape}")
        print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        print(f"   干预次数: {data['treatment'].sum()}")
        
        return data
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """生成合成数据"""
        
        print("🎲 生成合成时间序列数据...")
        
        # 导入系统（这里需要前面定义的类）
        from real_ts_causal_inference import RealTimeSeriesCausalInference
        
        if self.system is None:
            self.system = RealTimeSeriesCausalInference(
                use_real_models=self.config.use_real_models
            )
        
        data = self.system.generate_realistic_timeseries(
            n_days=self.config.n_days,
            intervention_probability=self.config.intervention_probability,
            base_intervention_effect=self.config.base_intervention_effect,
            effect_decay=self.config.effect_decay,
            seasonal_components=self.config.seasonal_components,
            noise_level=self.config.noise_level
        )
        
        print(f"   ✅ 合成数据生成完成: {data.shape}")
        
        return data
    
    def run_analysis(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """运行完整的因果分析"""
        
        print("\n🚀 开始因果推断分析")
        print("="*60)
        
        # 1. 准备数据
        if data is None:
            data = self.generate_synthetic_data()
        
        self.data = data
        
        # 2. 初始化系统
        if self.system is None:
            from real_ts_causal_inference import RealTimeSeriesCausalInference
            self.system = RealTimeSeriesCausalInference(
                use_real_models=self.config.use_real_models
            )
        
        # 3. 设置分析参数
        start_idx = self.config.start_idx or self.config.min_context
        end_idx = self.config.end_idx or len(data) - 1
        
        print(f"📊 分析配置:")
        print(f"   时间范围: {start_idx} 到 {end_idx}")
        print(f"   步长: {self.config.step_size}")
        print(f"   最小上下文: {self.config.min_context}")
        print(f"   使用真实模型: {self.config.use_real_models}")
        
        # 4. 运行时变因果分析
        try:
            results = self.system.run_time_varying_causal_analysis(
                data,
                intervention_col=self.config.intervention_col,
                target_col=self.config.target_col,
                start_idx=start_idx,
                end_idx=end_idx,
                step_size=self.config.step_size,
                min_context=self.config.min_context
            )
            
            self.results = results
            
            print(f"\n✅ 因果分析完成!")
            print(f"   成功分析 {len(results)} 个时间点")
            
        except Exception as e:
            print(f"\n❌ 因果分析失败: {e}")
            raise
        
        # 5. 保存结果
        if self.config.save_results:
            self._save_results()
        
        # 6. 生成可视化
        if self.config.plot_results:
            self._generate_plots()
        
        return {
            'data': self.data,
            'results': self.results,
            'config': self.config,
            'system': self.system
        }
    
    def _save_results(self):
        """保存分析结果"""
        
        print(f"\n💾 保存结果到: {self.config.output_dir}")
        
        # 保存原始数据
        self.data.to_csv(
            os.path.join(self.config.output_dir, 'original_data.csv'), 
            index=False
        )
        
        # 保存因果分析结果
        self.results.to_csv(
            os.path.join(self.config.output_dir, 'causal_results.csv'), 
            index=False
        )
        
        # 保存配置
        config_dict = asdict(self.config)
        with open(os.path.join(self.config.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # 保存汇总统计
        summary = self._generate_summary_stats()
        with open(os.path.join(self.config.output_dir, 'summary.txt'), 'w') as f:
            f.write(summary)
        
        print("   ✅ 结果保存完成")
    
    def _generate_plots(self):
        """生成可视化图表"""
        
        print("\n📈 生成可视化图表...")
        
        # 生成主要分析图表
        self.system.plot_comprehensive_analysis(
            self.data, self.results,
            intervention_col=self.config.intervention_col,
            target_col=self.config.target_col
        )
        
        # 如果需要保存图表
        if self.config.save_results:
            import matplotlib.pyplot as plt
            plt.savefig(
                os.path.join(self.config.output_dir, 'causal_analysis_plots.png'),
                dpi=300, bbox_inches='tight'
            )
            print(f"   📊 图表已保存到: {self.config.output_dir}/causal_analysis_plots.png")
    
    def _generate_summary_stats(self) -> str:
        """生成汇总统计信息"""
        
        summary_lines = [
            "时间序列因果推断分析报告",
            "=" * 50,
            "",
            f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"数据时间范围: {self.data['timestamp'].min()} 到 {self.data['timestamp'].max()}",
            f"总时间点数: {len(self.data)}",
            f"分析时间点数: {len(self.results)}",
            "",
            "干预统计:",
            f"  总干预次数: {self.data[self.config.intervention_col].sum()}",
            f"  干预率: {self.data[self.config.intervention_col].mean():.3f}",
            "",
            "因果效应估计:",
            f"  平均ATE: {self.results['ate'].mean():.4f}",
            f"  ATE标准差: {self.results['ate'].std():.4f}",
            f"  ATE范围: [{self.results['ate'].min():.4f}, {self.results['ate'].max():.4f}]",
            f"  平均不确定性: {self.results['uncertainty'].mean():.4f}",
            "",
            "显著性分析:",
            f"  显著正效应: {(self.results['ate'] > self.results['uncertainty']).sum()} 个时间点",
            f"  显著负效应: {(self.results['ate'] < -self.results['uncertainty']).sum()} 个时间点",
            ""
        ]
        
        # 如果有真实效应，添加精度评估
        if 'true_effect' in self.results.columns:
            true_effects = []
            for idx in self.results['query_idx']:
                if idx > 0:
                    current_true = self.data.iloc[idx]['true_intervention_effect']
                    prev_true = self.data.iloc[idx-1]['true_intervention_effect']
                    true_effects.append(current_true - prev_true)
                else:
                    true_effects.append(self.data.iloc[idx]['true_intervention_effect'])
            
            estimation_error = self.results['ate'] - true_effects
            rmse = np.sqrt(np.mean(estimation_error**2))
            mae = np.mean(np.abs(estimation_error))
            correlation = np.corrcoef(self.results['ate'], true_effects)[0, 1]
            
            summary_lines.extend([
                "估计精度评估:",
                f"  RMSE: {rmse:.4f}",
                f"  MAE: {mae:.4f}",
                f"  相关系数: {correlation:.4f}",
                ""
            ])
        
        summary_lines.extend([
            "配置参数:",
            f"  使用真实模型: {self.config.use_real_models}",
            f"  最小上下文长度: {self.config.min_context}",
            f"  分析步长: {self.config.step_size}",
        ])
        
        return "\n".join(summary_lines)

def load_config_from_file(config_path: str) -> CausalInferenceConfig:
    """从YAML文件加载配置"""
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return CausalInferenceConfig(**config_dict)

def create_example_config() -> CausalInferenceConfig:
    """创建示例配置"""
    
    return CausalInferenceConfig(
        # 数据配置
        n_days=200,
        intervention_probability=0.3,
        base_intervention_effect=5.0,
        effect_decay=0.8,
        seasonal_components=[7, 30, 90],
        noise_level=1.2,
        
        # 分析配置
        intervention_col='treatment',
        target_col='target',
        covariates=['is_weekend', 'is_month_start'],
        
        # 时变分析配置
        start_idx=50,
        end_idx=180,
        step_size=2,
        min_context=40,
        
        # 模型配置
        use_real_models=True,
        
        # 输出配置
        save_results=True,
        output_dir="example_causal_analysis",
        plot_results=True,
        verbose=True
    )

def run_quick_analysis(n_days: int = 150,
                      intervention_prob: float = 0.25,
                      intervention_effect: float = 4.0,
                      use_real_models: bool = True) -> Dict[str, Any]:
    """快速运行因果分析"""
    
    config = CausalInferenceConfig(
        n_days=n_days,
        intervention_probability=intervention_prob,
        base_intervention_effect=intervention_effect,
        use_real_models=use_real_models,
        output_dir=f"quick_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    runner = CausalAnalysisRunner(config)
    return runner.run_analysis()

def run_from_custom_data(data_path: str,
                        timestamp_col: str = 'timestamp',
                        target_col: str = 'target', 
                        intervention_col: str = 'treatment',
                        **kwargs) -> Dict[str, Any]:
    """从自定义数据运行因果分析"""
    
    config = CausalInferenceConfig(
        target_col=target_col,
        intervention_col=intervention_col,
        output_dir=f"custom_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        **kwargs
    )
    
    runner = CausalAnalysisRunner(config)
    data = runner.load_custom_data(data_path, timestamp_col, target_col, intervention_col)
    
    return runner.run_analysis(data)

def main():
    """命令行主函数"""
    
    parser = argparse.ArgumentParser(description='时间序列因果推断分析工具')
    
    parser.add_argument('--config', type=str, help='配置文件路径 (YAML格式)')
    parser.add_argument('--data', type=str, help='自定义数据文件路径')
    parser.add_argument('--quick', action='store_true', help='快速分析模式')
    parser.add_argument('--n-days', type=int, default=150, help='生成数据的天数')
    parser.add_argument('--intervention-prob', type=float, default=0.25, help='干预概率')
    parser.add_argument('--intervention-effect', type=float, default=4.0, help='干预效应大小')
    parser.add_argument('--no-real-models', action='store_true', help='不使用真实模型')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    if args.quick:
        # 快速分析模式
        print("🚀 启动快速因果分析...")
        results = run_quick_analysis(
            n_days=args.n_days,
            intervention_prob=args.intervention_prob,
            intervention_effect=args.intervention_effect,
            use_real_models=not args.no_real_models
        )
        print("✅ 快速分析完成!")
        
    elif args.data:
        # 自定义数据分析
        print(f"📊 从自定义数据运行分析: {args.data}")
        
        kwargs = {}
        if args.output_dir:
            kwargs['output_dir'] = args.output_dir
        if args.no_real_models:
            kwargs['use_real_models'] = False
            
        results = run_from_custom_data(args.data, **kwargs)
        print("✅ 自定义数据分析完成!")
        
    elif args.config:
        # 配置文件模式
        print(f"⚙️ 从配置文件运行: {args.config}")
        config = load_config_from_file(args.config)
        
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.no_real_models:
            config.use_real_models = False
            
        runner = CausalAnalysisRunner(config)
        results = runner.run_analysis()
        print("✅ 配置文件分析完成!")
        
    else:
        # 默认示例分析
        print("📋 运行示例分析...")
        config = create_example_config()
        
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.no_real_models:
            config.use_real_models = False
            
        runner = CausalAnalysisRunner(config)
        results = runner.run_analysis()
        print("✅ 示例分析完成!")
    
    return results

# 使用示例
if __name__ == "__main__":
    
    print("=" * 80)
    print("🔬 时间序列因果推断分析工具")
    print("=" * 80)
    
    # 示例1: 快速分析
    print("\n📊 示例1: 快速因果分析")
    quick_results = run_quick_analysis(
        n_days=100,
        intervention_prob=0.3,
        intervention_effect=5.0,
        use_real_models=True
    )
    
    # 示例2: 详细配置分析
    print("\n📊 示例2: 详细配置分析")
    detailed_config = CausalInferenceConfig(
        n_days=150,
        intervention_probability=0.25,
        base_intervention_effect=4.0,
        effect_decay=0.7,
        seasonal_components=[7, 30],
        noise_level=1.0,
        start_idx=30,
        step_size=3,
        min_context=25,
        output_dir="detailed_analysis_example",
        verbose=True
    )
    
    detailed_runner = CausalAnalysisRunner(detailed_config)
    detailed_results = detailed_runner.run_analysis()
    
    print("\n🎉 所有示例分析完成!")
    print("📁 查看输出目录获取详细结果")