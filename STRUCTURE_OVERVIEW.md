# AutoInterp Use Cases - Structure Overview

## Project Description

This repository contains comprehensive AutoInterp use cases and analysis tools for interpreting neural network features, with a focus on financial language models. The project provides a complete framework for feature interpretability analysis, circuit tracing, and feature steering.

## Repository Structure

```
InterpUseCases_autointerp/
├── .gitignore                          # Git ignore file for model files and large data
├── STRUCTURE_OVERVIEW.md               # This file - project structure overview
│
├── CircuitTracing/                     # Circuit tracing analysis tools
│   ├── circuit_tracer.py              # Main circuit tracing implementation
│   ├── circuit_tracer_analysis.png    # Circuit analysis visualization
│   └── circuit_tracer_report.txt      # Circuit tracing analysis report
│
├── FinanceLabeling/                    # Financial feature labeling and analysis
│   ├── multi_layer_full_results/      # Complete multi-layer analysis results
│   │   ├── multi_layer_full_layer4/   # Layer 4 analysis results
│   │   │   ├── explanations/          # Feature explanations
│   │   │   ├── results_summary.csv    # Summary of results
│   │   │   └── run_config.json        # Analysis configuration
│   │   ├── multi_layer_full_layer10/  # Layer 10 analysis results
│   │   ├── multi_layer_full_layer16/  # Layer 16 analysis results
│   │   ├── multi_layer_full_layer22/  # Layer 22 analysis results
│   │   ├── multi_layer_full_layer28/  # Layer 28 analysis results
│   │   └── working_delphi_openrouter_v2_summary.csv  # Consolidated summary
│   │
│   ├── multi_layer_lite_results/      # Lightweight multi-layer results
│   │   ├── features_layer4.csv        # Layer 4 feature analysis
│   │   ├── features_layer10.csv       # Layer 10 feature analysis
│   │   ├── features_layer16.csv       # Layer 16 feature analysis
│   │   ├── features_layer22.csv       # Layer 22 feature analysis
│   │   └── features_layer28.csv       # Layer 28 feature analysis
│   │
│   ├── single_layer_full_results/     # Single-layer detailed analysis
│   │   ├── single_layer_short_layer4/ # Layer 4 single analysis
│   │   ├── single_layer_short_layer4_feature127/  # Feature 127 analysis
│   │   ├── single_layer_short_layer4_feature141/  # Feature 141 analysis
│   │   ├── single_layer_short_layer4_feature3/    # Feature 3 analysis
│   │   └── single_layer_short_layer4_feature384/  # Feature 384 analysis
│   │
│   ├── single_layer_openrouter_results/  # OpenRouter single-layer results
│   │   ├── explanations/              # Feature explanations
│   │   ├── results_summary.csv        # Results summary
│   │   └── run_config.json           # Analysis configuration
│   │
│   ├── copy_all_results.sh           # Script to consolidate all results
│   ├── run_multi_layer_full_analysis.sh    # Multi-layer full analysis script
│   ├── run_multi_layer_lite_analysis.sh    # Multi-layer lite analysis script
│   ├── run_multi_layer_short.sh            # Multi-layer short analysis script
│   ├── run_single_layer_full_analysis.sh   # Single-layer full analysis script
│   ├── run_single_layer_openrouter.sh      # Single-layer OpenRouter script
│   └── run_single_layer_short.sh           # Single-layer short analysis script
│
├── Paper/                             # Research and documentation
│   └── FINANCIAL_FEATURE_INTERPRETABILITY_ANALYSIS.md  # Comprehensive analysis paper
│
├── Steering/                          # Feature steering and manipulation
│   └── feature_steering_demo.py       # Feature steering demonstration
│
├── consolidate_labels.py              # Script to consolidate feature labels
├── generic_comparison.py              # Generic comparison analysis tool
├── generic_delphi_runner.py           # Generic Delphi analysis runner
├── generic_feature_analysis.py        # Generic feature analysis tool
├── generic_feature_labeling.py        # Generic feature labeling tool
├── generic_master_script.py           # Master script for complete pipeline
├── multi_layer_financial_analysis.py  # Multi-layer financial analysis
└── run_financial_analysis.py          # Financial analysis runner
```

## Key Components

### 1. Circuit Tracing (`CircuitTracing/`)
- **Purpose**: Analyze and trace neural network circuits
- **Main File**: `circuit_tracer.py`
- **Features**: Circuit analysis, feature relationship mapping, visualization
- **Output**: Circuit analysis reports and visualizations

### 2. Financial Feature Labeling (`FinanceLabeling/`)
- **Purpose**: Comprehensive financial feature analysis and labeling
- **Structure**: 
  - Multi-layer analysis results (layers 4, 10, 16, 22, 28)
  - Single-layer detailed analysis
  - OpenRouter integration results
- **Features**: Feature explanations, activation analysis, financial relevance scoring

### 3. Research Documentation (`Paper/`)
- **Purpose**: Academic and research documentation
- **Main File**: `FINANCIAL_FEATURE_INTERPRETABILITY_ANALYSIS.md`
- **Content**: Comprehensive analysis of financial feature interpretability

### 4. Feature Steering (`Steering/`)
- **Purpose**: Demonstrate feature steering capabilities
- **Main File**: `feature_steering_demo.py`
- **Features**: Risk steering, sentiment steering, decision steering

### 5. Generic Analysis Tools (Root Directory)
- **`consolidate_labels.py`**: Consolidate feature labels from multiple analyses
- **`generic_comparison.py`**: Compare results across different analysis types
- **`generic_delphi_runner.py`**: Run Delphi-based interpretability analysis
- **`generic_feature_analysis.py`**: Generic feature analysis and visualization
- **`generic_feature_labeling.py`**: Automated feature labeling
- **`generic_master_script.py`**: Orchestrate complete analysis pipeline
- **`multi_layer_financial_analysis.py`**: Specialized financial analysis
- **`run_financial_analysis.py`**: Convenient financial analysis runner

## Analysis Types

### 1. Multi-Layer Analysis
- **Full Analysis**: Complete analysis of all features across layers
- **Lite Analysis**: Simplified analysis for quick insights
- **Short Analysis**: Minimal analysis for testing

### 2. Single-Layer Analysis
- **Full Analysis**: Detailed analysis of specific layers
- **Feature-Specific**: Analysis of individual features
- **OpenRouter Integration**: Using OpenRouter for interpretation

### 3. Cross-Layer Analysis
- **Feature Evolution**: How features change across layers
- **Domain Specialization**: Financial domain specialization by layer
- **Complexity Evolution**: Complexity changes across layers

## Financial Domains Covered

1. **Risk Assessment**: Risk analysis and volatility assessment
2. **Market Analysis**: Market sentiment and trend analysis
3. **Portfolio Management**: Portfolio optimization and management
4. **Trading Strategies**: Algorithmic and quantitative trading
5. **Financial Reporting**: Earnings and financial performance analysis

## Key Features

### 1. Interpretability Analysis
- Feature activation analysis
- Interpretability scoring
- Financial relevance assessment
- Human-readable explanations

### 2. Circuit Tracing
- Feature relationship mapping
- Circuit strength analysis
- Activation pattern analysis
- Connection visualization

### 3. Feature Steering
- Risk level steering
- Sentiment steering
- Decision bias steering
- Effect measurement

### 4. Visualization
- Layer-wise feature analysis
- Cross-layer evolution plots
- Correlation heatmaps
- Activation distributions

## Usage Examples

### 1. Run Complete Financial Analysis
```bash
python run_financial_analysis.py --model path/to/model --data path/to/data --output results/
```

### 2. Run Multi-Layer Analysis
```bash
python multi_layer_financial_analysis.py --model path/to/model --data path/to/data --layers 4 10 16 22 28
```

### 3. Run Feature Steering Demo
```bash
python Steering/feature_steering_demo.py
```

### 4. Run Circuit Tracing
```bash
python CircuitTracing/circuit_tracer.py
```

### 5. Run Master Pipeline
```bash
python generic_master_script.py --config config.json --data data.txt --output results/
```

## Configuration

The project uses JSON configuration files for:
- Model paths and parameters
- Analysis layers and features
- Output directories and formats
- Delphi API configuration
- Visualization settings

## Output Formats

### 1. CSV Files
- Feature analysis results
- Activation scores
- Interpretability metrics
- Financial relevance scores

### 2. JSON Files
- Detailed analysis results
- Configuration files
- Summary statistics
- Cross-layer analysis

### 3. Visualization Files
- PNG images for plots and heatmaps
- PDF files for detailed visualizations
- Interactive plots (when supported)

### 4. Text Files
- Feature explanations
- Analysis reports
- Circuit tracing reports

## Dependencies

- **Python 3.8+**
- **PyTorch**: For model loading and analysis
- **NumPy**: For numerical computations
- **Pandas**: For data manipulation
- **Matplotlib**: For visualization
- **Seaborn**: For statistical visualization
- **JSON**: For configuration and results

## Research Applications

This framework is designed for:
1. **Academic Research**: Feature interpretability in financial AI
2. **Industry Applications**: Financial model understanding and validation
3. **Model Development**: Improving financial language models
4. **Regulatory Compliance**: Understanding AI decision-making in finance
5. **Risk Management**: Identifying and steering model behavior

## Future Extensions

1. **Additional Domains**: Extend beyond financial analysis
2. **Real-time Analysis**: Live feature monitoring
3. **Interactive Visualization**: Web-based analysis interface
4. **Model Comparison**: Compare different model architectures
5. **Automated Reporting**: Generate automated analysis reports

## Contributing

To contribute to this project:
1. Follow the existing code structure
2. Add comprehensive documentation
3. Include test cases for new features
4. Update this structure overview
5. Ensure compatibility with existing tools

## License

This project is part of the AutoInterp framework and follows the same licensing terms.

---

*Last updated: January 15, 2024*
*Version: 1.0*
*Framework: AutoInterp with Delphi Integration*
