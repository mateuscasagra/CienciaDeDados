# Project Structure

## Directory Layout

```
CienciaDeDados/
├── src/                              # Source code modules
│   ├── xlsClass.py                   # Excel data loading and filtering
│   ├── calculosClass.py              # Basic statistical calculations
│   ├── analises_estatisticas.py     # Advanced statistical analyses
│   ├── regressoes.py                 # Regression models and optimization
│   ├── grafico.py                    # Chart generation utilities
│   ├── machine_learning/             # ML models (if present)
│   └── __init__.py                   # Package initialization
├── excel/                            # Data directory
│   └── dados.xlsx                    # Source dataset
├── .streamlit/                       # Streamlit configuration
│   └── config.toml                   # Dashboard settings
├── index.py                          # Main analysis script
├── dashboard.py                      # Interactive web dashboard
├── graficos.py                       # Additional plotting functions
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── *.png                             # Generated visualization outputs
```

## Module Organization

### `src/xlsClass.py`
- **Purpose**: Data loading and filtering from Excel files
- **Key Class**: `xlsClass`
- **Responsibilities**: 
  - Read Excel data with specific columns
  - Apply filtering rules (fatal accidents in Curitiba, PR)
  - Convert data to dictionary format for processing

### `src/calculosClass.py`
- **Purpose**: Basic statistical calculations
- **Key Class**: `calculosClass` (all static methods)
- **Methods**: `media()`, `desvioPadrao()`, `variancia()`, `percentil()`, `assimetria()`, `curtose()`, `contagemCondicaoMetereologica()`
- **Pattern**: Accepts both lists and pandas Series, converts internally

### `src/analises_estatisticas.py`
- **Purpose**: Advanced statistical analyses
- **Key Class**: `AnaliseEstatistica`
- **Methods**:
  - `teorema_central_limite()`: Central Limit Theorem demonstration
  - `analise_correlacao()`: Pearson and Spearman correlations
  - `analise_distribuicao_normal()`: Normality tests (Shapiro-Wilk, KS)
  - `teste_t_student()`: Group comparison tests
  - `teste_qui_quadrado()`: Independence tests for categorical variables
  - `gerar_relatorio_estatistico()`: Comprehensive report generation

### `src/regressoes.py`
- **Purpose**: Regression analysis and optimization methods
- **Key Class**: `AnaliseRegressao`
- **Methods**:
  - `regressao_linear_simples()`: Simple linear regression
  - `regressao_linear_multipla()`: Multiple linear regression
  - `regressao_nao_linear_parabola()`: Parabolic regression with multiple optimization methods
  - `regressao_nao_linear_exponencial()`: Exponential regression
  - `regressao_bayesiana()`: Bayesian regression with uncertainty quantification
  - `comparar_metodos()`: Comparative analysis of all methods
- **Optimization Methods**: Least squares, Gauss-Newton, Levenberg-Marquardt, Bayesian inference

### `index.py`
- **Purpose**: Main execution script
- **Flow**: Load data → Calculate statistics → Generate visualizations → Run advanced analyses → Save reports
- **Output**: Console output + text reports + PNG charts

### `dashboard.py`
- **Purpose**: Interactive Streamlit web application
- **Features**: Tabbed interface, real-time analysis, interactive charts, method comparison
- **Sections**: Overview, Statistical Analyses, Regressions, Temporal Patterns, ML, Method Comparison

## Data Schema

### Excel Columns (from `xlsClass.colunas`)
- `causa_acidente`: Accident cause
- `uf`: State (filtered to 'PR')
- `municipio`: Municipality (filtered to 'CURITIBA')
- `tipo_acidente`: Accident type
- `classificacao_acidente`: Classification (filtered to 'Com Vítimas Fatais')
- `mortos`: Number of deaths
- `sexo`: Gender
- `idade`: Age (primary analysis variable)
- `fase_dia`: Time of day phase
- `condicao_metereologica`: Weather condition

## Coding Conventions

### Class Design
- Use PascalCase for class names (e.g., `xlsClass`, `calculosClass`)
- Use camelCase for method names (e.g., `aplicaRegras`, `trazDados`)
- Static methods for utility functions that don't need instance state

### Data Handling
- Always check for and handle NaN values before statistical operations
- Use pandas DataFrames for structured data
- Convert between lists, Series, and DataFrames as needed
- Apply masks to filter data: `mascara = df['column'] == value`

### Analysis Pattern
1. Print section headers with `"="*60` separators
2. Execute analysis
3. Print results with formatted output
4. Return dictionary with results for programmatic access

### Visualization
- Save plots as PNG files in project root
- Use descriptive Portuguese titles for charts
- Include axis labels and legends
- Apply consistent color schemes

## Extension Points
- Add new statistical methods to `AnaliseEstatistica` class
- Implement additional regression types in `AnaliseRegressao`
- Create new dashboard tabs in `dashboard.py` for custom analyses
- Add ML models in `src/machine_learning/` directory
