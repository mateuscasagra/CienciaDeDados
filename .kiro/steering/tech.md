# Technical Stack

## Language & Runtime
- **Python 3.8+**: Primary language for all components

## Core Libraries

### Data Processing
- **pandas**: Data manipulation and Excel file reading
- **numpy**: Numerical computations and array operations
- **openpyxl/xlrd**: Excel file format support

### Statistical Analysis
- **scipy**: Advanced statistical tests and optimization algorithms
- **scikit-learn**: Machine learning models and regression analysis

### Visualization
- **matplotlib**: Static plot generation
- **seaborn**: Statistical data visualization
- **plotly**: Interactive charts and graphs

### Web Dashboard
- **streamlit**: Interactive web application framework

## Project Structure
- Class-based architecture with separation of concerns
- Static methods for reusable statistical calculations
- Modular design for different analysis types

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
# Run complete analysis (console output)
python index.py

# Launch interactive dashboard
streamlit run dashboard.py
```

### Output Files
- `relatorio-de-saida.txt`: Basic statistical report
- `relatorio-completo-acidentes-fatais.txt`: Comprehensive analysis report
- `*.png`: Generated statistical plots (histograms, boxplots, comparisons)

## Development Practices
- Use `warnings.filterwarnings('ignore')` to suppress non-critical warnings
- Set random seeds (`np.random.seed(42)`) for reproducible results
- Handle NaN values explicitly before statistical operations
- Use UTF-8 encoding for file operations to support Portuguese text
- Implement caching with `@st.cache_data` for dashboard performance
