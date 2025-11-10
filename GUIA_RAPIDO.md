# 🚀 GUIA RÁPIDO DO PROJETO

## ⚡ Início Rápido

### Executar o Projeto
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar análise completa
python index.py

# Testar clustering (3 métodos)
python src/machine_learning/testar_clustering_completo.py

# Abrir dashboard
streamlit run dashboard.py
```

---

## 📊 Para Apresentação (03/11 e 10/11)

### 1. Coletar Métricas (30 min)
Execute os scripts e **ANOTE** os valores:
- Média, desvio padrão, R², RMSE
- Silhouette Score (K-Means, Hierarchical, EM)
- Acurácia, F1-Score (4 modelos)

### 2. Criar Slides (2-3h)
Use `docs/apresentacao/TEMPLATE_SLIDES.txt` como guia:
- 15 slides no total
- Insira os valores anotados
- Adicione gráficos gerados

### 3. Dividir Falas (5 min cada)
- **Marco:** Slides 1-5 (Intro, Governança, LGPD, Estatística)
- **Mateus:** Slides 6-8 (Análises Avançadas, Regressões)
- **Rhillary:** Slides 9-15 (ML, Conclusões, Demo)

### 4. Ensaiar (1-2h)
- Ler `docs/apresentacao/GUIA_APRESENTACAO.md`
- Ensaiar 2-3 vezes com cronômetro
- Praticar demo do dashboard

---

## ✅ Checklist Pré-Apresentação

**1 Semana Antes:**
- [ ] Testar todo o código
- [ ] Coletar métricas
- [ ] Criar slides
- [ ] Primeiro ensaio

**3 Dias Antes:**
- [ ] Revisar documentação
- [ ] Segundo ensaio
- [ ] Preparar respostas para perguntas
- [ ] Criar backup (2 pen drives + nuvem)

**1 Dia Antes:**
- [ ] Terceiro ensaio completo
- [ ] Testar no computador da apresentação
- [ ] Imprimir slides (backup)
- [ ] Descansar bem

**Dia da Apresentação:**
- [ ] Chegar 15-30 min antes
- [ ] Testar projetor e dashboard
- [ ] Respirar e relaxar

---

## 🎯 Conseguimos Fazer Previsões? **SIM!**

1. **Regressão:** Prediz NÚMERO de acidentes (R², RMSE)
2. **Classificação:** Prediz se será FATAL (Acurácia, F1)
3. **Clustering:** Identifica PADRÕES de risco (Silhouette)

**Exemplo:** Sexta 18h + Chuva + BR-277 = XX% probabilidade fatal → Intensificar fiscalização

---

## 📁 Estrutura do Projeto

```
CienciaDeDados/
├── index.py                    # Script principal
├── dashboard.py                # Dashboard Streamlit
├── src/                        # Código fonte
│   └── machine_learning/       # ML (clustering, classificação)
├── docs/
│   ├── apresentacao/           # Guias de apresentação
│   └── governanca/             # Documentação governança
└── excel/                      # Dados
```

---

## ❓ Perguntas Frequentes

**P: Por que Curitiba?**  
R: Dados disponíveis, representativos, volume adequado.

**P: Como validaram?**  
R: Train/test split 70/30, múltiplas métricas.

**P: Qual a precisão?**  
R: [Informar após executar scripts]

**P: Funciona em outras cidades?**  
R: Sim, é escalável. Basta retreinar com dados locais.

**P: Custo?**  
R: TCO 3 anos: R$ 500-800k. ROI: 200-300%.

---

## 📞 Arquivos Importantes

- `README.md` - Documentação completa
- `docs/apresentacao/GUIA_APRESENTACAO.md` - Roteiro detalhado com falas
- `docs/apresentacao/TEMPLATE_SLIDES.txt` - Template dos 15 slides
- `docs/governanca/GOVERNANCA_CORPORATIVA_TI.md` - Governança completa

---

## ✅ Projeto Completo

- ✅ Governança Corporativa e TI
- ✅ LGPD
- ✅ Estatística Descritiva
- ✅ Análises Avançadas (TCL, Correlação, T-Student, Qui-quadrado)
- ✅ Regressão Linear e Não Linear (5 métodos otimização)
- ✅ ML Não Supervisionado (K-Means, Hierarchical, EM)
- ✅ ML Supervisionado (Árvore, Random Forest, KNN, Rede Neural)
- ✅ Avaliação de Qualidade (R², RMSE, Acurácia, F1, Matriz Confusão)
- ✅ Dashboard Interativo

**Vocês estão prontos! 🚀**
