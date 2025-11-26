# 💳 Sistema de Análise de Crédito

> Aplicação web para análise automática de aprovação de crédito usando Machine Learning

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🌐 Aplicação Online

**🚀 Acesse a aplicação:** [https://seu-nome-credito.streamlit.app](https://seu-nome-credito.streamlit.app)

> 💡 **Dica:** Substitua a URL acima pela URL real da sua aplicação após o deploy

## 📋 Sobre o Projeto

Este sistema utiliza **Machine Learning** para analisar automaticamente pedidos de crédito, fornecendo:

- ✅ **Decisão automática** de aprovação/negação
- 📊 **Probabilidade de aprovação** em tempo real  
- 🎯 **Categorização de risco** (baixo, médio, alto)
- 💡 **Recomendações personalizadas** baseadas no perfil
- 📈 **Visualizações interativas** para análise

## 🎯 Funcionalidades

### 📊 **Análise Inteligente**
- Modelo Random Forest treinado com dados históricos
- Avaliação de 4 variáveis principais: idade, renda, score de crédito e experiência
- Acurácia de 84.4% nos testes

### 🖥️ **Interface Intuitiva**
- Design responsivo e amigável
- Visualizações com gráficos de gauge interativos
- Feedback detalhado sobre a decisão
- Análise dos fatores que influenciam o resultado

### ⚡ **Deploy Automático**
- Hospedado no Streamlit Cloud
- Atualizações automáticas via GitHub
- Disponível 24/7 na internet

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Descrição |
|------------|--------|-----------|
| **Python** | 3.8+ | Linguagem principal |
| **Streamlit** | ≥1.28.0 | Framework web |
| **scikit-learn** | ≥1.3.0 | Machine Learning |
| **pandas** | ≥1.5.0 | Manipulação de dados |
| **plotly** | ≥5.15.0 | Visualizações interativas |
| **numpy** | ≥1.24.0 | Computação numérica |

## 📁 Estrutura do Projeto

```
📦 deploy-ml-credito/
├── 📄 app.py                 # Aplicação principal Streamlit
├── 🤖 modelo_credito.pkl     # Modelo ML treinado (Random Forest)
├── 📋 requirements.txt       # Dependências do projeto
└── 📖 README.md             # Este arquivo
```

## 🚀 Como Executar Localmente

### 1️⃣ **Clonar o Repositório**
```bash
git clone https://github.com/seu-usuario/deploy-ml-credito.git
cd deploy-ml-credito
```

### 2️⃣ **Instalar Dependências**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Executar a Aplicação**
```bash
streamlit run app.py
```

### 4️⃣ **Acessar no Navegador**
```
http://localhost:8501
```

## 🧠 Sobre o Modelo

### **Algoritmo:** Random Forest Classifier
- **20 estimadores** (árvores de decisão)
- **Profundidade máxima:** 6 níveis
- **Balanceamento automático** de classes

### **Variáveis de Entrada:**
1. **👤 Idade** (18-80 anos) - Peso: 11.0%
2. **💰 Renda Mensal** (R$ 1.000-50.000) - Peso: 22.0%  
3. **📈 Score de Crédito** (300-850) - Peso: 61.8%
4. **🏦 Experiência com Crédito** (0-4 anos) - Peso: 5.2%

### **Métricas de Performance:**
- **Acurácia:** 84.4%
- **Precisão:** 85% (aprovados) / 85% (negados)
- **Recall:** 84% (aprovados) / 85% (negados)

## 📊 Exemplos de Uso

### **✅ Cliente de Baixo Risco**
- Idade: 30 anos, Renda: R$ 8.000, Score: 750, Experiência: 3 anos
- **Resultado:** Aprovado (99.6% de probabilidade)

### **⚠️ Cliente de Risco Médio**  
- Idade: 25 anos, Renda: R$ 3.000, Score: 650, Experiência: 1 ano
- **Resultado:** Aprovado (61.1% de probabilidade)

### **❌ Cliente de Alto Risco**
- Idade: 60 anos, Renda: R$ 1.500, Score: 400, Experiência: 0 anos
- **Resultado:** Negado (4.0% de probabilidade)

## 🔄 Deploy no Streamlit Cloud

Este projeto está configurado para deploy automático:

1. **Crie seu repositório** com base neste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione este repositório
5. Configure: `Main file path: app.py`
6. Clique em **Deploy!**

A aplicação será atualizada automaticamente a cada push no GitHub.

## 📚 Contexto Acadêmico

### **Disciplina:** Introdução à Ciência de Dados
### **Aula:** 29 - Deploy de Modelos de Machine Learning  
### **Professor:** Adriano Branco
### **Objetivo:** Demonstrar deploy prático de modelos ML na web

#### **Competências Desenvolvidas:**
- ✅ Criação de pipelines de ML end-to-end
- ✅ Deploy de aplicações web com Streamlit
- ✅ Integração GitHub + Streamlit Cloud
- ✅ Boas práticas de documentação de projetos
- ✅ Experiência prática com MLOps

## ⚠️ Limitações e Considerações

### **🚧 Modelo Demonstrativo**
- Treinado com dados sintéticos para fins educacionais
- Simplificado (apenas 4 variáveis)
- Não considera fatores externos (economia, regulamentações)

### **🏭 Para Uso em Produção**
- Expandir dataset com dados reais e históricos
- Incluir mais variáveis (localização, tipo de emprego, etc.)
- Implementar monitoramento contínuo de performance
- Adicionar validações de compliance bancário
- Testes A/B para validação de melhorias


---

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos como parte da disciplina **Introdução à Ciência de Dados**.

---

<div align="center">

**🚀 Feito usando Streamlit e Machine Learning**

[![Streamlit Cloud](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-FF4B4B?style=for-the-badge)](https://streamlit.io/cloud)

*Sistema desenvolvido como demonstração de deploy de modelos ML*

</div>

---

## 📖 Recursos Adicionais

### **📚 Documentação Técnica**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python](https://plotly.com/python/)

### **🎓 Material da Disciplina**
- [Notebook Original - Aula 29 (Parte 1 e 2)]

