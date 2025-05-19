# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:14:30 2024

@author: Usuario
"""




#%% In[1.0] Crregamento bibliotecas
import os  # Biblioteca padrão

# Manipulação de dados
import numpy as np  # Operações numéricas
import pandas as pd  # Estruturas DataFrame

# Análise estatística e testes
import scipy.stats as stats  # Testes estatísticos gerais
from scipy.stats import boxcox, pearsonr, zscore, norm  # Transformações e correlações
import statsmodels.api as sm  # API principal do statsmodels
import statsmodels.formula.api as smf  # Interface de fórmulas para GLM
from statsmodels.iolib.summary2 import summary_col  # Comparação de modelos
from statsmodels.genmod.families import NegativeBinomial as NBFamily  # Família Binomial Negativa para GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Fator de inflação de variância (VIF)
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP  # Modelos ZIP e ZINB
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial  # Modelos de contagem Poisson e NB
from statstests.tests import overdisp
from statstests.process import stepwise  # Seleção stepwise
from statstests.tests import shapiro_francia, overdisp  # Teste de normalidade e superdispersão
import pingouin as pg  # Testes estatísticos adicionais
import statsmodels.discrete.discrete_model as dm  # API de modelos discretos (dm)

# Aprendizado de máquina
from sklearn.preprocessing import StandardScaler  # Padronização/Escalação
from sklearn.cluster import KMeans  # Algoritmo de agrupamento K-Means
from sklearn.linear_model import LinearRegression  # Regressão linear
from sklearn.model_selection import train_test_split  # Divisão de dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de avaliação

# Visualização
import matplotlib.pyplot as plt  # Gráficos estáticos
import matplotlib.cm as cm  # Mapas de cores
import matplotlib.colors as mcolors  # Utilitários de cor
import matplotlib.ticker as mtick  # Formatação de eixos
import seaborn as sns  # Plotagem estatística
import plotly.express as px  # Gráficos interativos simples
import plotly.graph_objects as go  # Figuras interativas personalizadas
import plotly.io as pio  # Configuração do Plotly
pio.renderers.default = 'browser'  # Rederizador padrão para Plotly

# Outros utilitários
import pycountry  # Metadados de países
from babel.core import Locale  # Localização e formatação de idiomas
import networkx as nx  # Análise de redes
from tabulate import tabulate  # Formatação de tabelas (tabulate.tabulate)
from tqdm import tqdm  # Barras de progresso


#%% In[2.0] Crregamento df_olimpiadas de verao parquet

# Carregar parquet olimpiadas de verao
olimpiadas_verao = pd.read_parquet('olimpiadas_verao.parquet') 
 


#%% In[3.0]: Função traduzir pais

# Remover caracteres indesejados dos nomes dos países
olimpiadas_verao['Pais'] = olimpiadas_verao['Pais'].str.replace(r'-[123]', '', regex=True)

def traduzir_nome_do_pais(nome, idioma_destino='pt_BR'):
    try:
        print(f"Processando: {nome}")  # Log para acompanhar o progresso
        
        # Tentar encontrar o país por nome aproximado
        correspondencias = pycountry.countries.search_fuzzy(nome)
        
        if correspondencias:
            codigo_pais = correspondencias[0].alpha_2  # Código ISO do país (ex: 'BR', 'US')
            
            # Criar um objeto Locale para o idioma desejado
            localidade = Locale.parse(idioma_destino)  # Ex: 'pt_BR' para português brasileiro
            
            # Obter o nome traduzido do país
            nome_traduzido = localidade.territories.get(codigo_pais, None)
            
            if nome_traduzido:
                print(f"Tradução encontrada para '{nome}': {nome_traduzido}")
                return nome_traduzido
            else:
                print(f"Nenhuma tradução disponível para '{nome}'")
                return nome  # Retorna o nome original se não houver tradução
        else:
            print(f"Nenhuma correspondência encontrada para: {nome}")
            return None  # Retorna None se não houver correspondência
    
    except Exception as e:
        # Log detalhado do erro
        print(f"Erro ao processar '{nome}': {e}")
        return None  # Retorna None em caso de erro

# Função para criar um DataFrame com países não traduzidos
def criar_df_com_paises_nao_traduzidos(df, nome_coluna):
    # Filtra os países que não foram traduzidos (None ou NaN)
    nao_traduzidos = df[df[nome_coluna].isnull() | (df[nome_coluna] == '')]
    return nao_traduzidos

# Função para aplicar tradução manual usando um dicionário
def aplicar_traducao_manual(df, nome_coluna, dicionario_traducao):
    # Aplica o dicionário de tradução manualmente
    df[nome_coluna] = df[nome_coluna].apply(lambda x: dicionario_traducao.get(x, x))
    return df

#%% In[4.0]: Lendo o merge parquet
merge_pib_pop_edu = pd.read_parquet('merge_pib_pop_edu.parquet') 


#%% In[5.0]: Análise de medalhas por IMC


# Passo 1: Estatísticas descritivas do IMC por esporte
imc_por_esporte = olimpiadas_verao.groupby('Esporte')['IMC'].describe()

# Passo 2: Visualizar distribuição de IMC por esporte
plt.figure(figsize=(15, 8))
sns.boxplot(data=olimpiadas_verao, x='Esporte', y='IMC', order=olimpiadas_verao['Esporte'].value_counts().index)

# Adicionar linhas horizontais para os valores de IMC 18,5 e 24,9
plt.axhline(y=18.5, color='r', linestyle='--', label='Abaixo do peso (IMC < 18.5)')
plt.axhline(y=24.9, color='g', linestyle='--', label='Peso normal (IMC <= 24.9)')
plt.axhline(y=30, color='r', linestyle='--', label='Acima do peso (IMC ≥ 30)')

# Adicionar título e rótulos
plt.title('Distribuição do IMC por Esporte', fontsize=16)
plt.xlabel('Esporte', fontsize=12)
plt.ylabel('IMC', fontsize=12)
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.show()

# Passo 3: Calcular porcentagem de competições e ganhos de medalhas
total_competicoes = olimpiadas_verao['Esporte'].count()
medalhas_por_esporte = olimpiadas_verao[olimpiadas_verao['Medalha'].notna()].groupby('Esporte')['Medalha'].count()
porcentagem_medalhas = (medalhas_por_esporte / total_competicoes) * 100

print("Porcentagem de medalhas por esporte:")
print(porcentagem_medalhas)

# Passo 4: Identificar outliers em cada esporte
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers_por_esporte = olimpiadas_verao.groupby('Esporte').apply(detect_outliers, 'IMC')

# Passo 5: Analisar a propensão para ganhar medalhas com outliers
propensao_medalhas_outliers = {}
for esporte in outliers_por_esporte.index.get_level_values('Esporte').unique():
    outliers = outliers_por_esporte.loc[esporte]
    total_outliers = len(outliers)
    medalhas_outliers = outliers['Medalha'].notna().sum()
    if total_outliers > 0:
        propensao_medalhas_outliers[esporte] = (medalhas_outliers / total_outliers) * 100

print("Propensão para ganhar medalhas com outliers por esporte:")
print(propensao_medalhas_outliers)

# Discussão final
esportes_com_outliers = list(propensao_medalhas_outliers.keys())
maior_propensao = max(propensao_medalhas_outliers.values())
esporte_maior_propensao = [esporte for esporte, propensao in propensao_medalhas_outliers.items() if propensao == maior_propensao]

print(f"O(s) esporte(s) com maior propensão para ganhar medalhas com outliers é/são: {esporte_maior_propensao} com uma propensão de {maior_propensao:.2f}%")


# Passo 3: Analisar IMC e desempenho por esporte
# Escolher um esporte específico, por exemplo, "Atletismo"
esporte_especifico = 'Ciclismo'
df_esporte = olimpiadas_verao[olimpiadas_verao['Esporte'] == esporte_especifico]

# Garantir que os valores NaN sejam tratados como uma categoria 'nan'
df_esporte['Medalha'] = df_esporte['Medalha'].fillna('Sem medalhas')

# Definir todas as categorias de medalhas
categorias_medalhas = ['Ouro', 'Prata', 'Bronze', 'Sem medalhas']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_esporte, x='Medalha', y='IMC', order=categorias_medalhas)

# Adicionar título e rótulos
plt.title(f'Relação entre IMC e Medalhas no {esporte_especifico}', fontsize=16)
plt.xlabel('Medalha', fontsize=12)
plt.ylabel('IMC', fontsize=12)
plt.grid(True)
plt.show()

#%% In[6.0]: Lendo o merge parquet
# 1) Carrega o CSV

df = pd.read_csv('df_agg.csv')
print("Total de linhas no df_agg.csv:", len(df))

# 2) Filtra apenas o ano de 2016
df = df[df['Ano'] == 2016].copy()
print("Total de linhas após filtrar Ano==2016:", len(df))  # deve dar 131



#%% In[7.0]: INSHTS

# 1) Carrega o CSV e armazena em df_agg
df_agg = pd.read_csv('df_agg.csv')

vars_analise = [
    'num_atletas_unicos',
    'idade_media',
    'num_eventos',
    'porcentagem_feminino',
    'hhi_esportes',
    'pct_jovens_menor_23',
    'pib_dolar',
    'pop_total',
    'gastos_edu_pib_perc',
    'pontuacao_total'
]

anos = sorted(df_agg['Ano'].unique())  # agora pega [2012, 2016]

for ano in anos:
    df_ano = df_agg[df_agg['Ano'] == ano]
    print(f'\n--- Ano {ano} ---')

    # 1) Correlações pontuacao_total vs cada variável
    for var in vars_analise:
        if var == 'pontuacao_total':
            continue
        corr, pval = pearsonr(df_ano[var], df_ano['pontuacao_total'])
        print(f'Correlação pontuacao_total × {var:20s}: {corr:6.2f} (p={pval:.4f})')

    # 2) Heatmap de correlação (matplotlib puro)
    corr_mat = df_ano[vars_analise].corr().values
    labels = vars_analise

    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr_mat, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{corr_mat[i, j]:.2f}", ha='center', va='center')
    plt.title(f'Matriz de Correlação – Olimpíadas {ano}')
    plt.tight_layout()
    plt.show()

    # 3) Scatter interativo (Plotly)
    fig = px.scatter(
        df_ano,
        x='pib_dolar',
        y='pontuacao_total',
        size='num_atletas_unicos',
        color='gastos_edu_pib_perc',
        hover_name='Pais_traduzido',
        title=f'Pontuação vs PIB – {ano}',
        labels={
            'pib_dolar': 'PIB (US$)',
            'pop_total': 'População',
            'gastos_edu_pib_perc': 'Educação (% PIB)',
            'pontuacao_total': 'Pontuação Total',
            'num_atletas_unicos': 'Nº Atletas'
        }
    )
    fig.update_layout(
        xaxis_title='PIB (US$)',
        yaxis_title='Pontuação Total',
        legend_title='Educação (% PIB)'
    )
    fig.show()
#%% In[8.0]: Diagrama interessante (grafo) que mostra a inter-relação entre as
#variáveis e a magnitude das correlações entre elas


# 1) Variáveis numéricas
vars_num = [
    'num_atletas_unicos','idade_media','num_eventos',
    'porcentagem_feminino','hhi_esportes','pct_jovens_menor_23',
    'pib_dolar','pop_total','gastos_edu_pib_perc','pontuacao_total'
]

# 2) Média 2012–2016
df_media = df_agg.groupby('Pais_traduzido')[vars_num].mean()

# 3) Matriz de correlação
corr_mat = df_media.corr()

# 4) Monta o grafo com arestas |corr|≥0.3
threshold = 0.3
G = nx.Graph()
G.add_nodes_from(vars_num)
edges = [
    (v1, v2, abs(corr_mat.loc[v1,v2]))
    for i,v1 in enumerate(vars_num)
    for v2 in vars_num[i+1:]
    if abs(corr_mat.loc[v1,v2]) >= threshold
]
G.add_weighted_edges_from(edges)

# 5) Normaliza larguras
weights = np.array([w for *_,w in edges])
min_w, max_w = weights.min(), weights.max()
norm_w = (weights - min_w)/(max_w - min_w + 1e-6)
edge_widths = 1 + norm_w*5

# 6) Layout e cores
pos = nx.kamada_kawai_layout(G)
cmap = plt.cm.coolwarm_r
norm = mcolors.Normalize(vmin=0, vmax=1)
edge_colors = [cmap(norm(w)) for *_,w in edges]

# 7) Plot com barra lateral
fig, (ax1, ax2) = plt.subplots(
    1,2, figsize=(14,8),
    gridspec_kw={'width_ratios':[4,0.3]}
)

nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='black', ax=ax1)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8, ax=ax1)

# labels deslocados para baixo
label_pos = {n:(x, y-0.12) for n,(x,y) in pos.items()}
nx.draw_networkx_labels(
    G, label_pos,
    font_size=10, font_color='black',
    verticalalignment='top',
    ax=ax1
)

# 7.1) Ajusta margem inferior para caber todos os labels
ax1.margins(y=0.2)

ax1.set_title('Grafo de Correlações (média 2012–2016)')
ax1.axis('off')

# 8) Barra de cores de 0 a 1
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(np.linspace(0,1,100))
cbar = fig.colorbar(sm, cax=ax2)
cbar.set_label('|Correlação|', rotation=270, labelpad=15)
ax2.yaxis.set_ticks_position('right')

plt.tight_layout()
plt.show()


#%% In[9.0]: Clustering 


# 1) Calcular pontuação e PIB médios por país
df_media = (
    df_agg
    .groupby('Pais_traduzido')[['pontuacao_total','pib_dolar']]
    .mean()
    .rename(columns={
        'pontuacao_total':'pontuacao_media',
        'pib_dolar':'pib_media'
    })
    .reset_index()
)

# 2) Converter PIB para trilhões de US$
df_media['pib_media_T'] = df_media['pib_media']

# 3) Preparar X e normalizar (usar colunas médias originais para clustering)
X = df_media[['pontuacao_media','pib_media_T']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Método do Cotovelo (opcional, para inspeção)
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo (média 2012–2016)')
plt.show()

# 5) Ajustar KMeans com k desejado
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_media['Cluster'] = kmeans.fit_predict(X_scaled)

# 6) Plot dos clusters, com formatação ajustada para eixo X
plt.figure(figsize=(12,8))
ax = sns.scatterplot(
    data=df_media,
    x='pib_media_T',
    y='pontuacao_media',
    hue='Cluster',
    palette='viridis',
    s=100
)

# Formatter para mostrar duas casas decimais em trilhões ("0.03T", "0.10T", etc.)
def trillions(x, pos):
    return f"{x:.2f}T"



# Anotar os 10 países com maior PIB médio
top10 = df_media.nlargest(10, 'pib_media_T')
for i, (_, row) in enumerate(top10.iterrows()):
    ha = 'right' if i % 2 == 0 else 'left'
    va = 'bottom' if i % 2 == 0 else 'top'
    ax.text(
        row['pib_media_T'],
        row['pontuacao_media'],
        row['Pais_traduzido'],
        fontsize=9,
        ha=ha,
        va=va,
        fontweight='bold'
    )

plt.title('Clusters de Países – PIB Médio vs Pontuação Média')
plt.xlabel('PIB Médio (US$ trilhões, média 2012–2016)')
plt.ylabel('Pontuação Média (2012–2016)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) (Opcional) Listar países de cada cluster
for cl in sorted(df_media['Cluster'].unique()):
    paises = df_media[df_media['Cluster']==cl]['Pais_traduzido'].tolist()
    print(f"Cluster {cl}: {', '.join(paises)}")
    
    
#%% In[10.0]: Regressão multipla


# Carrega e já filtra 2016

df_agg = pd.read_csv('df_agg.csv')

# filtra apenas 2016
df_2016 = df_agg[df_agg['Ano']==2016].copy()
print("Obs em 2016:", len(df_2016))  # 131


# Reescala se necessário
if df_2016['pib_dolar'].max() > 1e3:
    df_2016['pib_dolar'] = df_2016['pib_dolar'] / 1e9

if df_2016['pop_total'].max() > 1e3:
    df_2016['pop_total'] = df_2016['pop_total'] / 1e6


#%% In[10.1]: Regressão multipla
from scipy.stats import boxcox
# 2) Fórmula completa para OLS
fatores = (
    'pontuacao_total ~ num_atletas_unicos + idade_media + num_eventos + '
    'porcentagem_feminino + hhi_esportes + pct_jovens_menor_23 + '
    'pib_dolar + pop_total + gastos_edu_pib_perc'
)

# 3) Ajuste OLS completo
modelo = smf.ols(formula=fatores, data=df_2016).fit()
print(modelo.summary())

# 4) Stepwise
modelo_step = stepwise(modelo, pvalue_limit=0.05)
print(modelo_step.summary())

# 5) Shapiro–Francia nos resíduos do stepwise
sf1 = shapiro_francia(modelo_step.resid)
print(f"Shapiro–Francia (originais): W={sf1['statistics W']:.5f}, p‑value={sf1['p-value']:.6f}")

# 6) Box–Cox na variável dependente
y_min = df_2016['pontuacao_total'].min()
shift = abs(y_min) + 1e-6
bc_y, lmbda = boxcox(df_2016['pontuacao_total'] + shift)
print(f"Lambda estimado: {lmbda:.4f}")
df_2016['bc_pontuacao_total'] = bc_y

# 7) Limpa nomes do stepwise e monta fórmula para o modelo Box–Cox
vars_step = [v for v in modelo_step.model.exog_names if v!='Intercept']
formula_bc = 'bc_pontuacao_total ~ ' + ' + '.join(vars_step)
print("Usando fórmula BC:", formula_bc)

# 8) Ajuste OLS com a variável Box–Cox
modelo_bc = smf.ols(formula=formula_bc, data=df_2016).fit()
print(modelo_bc.summary())

# 9) Shapiro–Francia nos resíduos do modelo BC
sf_bc = shapiro_francia(modelo_bc.resid)
w_bc = sf_bc['statistics W']
p_bc = sf_bc['p-value']
alpha = 0.05

print(f"Shapiro–Francia (BC): W={w_bc:.5f}, p‑value={p_bc:.6f}")
if p_bc > alpha:
    print('Não se rejeita H0 — resíduos aderem à normalidade')
else:
    print('Rejeita-se H0 — resíduos não aderem à normalidade')
    
# 10) Q–Q Plot dos resíduos do modelo BC
plt.figure(figsize=(8, 6))
stats.probplot(modelo_bc.resid, dist="norm", plot=plt)
plt.title("Q–Q Plot dos Resíduos do Modelo Box–Cox")
plt.xlabel("Quantis Teóricos")
plt.ylabel("Resíduos")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% In[10.2]: Regressão multipla

import scipy.stats as stats
# 1) Extraia os resíduos do seu modelo:
resid = modelo_bc.resid

# 2) Rode o Shapiro–Francia
sf = shapiro_francia(resid)
w, p = sf['statistics W'], sf['p-value']
alpha = 0.05

# 3) Plote histograma (amarelo) + curva normal (preto)
plt.figure(figsize=(8, 6))
plt.hist(
    resid,
    bins=20,
    density=True,
    color='yellow',    # preenchimento amarelo
    edgecolor='black', # borda preta
    alpha=0.8
)
x = np.linspace(resid.min(), resid.max(), 200)
plt.plot(
    x,
    stats.norm.pdf(x, resid.mean(), resid.std()),
    color='black',
    linewidth=2
)

# 4) Anote o resultado do Shapiro–Francia
txt  = f"Shapiro–Francia: W={w:.5f}, p={p:.6f}\n"
txt += "Rejeita H0 — não normal" if p < alpha else "Não rejeita H0 — possivelmente normal"
plt.annotate(
    txt,
    xy=(0.05, 0.95),
    xycoords='axes fraction',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray')
)

# 5) Ajustes finais
plt.title("Distribuição dos Resíduos vs. Curva Normal")
plt.xlabel("Resíduos")
plt.ylabel("Densidade")
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
#%% In[11.0]: frequências da variável dependente pontuação total
contagem = df_2016['pontuacao_total'].value_counts(dropna=False)
percent = (df_2016['pontuacao_total'].value_counts(dropna=False, normalize=True)*100).round(2)
table = pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=True)
table


#%% In[11.1]: Visualização da tabela de frequências da variável dependente pontuação total


table.reset_index(level=0, inplace=True)
table.rename(columns={'index': 'n'}, inplace=True)


tabela = tabulate(table, headers='keys', tablefmt='grid', numalign='center')

plt.figure(figsize=(8, 3))
plt.text(0.1, 0.1, tabela, {'family': 'monospace', 'size': 15})
plt.axis('off')
plt.show()


#%% # In[12.0]: Histograma da variável dependente 'pontuacao_total'


with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,10))
    sns.histplot(data=df_2016, x='pontuacao_total', bins=20,
                 color='dodgerblue', edgecolor='white', kde=True)
    plt.xlabel('Quantidade de Pontução de medalhas', fontsize=20)
    plt.ylabel('Contagem', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


#%% # In[13]: Diagnóstico preliminar para observação de eventual igualdade entre a
#média e a variância da variável dependente '

pd.DataFrame({'Média':[df_2016.pontuacao_total.mean()],
              'Variância':[df_2016.pontuacao_total.var()]})


#%% In[13]: Estimação do modelo Poisson

import statsmodels.api as sm
import statsmodels.formula.api as smf
modelo_poisson = smf.glm(
    formula='pontuacao_total ~ num_atletas_unicos + idade_media + num_eventos + '
    'porcentagem_feminino + hhi_esportes + pct_jovens_menor_23 + '
    'pib_dolar + pop_total + gastos_edu_pib_perc',
    data=df_2016,
    family=sm.families.Poisson()
).fit()
print("LLF:", modelo_poisson.llf)  
print("AIC:", modelo_poisson.aic)  
modelo_poisson.summary()
#%% In[13.1]: modelo sem pop_total

modelo_poisson2 = smf.glm(formula='pontuacao_total ~  + idade_media + num_eventos + hhi_esportes + pct_jovens_menor_23 + pib_dolar',
                         data=df_2016,
                         family=sm.families.Poisson()).fit()

# Parâmetros do 'modelo_poisson'
modelo_poisson2.summary()

 
#%% In[13.2] Outro modo mais completo de apresentar os outputs do modelo,
#pela função 'summary_col'


summary_col([modelo_poisson2],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
        })

#%% In[14.0] Teste de superdispersão de Cameron e Trivedi

# nível de significância de 5%,


##############################################################################
#           TESTE DE SUPERDISPERSÃO DE CAMERON E TRIVEDI (1990)              #
##############################################################################
# CAMERON, A. C.; TRIVEDI, P. K. Regression-based tests for overdispersion in
#the Poisson model. Journal of Econometrics, v. 46, n. 3, p. 347-364, 1990.

# 1º Passo: estimar um modelo Poisson;
# 2º Passo: criar uma nova variável (Y*) utilizando os fitted values do modelo
#Poisson estimado anteriormente;
# 3º Passo: estimar um modelo auxiliar OLS, com a variável Y* como variável
#dependente, os fitted values do modelo Poisson como única variável preditora e 
#sem o intercepto;
# 4º Passo: Observar a significância do parâmetro beta.

# Adicionando os fitted values do modelo Poisson ('lambda_poisson') ao dataframe
df_2016['lambda_poisson'] = modelo_poisson2.fittedvalues
df_2016

# Criando a nova variável Y* ('ystar')
df_2016['ystar'] = (((df_2016['pontuacao_total']
                            -df_2016['lambda_poisson'])**2)
                          -df_2016['pontuacao_total'])/df_2016['lambda_poisson']
df_2016

# Estimando o modelo auxiliar OLS, sem o intercepto
modelo_auxiliar = sm.OLS.from_formula('ystar ~ 0 + lambda_poisson',
                                      df_2016).fit()

# Parâmetros do 'modelo_auxiliar'
modelo_auxiliar.summary()

# Caso o p-value do parâmetro de lambda_poisson seja maior que 0.05,
#verifica-se a existência de equidispersão nos dados.
# Caso contrário, diagnostica-se a existência de superdispersão nos dados, fato
#que favorecerá a estimação de um modelo binomial negativo, como ocorre nesse
#caso.

#%% In[15.0]: Função 'overdisp'

# Uma abordagem mais direta para a detecção da superdispersão pelo Teste de
#Cameron e Trivedi (1990) é por meio da utilização da função 'overdisp' do
#pacote 'statstests.tests'

# Instalação e carregamento da função 'overdisp' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/



# Elaboração direta do teste de superdispersão
overdisp(modelo_poisson2, df_2016)

#%% In[16.0]: Estimação do Binomial Negativa


# # 1) Matriz de preditoras (com constante) e vetor resposta
# X = sm.add_constant(df_2016[['idade_media', 'num_eventos', 'hhi_esportes', 'pct_jovens_menor_23', 'pib_dolar']])
# y = df_2016['pontuacao_total']


# # 2) Ajuste do NB2 via MLE
# nb2_mle = dm.NegativeBinomial(y, X).fit(disp=False)


# # 3) Sumário completo para conferir β’s e testes
# print(nb2_mle.summary())

# # 4) Extração do α estimado
# alpha_estimado = nb2_mle.params.get('alpha')
# print("α estimado (dispersion parameter):", alpha_estimado)

# #%% In[16.1]: Estimação do modelo binomial negativo do tipo NB2

# # O argumento 'family=sm.families.NegativeBinomial(alpha=2.0963)' da função
# # 'smf.glm' define a estimação de um modelo binomial negativo do tipo NB2
# # com valor de 'fi' ('alpha' no Python) igual a 2.0963 (valor proveniente da
# # estimação realizada por meio do Solver do Excel). Lembramos que 'fi' é o
# # inverso do parâmetro de forma 'theta' da distribuição Poisson-Gama.

# # use o α que você estimou com o MLE do NegativeBinomial (≈10.5546)
# alpha_meu = 2.0312

# modelo_bneg_glm = smf.glm(
#     formula='pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar',
#     data=df_2016,
#     family=NBFamily(alpha=alpha_meu)
# ).fit()

# print(modelo_bneg_glm.summary())


# #%% In[16.2]: Construção de uma função para a definição do 'fi' ótimo (argumento 'alpha')
# # que gera a maximização do valor de Log-Likelihood

# # Tempo aproximado de estimação desta célula: 1 min 40 seg



# n_samples = 5000
# alphas = np.linspace(0, 10, n_samples)
# llf = np.full(n_samples, fill_value=np.nan)

# for i, alpha in tqdm(enumerate(alphas), total=n_samples, desc='Estimating'):
#     try:
#         model = smf.glm(formula='pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar',
#                         data=df_2016,
#                         family=sm.families.NegativeBinomial(alpha=alpha)).fit()
#     except:
#         continue
#     llf[i] = model.llf

# fi_ótimo = alphas[np.nanargmax(llf)].round(4)
# fi_ótimo




# #%% In[16.3]: Plotagem dos resultados (Log-likelihood x fi)

# plt.figure(figsize=(12, 8))
# with plt.style.context('seaborn-v0_8-whitegrid'):
#     plt.plot(alphas, llf, label='Log-Likelihood', color='darkorchid', linewidth = 4)
#     plt.axvline(x=fi_ótimo, color='darkorange', linewidth = 4, linestyle='dashed',
#             label=f'$\phi$ ótimo: {round(fi_ótimo, 4)}')
# plt.xlabel('alpha', fontsize=20, style='italic')
# plt.ylabel('Log-Likelihood', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(loc='lower right', fontsize=17)
# plt.show()


# #%% In[16.4]: stimação do modelo binomial negativo do tipo NB2'fi' ótimo
# modelo_bneg = smf.glm(formula='pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar',
#                       data=df_2016,
#                       family=sm.families.NegativeBinomial(alpha=fi_ótimo)).fit()

# # Parâmetros do 'modelo_bneg'
# modelo_bneg.summary()
#%%

modelo_bneg_direto = sm.NegativeBinomial.from_formula('pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar', 
                                                      data= df_2016).fit()

modelo_bneg_direto.summary()

#%% In[17.0]: [Inflation Model] — Criação da variável de zeros e estimação do Logit
df_2016['is_zero'] = (df_2016['pontuacao_total'] == 0).astype(int)




X_inf_test = df_2016[['num_eventos', 'hhi_esportes', 'pib_dolar', 'idade_media']]
X_inf_test = sm.add_constant(X_inf_test)

y_zero = df_2016['is_zero']

modelo_logit = sm.Logit(y_zero, X_inf_test).fit()
print(modelo_logit.summary())
#%% In[18.0]: Comparando os modelos Poisson e binomial negativo

summary_col([modelo_poisson2, modelo_bneg_direto], 
            model_names=["Poisson","BNeg"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
                })

#%% In[19.0]: Definição da função para realização do teste de razão de verossimilhança

# Definição da função 'lrtest'

def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1) # 1 grau de liberdade
    
    print("Likelihood Ratio Test:")
    print(f"-2.(LL0-LLm): {round(LR_statistic, 2)}")
    print(f"p-value: {p_val:.3f}")
    print("")
    print("==================Result======================== \n")
    if p_val <= 0.05:
        print("H1: Different models, favoring the one with the highest Log-Likelihood")
    else:
        print("H0: Models with log-likelihoods that are not statistically different at 95% confidence level")

#%% In[19.1]: Teste de de razão de verossimilhança para comparar as estimações dos
#'modelo_poisson' e 'modelo_bneg_direto'

lrtest([modelo_poisson2, modelo_bneg_direto])

#%% In[19.2]: Gráfico para a comparação dos LogLiks dos modelos Poisson e
#binomial negativo

# Definição do dataframe com os modelos e respectivos LogLiks
df_llf = pd.DataFrame({'modelo':['Poisson','BNeg'],
                      'loglik':[modelo_poisson2.llf, modelo_bneg_direto.llf]})
df_llf

# Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['indigo', 'darkgoldenrod']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=30)
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()



#%% In[20.0]: Adicionando os fitted values dos modelos estimados até o momento,
#para fins de comparação

# Modelo Poisson:
df_2016['fitted_poisson'] = modelo_poisson2.fittedvalues

# Modelo binomial negativo:
df_2016['fitted_bneg'] = modelo_bneg_direto.fittedvalues

# Visualização do dataframe com os fitted values do modelos estimados
df_2016[['num_eventos', 'hhi_esportes', 'pib_dolar',
         'pontuacao_total', 'fitted_poisson', 'fitted_bneg']]



#%% In[20.1]: Cálculo de resíduos e métricas de erro (MAE e RMSE)


# 1) Calcule os resíduos
df_2016['resid_pois'] = df_2016['pontuacao_total'] - df_2016['fitted_poisson']
df_2016['resid_bneg'] = df_2016['pontuacao_total'] - df_2016['fitted_bneg']

# 2) Erro absoluto médio (MAE)
mae_pois = df_2016['resid_pois'].abs().mean()
mae_bneg = df_2016['resid_bneg'].abs().mean()

# 3) Root Mean Squared Error (RMSE)
rmse_pois = np.sqrt((df_2016['resid_pois']**2).mean())
rmse_bneg = np.sqrt((df_2016['resid_bneg']**2).mean())

print(f"Poisson MAE: {mae_pois:.3f}   |   BNeg MAE: {mae_bneg:.3f}")
print(f"Poisson RMSE: {rmse_pois:.3f}  |   BNeg RMSE: {rmse_bneg:.3f}")



# In[21.0]: Estimação do modelo zip
##############################################################################
#              ESTIMAÇÃO DO MODELO ZERO-INFLATED POISSON (ZIP)               #
##############################################################################
# Força a reescala correta (em caso de sobrescrita de df_2016 em outras células)

# Estimação do modelo ZIP pela função 'ZeroInflatedPoisson' do pacote
#'statsmodels.discrete.count_model'



# 2. Define y e X para o ZIP


y = df_2016['pontuacao_total']
X1 = sm.add_constant(df_2016[['num_eventos', 'idade_media', 'hhi_esportes', 'pct_jovens_menor_23', 'pib_dolar']])
X2 = sm.add_constant(df_2016[['num_eventos']])  # parte inflate

# 3. Estima modelo ZIP com bfgs e alta iteração
modelo_zip = ZeroInflatedPoisson(endog=y, exog=X1, exog_infl=X2, inflation='logit')\
    .fit(method='bfgs', maxiter=1000, disp=1)

# 4. Resultados
print("LLF ZIP:", modelo_zip.llf)
print("AIC ZIP:", modelo_zip.aic)
print(modelo_zip.summary())


#%% In[21.0]: Definição de função para elaboração do teste de Vuong

# VUONG, Q. H. Likelihood ratio tests for model selection and non-nested
#hypotheses. Econometrica, v. 57, n. 2, p. 307-333, 1989.

# Definição de função para elaboração do teste de Vuong
# Autores: Luiz Paulo Fávero e Helder Prado Santos

def vuong_test(m1, m2):

    from scipy.stats import norm

    if m1.__class__.__name__ == "GLMResultsWrapper":
        
        glm_family = modelo_poisson.model.family

        X = pd.DataFrame(data=m1.model.exog, columns=m1.model.exog_names)
        y = pd.Series(m1.model.endog, name=m1.model.endog_names)

        if glm_family.__class__.__name__ == "Poisson":
            m1 = Poisson(endog=y, exog=X).fit()
            
        if glm_family.__class__.__name__ == "NegativeBinomial":
            m1 = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

    supported_models = [ZeroInflatedPoisson,ZeroInflatedNegativeBinomialP,Poisson,NegativeBinomial]
    
    if type(m1.model) not in supported_models:
        raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
        
    if type(m2.model) not in supported_models:
        raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
    
    # Extração das variáveis dependentes dos modelos
    m1_y = m1.model.endog
    m2_y = m2.model.endog

    m1_n = len(m1_y)
    m2_n = len(m2_y)

    if m1_n == 0 or m2_n == 0:
        raise ValueError("Could not extract dependent variables from models.")

    if m1_n != m2_n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         f"Model 1 has {m1_n} observations.\n"
                         f"Model 2 has {m2_n} observations.")

    if np.any(m1_y != m2_y):
        raise ValueError("Models appear to have different values on dependent variables.")
        
    m1_linpred = pd.DataFrame(m1.predict(which="prob"))
    m2_linpred = pd.DataFrame(m2.predict(which="prob"))        

    m1_probs = np.repeat(np.nan, m1_n)
    m2_probs = np.repeat(np.nan, m2_n)

    which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(m1_linpred.columns) else None for x in m1_y]    
    which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(m2_linpred.columns) else None for x in m2_y]

    for i, v in enumerate(m1_probs):
        m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

    for i, v in enumerate(m2_probs):
        m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

    lm1p = np.log(m1_probs)
    lm2p = np.log(m2_probs)

    m = lm1p - lm2p

    v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

    pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

    print("Vuong Non-Nested Hypothesis Test-Statistic (Raw):")
    print(f"Vuong z-statistic: {round(v, 3)}")
    print(f"p-value: {pval:.3f}")
    print("")
    print("==================Result======================== \n")
    if pval <= 0.05:
        print("H1: Indicates inflation of zeros at 95% confidence level")
    else:
        print("H0: Indicates no inflation of zeros at 95% confidence level")

#%% In[21.1]: Teste de Vuong propriamente dito para verificação de existência de
#inflação de zeros no modelo ZIP, em comparação com o modelo Poisson 2


vuong_test(modelo_poisson2, modelo_zip)

# Ocorrência de inflação de zeros!


#%% In[22.0]: Comparando os modelos Poisson e ZIP

summary_col([modelo_poisson2, modelo_zip], 
            model_names=["Poisson","ZIP"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
                })

#%% In[23.0]: Definição da função para realização do teste de razão de verossimilhança

# Definição da função 'lrtest'

def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 2) # 2 graus de liberdade
    
    print("Likelihood Ratio Test:")
    print(f"-2.(LL0-LLm): {round(LR_statistic, 2)}")
    print(f"p-value: {p_val:.3f}")
    print("")
    print("==================Result======================== \n")
    if p_val <= 0.05:
        print("H1: Different models, favoring the one with the highest Log-Likelihood")
    else:
        print("H0: Models with log-likelihoods that are not statistically different at 95% confidence level")

#%%  In[23.1]: Teste de de razão de verossimilhança para comparar as estimações dos
#'modelo_poisson' e 'modelo_zip'

lrtest([modelo_poisson2, modelo_zip])


#%% In[24.0]: Gráfico para a comparação dos LogLiks dos modelos Poisson,
#binomial negativo e ZIP

# Definição do dataframe com os modelos e respectivos LogLiks
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','BNeg'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg_direto.llf]})
df_llf

# Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['indigo', 'deeppink', 'darkgoldenrod']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=30)
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()
#%% In[25.0]: Estimação do modelo Zinb

##############################################################################
#        ESTIMAÇÃO DO MODELO ZERO-INFLATED BINOMIAL NEGATIVO (ZINB)          #
##############################################################################

# Estimação do modelo ZINB pela função 'ZeroInflatedNegativeBinomialP' do pacote
#'statsmodels.discrete.count_model'

# Definição da variável dependente (voltando ao dataset 'df_corruption')
y = df_2016['pontuacao_total']

# Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_2016[['num_eventos', 'idade_media', 'hhi_esportes', 'pct_jovens_menor_23', 'pib_dolar']]
X1 = sm.add_constant(x1)



# Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_2016[['num_eventos']]
X2 = sm.add_constant(x2)

# O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit()

# Parâmetros do 'modelo_zinb'
modelo_zinb.summary()

# O parâmetro 'alpha' representa o 'fi' e é o inverso do parâmetro 'theta',
#ou seja, o inverso do parâmetro de forma da distribuição Poisson-Gama.
# Como 'alpha' (e da mesma forma 'theta') é estatisticamente diferente de
#zero, podemos afirmar que há superdispersão nos dados (outra forma de
#verificar o fenômeno da superdispersão!)


#%% In[26.0]: Teste de Vuong para verificação de existência de inflação de zeros
#no modelo ZINB, em comparação com o modelo binomial negativo

vuong_test(modelo_bneg_direto, modelo_zinb)

# Ocorrência de inflação de zeros!


#%% In[27.0]: Sumário comparativo BNeg vs ZINB

summary_col([modelo_bneg_direto, modelo_zinb], 
            model_names=["BNeg","ZINB"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
                })



#%% In[28.0]: Teste de razão de verossimilhança para comparar as estimações dos
#'modelo_bneg_direto' e 'modelo_zinb' (função 'lrtest' definida anteriormente)

lrtest([modelo_bneg_direto, modelo_zinb])

#%% In[28.1]: Adicionando os fitted values dos modelos estimados para fins de
#comparação

df_2016['fitted_zip'] = modelo_zip.predict(X1, exog_infl=X2)
df_2016['fitted_zinb'] = modelo_zinb.predict(X1, exog_infl=X2)

df_2016[['pontuacao_total','fitted_poisson','fitted_bneg',
               'fitted_zip','fitted_zinb']]

#%% In[29.0]: Gráfico para a comparação dos LogLiks dos modelos Poisson,
#binomial negativo, ZIP e ZINB

# Definição do dataframe com os modelos e respectivos LogLiks
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','BNeg','ZINB'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg_direto.llf,
                                modelo_zinb.llf]})
df_llf

# Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ['indigo', 'deeppink', 'darkgoldenrod', 'darkorange']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=30)
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()



#%% In[30.0]: Comparação de LogLik incluindo o OLS (para fins didáticos)



# 1) Fórmula reduzida
formula_simples = 'pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar'

# 2) Estimar o modelo OLS
modelo_ols = smf.ols(formula=formula_simples, data=df_2016).fit()
print(modelo_ols.summary())

# 3) Teste de normalidade dos resíduos
sf_ols = shapiro_francia(modelo_ols.resid)
print(f"Shapiro–Francia (OLS): W={sf_ols['statistics W']:.5f}, p‑value={sf_ols['p-value']:.6f}")

# 4) Box–Cox na variável dependente
y_min = df_2016['pontuacao_total'].min()
shift = abs(y_min) + 1e-6
bc_y, lmbda = boxcox(df_2016['pontuacao_total'] + shift)
df_2016['bc_pontuacao_total'] = bc_y
print(f"Lambda estimado: {lmbda:.4f}")

# 5) Reajuste do modelo com a variável Box–Cox
modelo_bc = smf.ols(formula='bc_pontuacao_total ~ num_eventos + hhi_esportes + pib_dolar', data=df_2016).fit()
print(modelo_bc.summary())

# 6) Normalidade dos resíduos após Box–Cox
sf_bc = shapiro_francia(modelo_bc.resid)
print(f"Shapiro–Francia (Box–Cox): W={sf_bc['statistics W']:.5f}, p‑value={sf_bc['p-value']:.6f}")

# 7) Comparação de LogLik com os demais modelos
# DataFrame com os LogLik de todos os modelos
df_llf = pd.DataFrame({
    'modelo': ['OLS', 'Poisson', 'ZIP', 'BNeg', 'ZINB'],
    'loglik': [
        modelo_ols.llf,
        modelo_poisson.llf,
        modelo_zip.llf,
        modelo_bneg_direto.llf,
        modelo_zinb.llf
    ]
})

# Cores correspondentes
cores = ['gray', 'indigo', 'deeppink', 'darkgoldenrod', 'darkorange']

# Gráfico
fig, ax = plt.subplots(figsize=(14, 8))
barras = ax.barh(df_llf['modelo'], df_llf['loglik'], color=cores)

# Anotações dos valores no gráfico
for barra in barras:
    largura = barra.get_width()
    ax.text(largura - 20, barra.get_y() + barra.get_height()/2,
            f'{largura:.3f}', ha='right', va='center',
            fontsize=14, color='white', weight='bold')

# Configurações finais
ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=30)
ax.set_ylabel("Modelo Proposto", fontsize=20)
ax.set_xlabel("LogLik", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()


#%%
import numpy as np

# supondo que você já tenha:
# modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2, inflation='logit').fit()

# 1) extraia diretamente os coeficientes do seu modelo já ajustado
b_count = modelo_zinb.params['num_eventos']           # β do componente de contagem
b_infl  = modelo_zinb.params['inflate_num_eventos']   # β do componente de inflação

# 2) converta em variação percentual
pct_count = (np.exp(b_count) - 1) * 100
pct_infl  = (np.exp(b_infl)  - 1) * 100

print(f"Contagem: +{pct_count:.2f}% por evento adicional")
print(f"Inflação: {pct_infl:.2f}% de variação nas odds de zerar")





#%%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 1) Ordena o DataFrame pelo número de eventos
df_ord = df_2016.sort_values('num_eventos')

# 2) Extrai cada série
x      = df_ord['num_eventos']
y_real = df_ord['pontuacao_total']
y_pois = df_ord['fitted_poisson']
y_bneg = df_ord['fitted_bneg']
y_zip  = df_ord['fitted_zip']
y_zinb = df_ord['fitted_zinb']

# 3) Cria figura com GridSpec: 2 linhas, proporção 3:1
fig = plt.figure(constrained_layout=True, figsize=(12, 10))
gs  = GridSpec(4, 1, figure=fig)
ax_main = fig.add_subplot(gs[0:3, 0])  # ocupa 3/4 da altura
ax_zoom = fig.add_subplot(gs[3, 0])    # ocupa 1/4 da altura

# 4) Plot principal
ax_main.scatter(x, y_real, color='gray', s=40, alpha=0.7)
ax_main.plot(   x, y_pois, '-',  color='blue',   linewidth=2)
ax_main.plot(   x, y_bneg, '--', color='orange', linewidth=2)
ax_main.plot(   x, y_zip,  ':',  color='green',  linewidth=2)
ax_main.plot(   x, y_zinb, '-.', color='red',    linewidth=2)
ax_main.set_title('Fitted Values vs Observados para Modelos de Contagem', fontsize=18)
ax_main.set_ylabel('Pontuação Total', fontsize=14)
ax_main.grid(True, linestyle=':', alpha=0.5)

# 5) Plot de zoom (0–20 eventos) e legenda aqui embaixo
mask = x <= 20
h_pois, = ax_zoom.plot(x[mask], y_pois[mask],  '-',  color='blue',   linewidth=1.5)
h_bneg, = ax_zoom.plot(x[mask], y_bneg[mask], '--', color='orange', linewidth=1.5)
h_zip,  = ax_zoom.plot(x[mask], y_zip[mask],  ':',  color='green',  linewidth=1.5)
h_zinb, = ax_zoom.plot(x[mask], y_zinb[mask], '-.', color='red',    linewidth=1.5)
ax_zoom.scatter(x[mask], y_real[mask], color='gray', s=20, alpha=0.7)

ax_zoom.set_xlim(0, 20)
ax_zoom.set_xlabel('Número de Eventos (num_eventos)', fontsize=14)
ax_zoom.set_ylabel('Pontuação Total', fontsize=14)
ax_zoom.grid(True, linestyle=':', alpha=0.5)

# Legenda única embaixo
handles = [h_pois, h_bneg, h_zip, h_zinb]
labels  = ['Poisson', 'Binomial Negativo', 'ZIP', 'ZINB']
ax_zoom.legend(handles, labels, loc='center', ncol=4, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.4), bbox_transform=ax_zoom.transAxes)

plt.show()