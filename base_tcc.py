# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 18:55:31 2025

@author: Usuario
"""

#%% In[1.0]: Importação dos pacotes
# Bibliotecas padrão do Python
# import os

# # Manipulação de dados
# import pandas as pd  # Manipulação de DataFrames
# import numpy as np  # Operações matemáticas

# # Estatísticas e testes estatísticos
# import scipy.stats as stats  # Testes estatísticos gerais
# from scipy.stats import pearsonr, boxcox, norm, zscore  # Correlações, transformações e distribuições
# from statsmodels.api import OLS  # Estimação de modelos estatísticos
# from statsmodels.iolib.summary2 import summary_col  # Comparação entre modelos
# from statsmodels.stats.outliers_influence import variance_inflation_factor  # Multicolinearidade
# import statsmodels.api as sm
# from statstests.process import stepwise  # Procedimento Stepwise
# from statstests.tests import shapiro_francia  # Teste de Shapiro-Francia
# import pingouin as pg  # Análises estatísticas adicionais


# # Machine Learning
# from sklearn.preprocessing import LabelEncoder, StandardScaler  # Transformação de dados
# from sklearn.model_selection import train_test_split  # Separação de dados
# from sklearn.linear_model import LinearRegression  # Regressão Linear
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de avaliação
# from sklearn.cluster import KMeans  # Algoritmo de agrupamento K-Means

# # Visualização de dados
# import seaborn as sns  # Visualização gráfica
# import matplotlib.pyplot as plt  # Gráficos estáticos
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import plotly.graph_objects as go  # Gráficos interativos em 3D
# import plotly.express as px  # Visualizações interativas simplificadas
# import plotly.io as pio  # Configuração do Plotly
# pio.renderers.default = 'browser'

# # Outros
# import pycountry  # Informações sobre países
# from babel.core import Locale  # Localização e formatação de idiomas
# import networkx as nx
# from linearmodels.panel import PanelOLS

#%% In[2.0]: Carregamento do data frame atleta eventos

# # Carregar os dados
# dados_olimpiadas = pd.read_csv('athlete_events.csv')

#%% In[3.0]: Limpeza e estruturação dos dados
# dados_olimpiadas = dados_olimpiadas.drop(columns=['ID', 'Games'])

# # Alterar os nomes das colunas
# novos_nomes_colunas = [
#     'Nome', 'Sexo', 'Idade', 'Altura_cm', 'Peso_kg', 'Pais', 'Código_pais', 
#     'Ano', 'Estação', 'Cidade', 'Esporte', 'Evento', 'Medalha', 
# ]
# dados_olimpiadas.columns = novos_nomes_colunas

# # Filtrar apenas os Jogos de Verão e ano a partir de 2012
# olimpiadas_verao = dados_olimpiadas.query("Ano >= 2012 & Estação == 'Summer'")

# # IMC
# # Excluir linhas com valores ausentes em 'Height_m' e 'Weight'
# olimpiadas_verao = olimpiadas_verao.dropna(subset=['Altura_cm', 'Peso_kg'])
# # Calcular IMC (com conversão de altura para metros)
# olimpiadas_verao['Altura_m'] = olimpiadas_verao['Altura_cm'] / 100
# olimpiadas_verao['IMC'] = olimpiadas_verao['Peso_kg'] / (olimpiadas_verao['Altura_m'] ** 2)
# olimpiadas_verao.Pais.nunique()


# # Traduzir Medalhas e Estação
# olimpiadas_verao['Medalha'] = olimpiadas_verao['Medalha'].replace({'Silver': 'Prata', 'Gold': 'Ouro'})
# olimpiadas_verao['Estação'] = olimpiadas_verao['Estação'].replace('Summer', 'Verão')


#%% In[4.0]: Crregamento df_olimpiadas de verao parquet

# Carregar parquet olimpiadas de verao
# olimpiadas_verao = pd.read_parquet('olimpiadas_verao.parquet') 
 


#%% In[5.0]: Função traduzir pais

# # Remover caracteres indesejados dos nomes dos países
# olimpiadas_verao['Pais'] = olimpiadas_verao['Pais'].str.replace(r'-[123]', '', regex=True)

# def traduzir_nome_do_pais(nome, idioma_destino='pt_BR'):
#     try:
#         print(f"Processando: {nome}")  # Log para acompanhar o progresso
        
#         # Tentar encontrar o país por nome aproximado
#         correspondencias = pycountry.countries.search_fuzzy(nome)
        
#         if correspondencias:
#             codigo_pais = correspondencias[0].alpha_2  # Código ISO do país (ex: 'BR', 'US')
            
#             # Criar um objeto Locale para o idioma desejado
#             localidade = Locale.parse(idioma_destino)  # Ex: 'pt_BR' para português brasileiro
            
#             # Obter o nome traduzido do país
#             nome_traduzido = localidade.territories.get(codigo_pais, None)
            
#             if nome_traduzido:
#                 print(f"Tradução encontrada para '{nome}': {nome_traduzido}")
#                 return nome_traduzido
#             else:
#                 print(f"Nenhuma tradução disponível para '{nome}'")
#                 return nome  # Retorna o nome original se não houver tradução
#         else:
#             print(f"Nenhuma correspondência encontrada para: {nome}")
#             return None  # Retorna None se não houver correspondência
    
#     except Exception as e:
#         # Log detalhado do erro
#         print(f"Erro ao processar '{nome}': {e}")
#         return None  # Retorna None em caso de erro

# # Função para criar um DataFrame com países não traduzidos
# def criar_df_com_paises_nao_traduzidos(df, nome_coluna):
#     # Filtra os países que não foram traduzidos (None ou NaN)
#     nao_traduzidos = df[df[nome_coluna].isnull() | (df[nome_coluna] == '')]
#     return nao_traduzidos

# # Função para aplicar tradução manual usando um dicionário
# def aplicar_traducao_manual(df, nome_coluna, dicionario_traducao):
#     # Aplica o dicionário de tradução manualmente
#     df[nome_coluna] = df[nome_coluna].apply(lambda x: dicionario_traducao.get(x, x))
#     return df


#%% In[6.0]: Aplicar a tradução pais
# Passo 1: Traduzir automaticamente
# olimpiadas_verao['Pais_traduzido'] = olimpiadas_verao['Pais'].apply(traduzir_nome_do_pais)

# # Passo 2: Criar um DataFrame com países não traduzidos
# df_nao_traduzidos = criar_df_com_paises_nao_traduzidos(olimpiadas_verao, 'Pais_traduzido')

# # Mostrar os países que precisam de tradução manual
# print("Países não traduzidos:")
# print(df_nao_traduzidos)

# # Passo 3: Criar um dicionário manual para tradução dos países restantes
# dicionario_traducao_manual = {
#     'Turkey': 'Turquia',
#     'Cape Verde': 'Cabo Verde',
#     'Timor Leste': 'Timor Leste'

# }

# # Passo 3: Aplicar tradução manual
# aplicar_traducao_manual(
#     olimpiadas_verao, 
#     'Pais_traduzido', 
#     dicionario_traducao_manual
# ) 

# # Resultado final
# print("\nDataFrame final com traduções:")
# print(olimpiadas_verao)





#%% In[7.0]: Tradução eventos do data frame olimpiadas_verao
# valores_unicos_evento = olimpiadas_verao['Evento'].unique()
# print(valores_unicos_evento)


# traducoes_eventos = {
#     "Basketball Men's Basketball": "Basquete Masculino",
#     "Judo Men's Extra-Lightweight": "Judô Masculino - Peso Meio-Leve",
#     "Badminton Men's Singles": "Badminton Masculino - Simples",
#     "Sailing Women's Windsurfer": "Vela Feminina - Prancha à Vela",
#     "Athletics Men's Shot Put": "Atletismo Masculino - Arremesso de Peso",
#     "Weightlifting Women's Super-Heavyweight": "Halterofilismo Feminino - Super Pesado",
#     "Wrestling Men's Light-Heavyweight, Greco-Roman": "Luta Greco-Romana Masculina - Meio Pesado",
#     "Rowing Men's Lightweight Double Sculls": "Remo Masculino - Double Skiff Peso Leve",
#     "Sailing Men's Two Person Dinghy": "Vela Masculina - Dingue para Duas Pessoas",
#     "Athletics Men's 1,500 metres": "Atletismo Masculino - 1.500 Metros",
#     "Swimming Men's 100 metres Butterfly": "Natação Masculina - 100 Metros Borboleta",
#     "Swimming Men's 200 metres Butterfly": "Natação Masculina - 200 Metros Borboleta",
#     "Swimming Men's 4 x 100 metres Medley Relay": "Natação Masculina - Revezamento 4x100 Metros Medley",
#     "Football Women's Football": "Futebol Feminino",
#     "Equestrianism Mixed Jumping, Individual": "Hipismo Misto - Salto Individual",
#     "Athletics Women's Javelin Throw": "Atletismo Feminino - Lançamento de Dardo",
#     "Wrestling Men's Heavyweight, Freestyle": "Luta Livre Masculina, Peso Pesado",
#     "Gymnastics Men's Individual All-Around": "Ginástica Masculina, Individual Geral",
#     "Gymnastics Men's Floor Exercise": "Ginástica Masculina, Exercício no Solo",
#     "Gymnastics Men's Parallel Bars": "Ginástica Masculina, Barras Paralelas",
#     "Gymnastics Men's Horizontal Bar": "Ginástica Masculina, Barra Fixa",
#     "Gymnastics Men's Rings": "Ginástica Masculina, Argolas",
#     "Gymnastics Men's Pommelled Horse": "Ginástica Masculina, Cavalo com Alças",
#     "Taekwondo Men's Flyweight": "Taekwondo Masculino, Peso Mosca",
#     "Athletics Men's 5,000 metres": "Atletismo Masculino, 5.000 metros",
#     "Rowing Men's Coxless Pairs": "Remo Masculino, Casal Sem Cargueiro",
#     "Fencing Men's epee, Individual": "Esgrima Masculina, Florete Individual",
#     "Taekwondo Women's Flyweight": "Taekwondo Feminino, Peso Mosca",
#     "Basketball Women's Basketball": "Basquete Feminino",
#     "Diving Men's Platform": "Saltos Masculinos, Plataforma",
#     "Canoeing Men's Canadian Doubles, 500 metres": "Canoagem Masculina, Duplas Canadenses, 500 metros",
#     "Canoeing Men's Canadian Doubles, 1,000 metres": "Canoagem Masculina, Duplas Canadenses, 1.000 metros",
#     "Canoeing Men's Kayak Fours, 1,000 metres": "Canoagem Masculina, Canoa Quádrupla, 1.000 metros",
#     "Handball Men's Handball": "Handebol Masculino",
#     "Rowing Women's Coxless Pairs": "Remo Feminino, Casal Sem Cargueiro",
#     "Football Men's Football": "Futebol Masculino",
#     "Water Polo Men's Water Polo": "Pólo Aquático Masculino",
#     "Tennis Men's Doubles": "Tênis Masculino, Duplas",
#     "Wrestling Men's Featherweight, Freestyle": "Luta Livre Masculina, Peso Pena",
#     "Cycling Women's Sprint": "Ciclismo Feminino, Sprint",
#     "Cycling Women's 500 metres Time Trial": "Ciclismo Feminino, Prova de Velocidade 500 metros",
#     "Athletics Men's 110 metres Hurdles": "Atletismo Masculino, 110 metros com Barreiras",
#     "Athletics Women's Marathon": "Atletismo Feminino, Maratona",
#     "Athletics Men's 100 metres": "Atletismo Masculino, 100 metros",
#     "Fencing Men's Sabre, Team": "Esgrima Masculina, Sabre, Equipe",
#     "Boxing Men's Welterweight": "Boxe Masculino, Peso Meio-Médio",
#     "Boxing Men's Middleweight": "Boxe Masculino, Peso Médio",
#     "Rowing Men's Double Sculls": "Remo Masculino, Dupla de Remo",
#     "Rowing Men's Quadruple Sculls": "Remo Masculino, Quádrupla de Remo",
#     "Rowing Men's Coxed Pairs": "Remo Masculino, Casal com Cargueiro",
#     "Rowing Men's Coxed Eights": "Remo Masculino, Oito com Cargueiro",
#     "Athletics Men's 400 metres": "Atletismo Masculino, 400 metros",
#     "Athletics Men's Hammer Throw": "Atletismo Masculino, Lançamento de Martelo",
#     "Athletics Men's 400 metres Hurdles": "Atletismo Masculino, 400 metros com Barreiras",
#     "Cycling Men's Road Race, Individual": "Ciclismo Masculino, Prova de Estrada Individual",
#     "Swimming Men's 100 metres Freestyle": "Natação Masculina, 100 metros Livre",
#     "Weightlifting Men's Middleweight": "Halterofilismo Masculino, Peso Médio",
#     "Hockey Men's Hockey": "Hóquei Masculino",
#     "Rowing Women's Single Sculls": "Remo Feminino, Skiff Individual",
#     "Swimming Men's 50 metres Freestyle": "Natação Masculina, 50 metros Livre",
#     "Weightlifting Women's Featherweight": "Halterofilismo Feminino, Peso Pena",
#     "Water Polo Women's Water Polo": "Pólo Aquático Feminino",
#     "Handball Women's Handball": "Handebol Feminino",
#     "Weightlifting Men's Heavyweight": "Halterofilismo Masculino, Peso Pesado",
#     'Sailing Mixed Three Person Keelboat': 'Vela Mista, Barco de Quilha para Três Pessoas',
#     'Equestrianism Mixed Three-Day Event, Individual': 'Hipismo Misturado, Evento de Três Dias, Individual',
#     'Equestrianism Mixed Three-Day Event, Team': 'Hipismo Misturado, Evento de Três Dias, Equipe',
#     "Sailing Women's Three Person Keelboat": "Vela Feminina, Barco de Quilha para Três Pessoas",
#     "Cycling Women's Road Race, Individual": "Ciclismo Feminino, Prova de Estrada Individual",
#     "Softball Women's Softball": "Softbol Feminino",
#     "Archery Women's Individual": "Arqueria Feminina, Individual",
#     "Wrestling Men's Heavyweight, Greco-Roman": "Luta Greco-Romana Masculina, Peso Pesado",
#     "Volleyball Men's Volleyball": "Vôlei Masculino",
#     "Synchronized Swimming Women's Duet": "Nado Sincronizado Feminino, Dueto",
#     "Synchronized Swimming Women's Team": "Nado Sincronizado Feminino, Equipe",
#     "Taekwondo Women's Featherweight": "Taekwondo Feminino, Peso Galo",
#     "Diving Women's Platform": "Saltos Femininos, Plataforma",
#     "Shooting Men's Air Rifle, 10 metres": "Tiro Masculino, Rifle de Ar, 10 metros",
#     "Wrestling Men's Super-Heavyweight, Greco-Roman": "Luta Greco-Romana Masculina, Super Peso Pesado",
#     "Shooting Men's Free Pistol, 50 metres": "Tiro Masculino, Pistola Livre, 50 metros",
#     "Shooting Men's Air Pistol, 10 metres": "Tiro Masculino, Pistola de Ar, 10 metros",
#     "Fencing Men's epee, Team": "Esgrima Masculina, Espada, Equipe",
#     "Rowing Men's Coxless Fours": "Remo Masculino, Quatro Sem Cargueiro",
#     "Boxing Men's Light-Flyweight": "Boxe Masculino, Peso Minimosca",
#     "Boxing Men's Super-Heavyweight": "Boxe Masculino, Super Peso Pesado",
#     "Shooting Women's Air Rifle, 10 metres": "Tiro Feminino, Rifle de Ar, 10 metros",
#     "Boxing Men's Lightweight": "Boxe Masculino, Peso Leve",
#     "Judo Men's Half-Middleweight": "Judô Masculino, Meio Médio",
#     "Weightlifting Men's Middle-Heavyweight": "Halterofilismo Masculino, Peso Médio-Pesado",
#     "Wrestling Men's Lightweight, Greco-Roman": "Luta Greco-Romana Masculina, Peso Leve",
#     "Athletics Men's Javelin Throw": "Atletismo Masculino, Lançamento de Dardo",
#     "Gymnastics Men's Horse Vault": "Ginástica Masculina, Salto sobre Cavalo",
#     "Athletics Men's 800 metres": "Atletismo Masculino, 800 metros",
#     "Volleyball Women's Volleyball": "Vôlei Feminino",
#     "Wrestling Men's Welterweight, Greco-Roman": "Luta Greco-Romana Masculina, Peso Meio-Médio",
#     "Wrestling Men's Middleweight, Greco-Roman": "Luta Greco-Romana Masculina, Peso Médio",
#     "Athletics Men's 10,000 metres": "Atletismo Masculino, 10.000 metros",
#     "Athletics Men's 3,000 metres Steeplechase": "Atletismo Masculino, 3.000 metros com Obstáculos",
#     "Athletics Men's Marathon": "Atletismo Masculino, Maratona",
#     "Wrestling Men's Middleweight, Freestyle": "Luta Livre Masculina, Peso Médio",
#     "Wrestling Men's Light-Heavyweight, Freestyle": "Luta Livre Masculina, Peso Meio-Pesado",
#     "Modern Pentathlon Men's Individual": "Pentatlo Moderno Masculino, Individual",
#     "Modern Pentathlon Men's Team": "Pentatlo Moderno Masculino, Equipe",
#     "Athletics Women's 200 metres": "Atletismo Feminino, 200 metros",
#     "Table Tennis Women's Singles": "Tênis de Mesa Feminino, Individual",
#     "Table Tennis Women's Doubles": "Tênis de Mesa Feminino, Duplas",
#     "Shooting Men's Skeet": "Tiro Masculino, Skeet",
#     "Athletics Men's 4 x 400 metres Relay": "Revezamento 4 x 400 metros Masculino",
#     "Athletics Women's 100 metres": "Atletismo Feminino, 100 metros",
#     "Weightlifting Women's Lightweight": "Levantamento de Peso Feminino Leve",
#     "Athletics Women's Long Jump": "Salto em Distância Feminino Atletismo",
#     "Fencing Women's epee, Individual": "Esgrima Feminino Epee, Individual",
#     "Swimming Men's 200 metres Individual Medley": "Natação Masculino 200 metros Medley Individual",
#     "Swimming Men's 400 metres Individual Medley": "Natação Masculino 400 metros Medley Individual",
#     "Boxing Men's Heavyweight": "Boxe Masculino Peso Pesado",
#     "Wrestling Men's Bantamweight, Freestyle": "Luta Livre Masculino Peso Galo",
#     "Boxing Men's Light-Welterweight": "Boxe Masculino Meio-Médio Leve",
#     "Wrestling Men's Flyweight, Freestyle": "Luta Livre Masculino Peso Mosca",
#     "Athletics Women's 5,000 metres": "Atletismo Feminino 5.000 metros",
#     "Weightlifting Women's Light-Heavyweight": "Levantamento de Peso Feminino Meio-Pesado",
#     "Weightlifting Women's Heavyweight": "Levantamento de Peso Feminino Pesado",
#     "Weightlifting Men's Featherweight": "Levantamento de Peso Masculino Peso Pena",
#     "Weightlifting Men's Lightweight": "Levantamento de Peso Masculino Leve",
#     "Taekwondo Men's Featherweight": "Taekwondo Masculino Peso Pena",
#     "Taekwondo Men's Welterweight": "Taekwondo Masculino Peso Meio-Médio",
#     "Judo Men's Heavyweight": "Judô Masculino Peso Pesado",
#     "Boxing Men's Bantamweight": "Boxe Masculino Peso Galo",
#     "Fencing Men's Foil, Individual": "Esgrima Masculino Florete, Individual",
#     "Baseball Men's Baseball": "Beisebol Masculino",
#     "Cycling Men's 100 kilometres Team Time Trial": "Ciclismo Masculino 100 km Contrarrelógio por Equipes",
#     "Fencing Men's Sabre, Individual": "Esgrima Masculino Sabre, Individual",
#     "Rhythmic Gymnastics Women's Group": "Ginástica Rítmica Feminina Equipe",
#     "Diving Women's Springboard": "Saltos Ornamentais Feminino Plataforma",
#     "Diving Women's Synchronized Springboard": "Saltos Ornamentais Feminino Sincro",
#     "Gymnastics Women's Individual All-Around": "Ginástica Feminina Individual Geral",
#     "Gymnastics Women's Team All-Around": "Ginástica Feminina Equipe Geral",
#     "Gymnastics Women's Floor Exercise": "Ginástica Feminina Solo",
#     "Gymnastics Women's Uneven Bars": "Ginástica Feminina Barras Assimétricas",
#     "Gymnastics Women's Balance Beam": "Ginástica Feminina Trave",
#     "Athletics Women's 10,000 metres": "Atletismo Feminino 10.000 metros",
#     "Athletics Men's Decathlon": "Atletismo Masculino Decatlo",
#     "Athletics Women's 4 x 100 metres Relay": "Revezamento 4 x 100 metros Feminino Atletismo",
#     "Athletics Women's 1,500 metres": "Atletismo Feminino 1.500 metros",
#     "Shooting Women's Air Pistol, 10 metres": "Tiro Feminino Pistola Ar, 10 metros",
#     "Shooting Women's Sporting Pistol, 25 metres": "Tiro Feminino Pistola Esportiva, 25 metros",
#     "Boxing Men's Flyweight": "Boxe Masculino Peso Mosca",
#     "Canoeing Men's Kayak Doubles, 500 metres": "Canoagem Masculina Canoa Dupla, 500 metros",
#     "Canoeing Men's Kayak Singles, 500 metres": "Canoagem Masculina Canoa Individual, 500 metros",
#     "Canoeing Men's Kayak Singles, 1,000 metres": "Canoagem Masculina Canoa Individual, 1.000 metros",
#     "Judo Women's Half-Heavyweight": "Judô Feminino Meio-Pesado",
#     "Athletics Women's Pole Vault": "Atletismo Feminino Salto com Vara",
#     "Rugby Sevens Women's Rugby Sevens": "Rugby Sevens Feminino",
#     "Table Tennis Men's Team": "Tênis de Mesa Masculino Equipe",
#     "Gymnastics Men's Team All-Around": "Ginástica Masculina Equipe Geral",
#     "Athletics Women's 4 x 400 metres Relay": "Revezamento 4 x 400 metros Feminino Atletismo",
#     "Swimming Men's 4 x 100 metres Freestyle Relay": "Natação Masculino Revezamento 4 x 100 metros Livre",
#     "Wrestling Men's Bantamweight, Greco-Roman": "Luta Greco-Romana Masculino Peso Galo",
#     "Athletics Men's Triple Jump": "Atletismo Masculino Salto Triplo",
#     "Fencing Men's Foil, Team": "Esgrima Masculino Florete, Equipe",
#     "Rowing Women's Lightweight Double Sculls": "Remo Feminino Dupla Leve",
#     "Athletics Women's 800 metres": "Atletismo Feminino 800 metros",
#     "Athletics Women's Shot Put": "Atletismo Feminino Arremesso de Peso",
#     "Rhythmic Gymnastics Women's Individual": "Ginástica Rítmica Feminina Individual",
#     "Canoeing Men's Kayak Singles, Slalom": "Canoagem Masculina Canoa Individual, Slalom",
#     "Archery Men's Individual": "Arco e Flecha Masculino Individual",
#     "Archery Men's Team": "Arco e Flecha Masculino Equipe",
#     "Athletics Women's 400 metres": "Atletismo Feminino 400 metros",
#     "Athletics Men's 200 metres": "Atletismo Masculino 200 metros",
#     "Trampolining Men's Individual": "Trampolim Masculino Individual",
#     "Beach Volleyball Men's Beach Volleyball": "Vôlei de Praia Masculino",
#     "Cycling Women's Mountainbike, Cross-Country": "Ciclismo Feminino Mountainbike, Cross-Country",
#     "Triathlon Women's Olympic Distance": "Triatlo Feminino Distância Olímpica",
#     "Cycling Men's Mountainbike, Cross-Country": "Ciclismo Masculino Mountainbike, Cross-Country",
#     "Judo Men's Lightweight": "Judô Masculino Leve",
#     "Swimming Men's 100 metres Backstroke": "Natação Masculino 100 metros Costas",
#     "Swimming Men's 400 metres Freestyle": "Natação Masculino 400 metros Livre",
#     "Table Tennis Men's Singles": "Tênis de Mesa Masculino Individual",
#     "Boxing Men's Featherweight": "Boxe Masculino Peso Pena",
#     "Rowing Women's Coxed Eights": "Remo Feminino Oito com Cocheiro",
#     "Rowing Women's Quadruple Sculls": "Remo Feminino Quádrupla",
#     "Athletics Men's Long Jump": "Atletismo Masculino Salto em Distância",
#     "Athletics Men's 4 x 100 metres Relay": "Revezamento 4 x 100 metros Masculino Atletismo",
#     "Weightlifting Women's Middleweight": "Levantamento de Peso Feminino Médio",
#     "Wrestling Women's Light-Heavyweight, Freestyle": "Luta Livre Feminino Meio Pesado, Estilo Livre",
#     "Athletics Women's High Jump": "Atletismo Feminino Salto em Altura",
#     "Swimming Men's 200 metres Freestyle": "Natação Masculino 200 metros Livre",
#     "Canoeing Women's Kayak Fours, 500 metres": "Canoagem Feminina Canoa Quádrupla, 500 metros",
#     "Sailing Men's One Person Dinghy": "Vela Masculina Dingue Individual",
#     "Trampolining Women's Individual": "Trampolim Feminino Individual",
#     "Shooting Mixed Skeet": "Tiro Skeet Misto",
#     "Judo Men's Half-Lightweight": "Judô Masculino Meio-Leve",
#     "Swimming Women's 50 metres Freestyle": "Natação Feminino 50 metros Livre",
#     "Swimming Women's 200 metres Butterfly": "Natação Feminino 200 metros Borboleta",
#     "Athletics Men's 20 kilometres Walk": "Atletismo Masculino 20 km Marcha",
#     "Athletics Men's 50 kilometres Walk": "Atletismo Masculino 50 km Marcha",
#     "Rowing Women's Double Sculls": "Remo Feminino Dupla",
#     "Boxing Women's Flyweight": "Boxe Feminino Peso Mosca",
#     "Athletics Women's 100 metres Hurdles": "Atletismo Feminino 100 metros Com Barreiras",
#     'Sailing Mixed One Person Dinghy': "Vela Mista Dingue Individual",
#     "Boxing Men's Light-Heavyweight": "Boxe Masculino Meio Pesado",
#     "Wrestling Women's Heavyweight, Freestyle": "Luta Livre Feminino Peso Pesado, Estilo Livre",
#     'Badminton Mixed Doubles': "Badminton Duplas Mistas",
#     "Swimming Women's 200 metres Backstroke": "Natação Feminino 200 metros Costas",
#     'Sailing Mixed Two Person Keelboat': 'Vela Mista Dois Pessoal Kielboat',
#     "Wrestling Men's Light-Flyweight, Freestyle": 'Luta Livre Masculina, Peso Mosca',
#     "Wrestling Women's Featherweight, Freestyle": 'Luta Livre Feminina, Peso Galo',
#     "Swimming Women's 200 metres Freestyle": 'Natação Feminina 200 metros Livre',
#     "Swimming Women's 400 metres Freestyle": 'Natação Feminina 400 metros Livre',
#     "Swimming Women's 200 metres Individual Medley": 'Natação Feminina 200 metros Medley Individual',
#     "Swimming Women's 400 metres Individual Medley": 'Natação Feminina 400 metros Medley Individual',
#     "Rugby Sevens Men's Rugby Sevens": 'Rugby Sevens Masculino',
#     "Wrestling Women's Lightweight, Freestyle": 'Luta Livre Feminina, Peso Leve',
#     "Modern Pentathlon Women's Individual": 'Pentatlo Moderno Feminino Individual',
#     "Canoeing Men's Canadian Doubles, Slalom": 'Canoagem Masculina Dobro Canadense, Slalom',
#     "Judo Women's Half-Lightweight": 'Judô Feminino Meio-Leve',
#     "Athletics Men's High Jump": 'Atletismo Masculino Salto em Altura',
#     'Sailing Mixed Two Person Heavyweight Dinghy': 'Vela Mista Dois Pessoal Dinghy Pesado',
#     "Swimming Women's 800 metres Freestyle": 'Natação Feminina 800 metros Livre',
#     "Sailing Women's Two Person Dinghy": 'Vela Feminina Dois Pessoal Dinghy',
#     "Wrestling Men's Featherweight, Greco-Roman": 'Luta Greco-Romana Masculina, Peso Pena',
#     "Athletics Women's Discus Throw": 'Atletismo Feminino Lançamento de Disco',
#     "Swimming Men's 4 x 200 metres Freestyle Relay": 'Natação Masculina Revezamento 4 x 200 metros Livre',
#     "Judo Women's Half-Middleweight": 'Judô Feminino Meio-Médio',
#     "Athletics Women's Heptathlon": 'Atletismo Feminino Heptatlo',
#     "Cycling Men's Points Race": 'Ciclismo Masculino Corrida de Pontos',
#     "Synchronized Swimming Women's Solo": 'Nado Sincronizado Feminino Solo',
#     "Swimming Women's 100 metres Backstroke": 'Natação Feminina 100 metros Costas',
#     'Equestrianism Mixed Dressage, Individual': 'Equitação Mista Adestramento, Individual',
#     "Cycling Men's Individual Pursuit, 4,000 metres": 'Ciclismo Masculino Perseguição Individual, 4.000 metros',
#     "Cycling Men's Team Pursuit, 4,000 metres": 'Ciclismo Masculino Perseguição por Equipes, 4.000 metros',
#     "Swimming Men's 100 metres Breaststroke": 'Natação Masculina 100 metros Peito',
#     "Tennis Men's Singles": 'Tênis Masculino Individual',
#     "Swimming Men's 200 metres Backstroke": 'Natação Masculina 200 metros Costas',
#     "Hockey Women's Hockey": 'Hóquei Feminino',
#     "Triathlon Men's Olympic Distance": 'Triatlo Masculino Distância Olímpica',
#     "Sailing Women's One Person Dinghy": 'Vela Feminina Um Pessoal Dinghy',
#     "Beach Volleyball Women's Beach Volleyball": 'Vôlei de Praia Feminino',
#     "Golf Men's Individual": 'Golfe Masculino Individual',
#     "Canoeing Men's Canadian Singles, 1,000 metres": 'Canoagem Masculina Individual Canadense, 1.000 metros',
#     "Archery Women's Team": 'Arco e Flecha Feminino por Equipes',
#     "Diving Men's Synchronized Platform": 'Mergulho Masculino Plataforma Sincronizada',
#     "Canoeing Women's Kayak Singles, Slalom": 'Canoagem Feminina Kayak Individual, Slalom',
#     "Weightlifting Women's Flyweight": 'Levantamento de Peso Feminino Peso Mosca',
#     'Equestrianism Mixed Jumping, Team': 'Equitação Mista Salto, por Equipes',
#     "Shooting Women's Small-Bore Rifle, Three Positions, 50 metres": 'Tiro Feminino Rifle de Pequeno Calibre, Três Posições, 50 metros',
#     "Swimming Women's 4 x 100 metres Freestyle Relay": 'Natação Feminina Revezamento 4 x 100 metros Livre',
#     "Swimming Women's 100 metres Butterfly": 'Natação Feminina 100 metros Borboleta',
#     "Swimming Women's 4 x 100 metres Medley Relay": 'Natação Feminina Revezamento 4 x 100 metros Medley',
#     "Athletics Women's 3,000 metres Steeplechase": 'Atletismo Feminino 3.000 metros Com Barreira',
#     "Shooting Women's Trap": 'Tiro Feminino Fossa',
#     "Diving Men's Springboard": 'Mergulho Masculino Prancha',
#     "Badminton Men's Doubles": 'Badminton Masculino Duplas',
#     "Sailing Men's Windsurfer": 'Vela Masculina Windsurf',
#     "Swimming Women's 4 x 200 metres Freestyle Relay": 'Natação Feminina Revezamento 4 x 200 metros Livre',
#     "Rowing Men's Single Sculls": 'Remo Masculino Skiff Individual',
#     "Athletics Women's 20 kilometres Walk": 'Atletismo Feminino Marcha 20 quilômetros',
#     "Sailing Men's One Person Heavyweight Dinghy": 'Vela Masculina Um Pessoal Dinghy Pesado',
#     "Sailing Women's Skiff": 'Vela Feminina Skiff',
#     "Cycling Men's Madison": 'Ciclismo Masculino Madison',
#     "Shooting Men's Small-Bore Rifle, Three Positions, 50 metres": 'Tiro Masculino Rifle de Pequeno Calibre, Três Posições, 50 metros',
#     "Shooting Men's Small-Bore Rifle, Prone, 50 metres": 'Tiro Masculino Rifle de Pequeno Calibre, Deitado, 50 metros',
#     "Canoeing Women's Kayak Doubles, 500 metres": 'Canoagem Feminina Kayak Duplo, 500 metros',
#     "Wrestling Men's Lightweight, Freestyle": 'Luta Livre Masculina, Peso Leve',
#     "Swimming Men's 1,500 metres Freestyle": 'Natação Masculina 1.500 metros Livre',
#     "Cycling Men's Individual Time Trial": 'Ciclismo Masculino Contra o Tempo Individual',
#     "Judo Women's Heavyweight": 'Judô Feminino Peso Pesado',
#     "Wrestling Men's Super-Heavyweight, Freestyle": 'Luta Livre Masculina, Super Pesado',
#     "Wrestling Men's Flyweight, Greco-Roman": 'Luta Greco-Romana Masculina, Peso Mosca',
#     "Swimming Women's 100 metres Breaststroke": 'Natação Feminina 100 metros Peito',
#     "Gymnastics Women's Horse Vault": 'Ginástica Feminina Cavalo com Arco',
#     "Table Tennis Men's Doubles": 'Tênis de Mesa Masculino Duplas',
#     "Athletics Women's 400 metres Hurdles": 'Atletismo Feminino 400 metros com Barreiras',
#     "Shooting Men's Rapid-Fire Pistol, 25 metres": 'Tiro Masculino Pistola de Fogo Rápido, 25 metros',
#     "Fencing Women's Sabre, Team": 'Esgrima Feminina Sabre, por Equipes',
#     "Swimming Women's 100 metres Freestyle": 'Natação Feminina 100 metros Livre',
#     "Shooting Men's Trap": 'Tiro Masculino Fossa',
#     "Shooting Men's Double Trap": 'Tiro Masculino Fossa Dupla',
#     "Athletics Men's Discus Throw": 'Atletismo Masculino Lançamento de Disco',
#     'Shooting Mixed Trap': 'Tiro Fossa Mista',
#     "Taekwondo Men's Heavyweight": 'Taekwondo Masculino Peso Pesado',
#     "Judo Men's Half-Heavyweight": 'Judô Masculino Meio-Pesado',
#     "Fencing Women's Foil, Individual": 'Esgrima Feminina Florete, Individual',
#     "Swimming Men's 200 metres Breaststroke": 'Natação Masculina 200 metros Peito',
#     "Judo Men's Middleweight": 'Judô Masculino Médio',
#     "Weightlifting Men's Super-Heavyweight": 'Levantamento de Peso Masculino Super Pesado',
#     'Sailing Mixed Multihull': 'Vela Mista Multicasco',
#     "Badminton Women's Singles": 'Badminton Feminino Individual',
#     "Badminton Women's Doubles": 'Badminton Feminino Duplas',
#     "Diving Women's Synchronized Platform": 'Mergulho Feminino Plataforma Sincronizada',
#     "Athletics Women's Triple Jump": 'Atletismo Feminino Salto Triplo',
#     "Rowing Men's Lightweight Coxless Fours": 'Remo Masculino Coxless Quatro Peso Leve',
#     "Boxing Women's Lightweight": 'Boxe Feminino Peso Leve',
#     "Cycling Women's BMX": 'Ciclismo Feminino BMX',
#     "Canoeing Men's Kayak Doubles, 200 metres": 'Canoagem Masculina Kayak Duplo, 200 metros',
#     "Canoeing Men's Kayak Doubles, 1,000 metres": 'Canoagem Masculina Kayak Duplo, 1.000 metros',
#     "Canoeing Women's Kayak Singles, 500 metres": 'Canoagem Feminina Kayak Individual, 500 metros',
#     "Athletics Women's 10 kilometres Walk": 'Atletismo Feminino Marcha 10 quilômetros',
#     "Weightlifting Men's Bantamweight": 'Levantamento de Peso Masculino Peso Galo',
#     "Diving Men's Synchronized Springboard": 'Mergulho Masculino Prancha Sincronizada',
#     "Shooting Men's Running Target, 10 metres": 'Tiro Masculino Alvo Móvel, 10 metros',
#     "Cycling Men's Sprint": 'Ciclismo Masculino Sprint',
#     "Cycling Men's 1,000 metres Time Trial": 'Ciclismo Masculino Contra o Tempo 1.000 metros',
#     "Cycling Men's BMX": 'Ciclismo Masculino BMX',
#     'Equestrianism Mixed Dressage, Team': 'Equitação Mista Adestramento, por Equipes',
#     "Taekwondo Women's Heavyweight": 'Taekwondo Feminino Peso Pesado',
#     "Judo Women's Middleweight": 'Judô Feminino Médio',
#     "Tennis Women's Singles": 'Tênis Feminino Individual',
#     "Judo Women's Lightweight": 'Judô Feminino Leve',
#     "Canoeing Women's Kayak Singles, 200 metres": 'Canoagem Feminina Kayak Individual, 200 metros',
#     "Fencing Women's epee, Team": 'Esgrima Feminina Espada, por Equipes',
#     'Sailing Mixed Skiff': 'Vela Mista Skiff',
#     "Swimming Women's 10 kilometres Open Water": 'Natação Feminina 10 quilômetros Águas Abertas',
#     "Athletics Men's Pole Vault": 'Atletismo Masculino Salto com Vara',
#     "Sailing Men's Skiff": 'Vela Masculina Skiff',
#     "Shooting Women's Skeet": 'Tiro Feminino Fossa Olímpica',
#     "Swimming Women's 200 metres Breaststroke": 'Natação Feminina 200 metros Peito',
#     "Judo Women's Extra-Lightweight": 'Judô Feminino Extra-Leve',
#     "Tennis Women's Doubles": 'Tênis Feminino Duplas',
#     "Taekwondo Women's Welterweight": 'Taekwondo Feminino Peso Meio-Médio',
#     "Cycling Women's Team Pursuit": 'Ciclismo Feminino Perseguição por Equipes',
#     "Cycling Women's Individual Time Trial": 'Ciclismo Feminino Contra o Tempo Individual',
#     "Fencing Women's Sabre, Individual": 'Esgrima Feminina Sabre, Individual',
#     "Cycling Men's Omnium": 'Ciclismo Masculino Omnium',
#     "Cycling Women's Individual Pursuit, 3,000 metres": 'Ciclismo Feminino Perseguição Individual, 3.000 metros',
#     "Cycling Women's Points Race": 'Ciclismo Feminino Corrida de Pontos',
#     "Sailing Men's Two Person Keelboat": 'Vela Masculina Dois Pessoal Kielboat',
#     "Cycling Men's Keirin": 'Ciclismo Masculino Keirin',
#     "Cycling Men's Team Sprint": 'Ciclismo Masculino Sprint por Equipes',
#     'Tennis Mixed Doubles': 'Tênis Duplas Mistas',
#     "Boxing Men's Light-Middleweight": 'Boxe Masculino Peso Meio-Médio',
#     "Golf Women's Individual": 'Golfe Feminino Individual',
#     "Wrestling Men's Welterweight, Freestyle": 'Luta Livre Masculina, Peso Meio-Médio',
#     "Canoeing Men's Canadian Singles, 500 metres": 'Canoagem Masculina Individual Canadense, 500 metros',
#     "Fencing Women's Foil, Team": 'Esgrima Feminina Florete, por Equipes',
#     "Wrestling Women's Flyweight, Freestyle": 'Luta Livre Feminina, Peso Mosca',
#     "Canoeing Men's Canadian Singles, Slalom": 'Canoagem Masculina Individual Canadense, Slalom',
#     "Swimming Men's 10 kilometres Open Water": 'Natação Masculina 10 quilômetros Águas Abertas',
#     "Wrestling Women's Middleweight, Freestyle": 'Luta Livre Feminina, Peso Médio',
#     "Table Tennis Women's Team": 'Tênis de Mesa Feminino por Equipes',
#     "Athletics Women's Hammer Throw": 'Atletismo Feminino Lançamento de Martelo',
#     "Canoeing Men's Kayak Singles, 200 metres": 'Canoagem Masculina Kayak Individual, 200 metros',
#     "Shooting Women's Double Trap": 'Tiro Feminino Fossa Dupla',
#     "Weightlifting Men's Heavyweight II": 'Levantamento de Peso Masculino Peso Pesado II',
#     "Boxing Women's Middleweight": 'Boxe Feminino Peso Médio',
#     "Rowing Women's Coxless Fours": 'Remo Feminino Quatro Sem Coxe',
#     "Weightlifting Men's Heavyweight I": 'Levantamento de Peso Masculino Peso Pesado I',
#     "Rowing Men's Coxed Fours": 'Remo Masculino Quatro Com Coxe',
#     "Cycling Women's Keirin": 'Ciclismo Feminino Keirin',
#     "Canoeing Men's Canadian Singles, 200 metres": 'Canoagem Masculina Individual Canadense, 200 metros',
#     "Cycling Women's Omnium": 'Ciclismo Feminino Omnium',
#     "Weightlifting Men's Flyweight": 'Levantamento de Peso Masculino Peso Mosca',
#     "Athletics Women's 3,000 metres": 'Atletismo Feminino 3.000 metros',
#     "Cycling Women's Team Sprint": 'Ciclismo Feminino Sprint por Equipes'
# }

# #com as traduções
# olimpiadas_verao['Evento'] = olimpiadas_verao['Evento'].replace(traducoes_eventos)


# # Apenas, para conferir valores que não foram substituídos
# excluidos_evento = olimpiadas_verao[~olimpiadas_verao['Evento'].isin(traducoes_eventos.values())]['Evento'].unique()
# print(excluidos_evento)

# # Função para padronizar
# def padronizar_texto(texto):
#     # Substituir hífens e vírgulas por espaços
#     texto = texto.replace("-", " ").replace(",", " ")
#     # Remover múltiplos espaços dividindo e juntando as palavras
#     texto = " ".join(texto.split())
#     return texto.strip()

# # Aplicar a padronização no DataFrame
# olimpiadas_verao['Evento'] = olimpiadas_verao['Evento'].apply(padronizar_texto)


#%% In[8.0]: Tradução esportes do data frame olimpiadas_verao
# valores_unicos_esporte = olimpiadas_verao['Esporte'].unique()
# print(valores_unicos_esporte)


# traducoes_esportes = {
#     'Basketball': 'Basquete',
#     'Judo': 'Judô',
#     'Badminton': 'Badminton',
#     'Sailing': 'Vela',
#     'Athletics': 'Atletismo',
#     'Weightlifting': 'Levantamento de Peso',
#     'Wrestling': 'Luta Livre',
#     'Rowing': 'Remo',
#     'Swimming': 'Natação',
#     'Football': 'Futebol',
#     'Equestrianism': 'Hipismo',
#     'Gymnastics': 'Ginástica',
#     'Taekwondo': 'Taekwondo',
#     'Fencing': 'Esgrima',
#     'Diving': 'Saltos Ornamentais',
#     'Canoeing': 'Canoagem',
#     'Handball': 'Handebol',
#     'Water Polo': 'Polo Aquático',
#     'Tennis': 'Tênis',
#     'Cycling': 'Ciclismo',
#     'Boxing': 'Boxe',
#     'Hockey': 'Hóquei',
#     'Softball': 'Softbol',
#     'Archery': 'Tiro com Arco',
#     'Volleyball': 'Voleibol',
#     'Synchronized Swimming': 'Nado Sincronizado',
#     'Shooting': 'Tiro Esportivo',
#     'Modern Pentathlon': 'Pentatlo Moderno',
#     'Table Tennis': 'Tênis de Mesa',
#     'Baseball': 'Beisebol',
#     'Rhythmic Gymnastics': 'Ginástica Rítmica',
#     'Rugby Sevens': 'Rúgbi de Sete',
#     'Trampolining': 'Ginástica de Trampolim',
#     'Beach Volleyball': 'Vôlei de Praia',
#     'Triathlon': 'Triatlo',
#     'Golf': 'Golfe'
# }

# #com as traduções
# olimpiadas_verao['Esporte'] = olimpiadas_verao['Esporte'].replace(traducoes_esportes)


# # Apenas, para conferir valores que não foram substituídos
# excluidos_esportes = olimpiadas_verao[~olimpiadas_verao['Esporte'].isin(traducoes_esportes.values())]['Esporte'].unique()
# print(excluidos_esportes)


# # Salvar em Parquet
# olimpiadas_verao.to_parquet('olimpiadas_verao.parquet')


#%% In[9.0]: Data frame e Tradução população

# # Carregar os dados
# population = pd.read_csv('pop1216.csv', delimiter=',')

# '''
# populacao = population.pivot_table(
#     index=['Country Name', 'Country Code'],  # Agrupa pelos nomes e códigos dos países
#     columns='Series Name',  # As colunas serão baseadas no nome da série
#     values=['2012 [YR2012]','2016 [YR2016]'],  
#     aggfunc='sum'  # Função de agregação (escolha 'sum' para somar ou 'mean' para média)
# ).reset_index()
# '''

# population.rename(columns={
#     'Country Name': 'Pais',
#     'Country Code': 'cod_pais',
#     #'Population, female': 'pop_feminina',
#     #'Population, male': 'pop_masculina',
#     '2012 [YR2012]' : 'pop_total_2012',
#     '2016 [YR2016]' : 'pop_total_2016'
# }, inplace=True)


# # Selecionar colunas relevantes e converter colunas numéricas
# populacao = population[['Pais', 'cod_pais', 'pop_total_2012', 'pop_total_2016']]
# populacao[['pop_total_2012', 'pop_total_2016']] = populacao[['pop_total_2012', 'pop_total_2016']].apply(pd.to_numeric, errors='coerce')


# populacao['Pais_traduzido'] = populacao['Pais'].apply(traduzir_nome_do_pais)

# # Exibir o DataFrame resultante
# print(populacao.head())
#%% In[10.0]: Data frame e Tradução pib


# # Carregar os dados de PIB
# pib = pd.read_csv('pib1216.csv', delimiter=',', on_bad_lines='skip')

# # Substituir valores não numéricos e remover linhas com valores ausentes no PIB de 2016
# pib['2016 [YR2016]'] = pd.to_numeric(pib['2016 [YR2016]'].replace('..', np.nan), errors='coerce')
# pib = pib.dropna(subset=['2016 [YR2016]']).reset_index(drop=True)

# # Substituir valores não numéricos e remover linhas com valores ausentes no PIB de 2012
# pib['2012 [YR2012]'] = pd.to_numeric(pib['2012 [YR2012]'].replace('..', np.nan), errors='coerce')
# pib = pib.dropna(subset=['2012 [YR2012]']).reset_index(drop=True)

# # Remover colunas que não serão utililizadas
# pib = pib.drop(columns=['Series Name', 'Series Code'])

# # Renomear colunas para maior clareza
# pib.rename(
#     columns={
#         '2016 [YR2016]': 'pib_dolar_2016',
#         '2012 [YR2012]': 'pib_dolar_2012',
#         'Country Code': 'cod_pais',
#         'Country Name': 'Pais'
#     },
#     inplace=True
# )


# pib['Pais_traduzido'] = pib['Pais'].apply(traduzir_nome_do_pais)


# # Verificar o resultado
# print(pib.head())


#%% In[11.0]: Data frame e Tradução education

# # Carregar os dados
# education = pd.read_csv('education.csv', delimiter=',')

# # Substituir valores não numéricos e remover linhas com valores ausentes no PIB de 2016
# education['2016 [YR2016]'] = pd.to_numeric(education['2016 [YR2016]'].replace('..', np.nan), errors='coerce')
# education = education.dropna(subset=['2016 [YR2016]']).reset_index(drop=True)

# # Substituir valores não numéricos e remover linhas com valores ausentes no PIB de 2012
# education['2012 [YR2012]'] = pd.to_numeric(education['2012 [YR2012]'].replace('..', np.nan), errors='coerce')
# education = education.dropna(subset=['2012 [YR2012]']).reset_index(drop=True)

# # Remover colunas que não serão utililizadas
# education = education.drop(columns=['Series Name', 'Series Code'])

# # Renomear colunas para maior clareza
# education.rename(
#     columns={
#         '2016 [YR2016]': 'gastos_edu_pib_perc_2016',
#         '2012 [YR2012]': 'gastos_edu_pib_perc_2012',
#         'Country Code': 'cod_pais',
#         'Country Name': 'Pais'
#     },
#     inplace=True
# )

# education['Pais_traduzido'] = education['Pais'].apply(traduzir_nome_do_pais)


# # Verificar o resultado
# print(education.head())

#%% In[12.0]: Marge dados dos paises_pop_pib_edu

# merge_pib_pop_edu = pd.merge(pd.merge(pib, populacao, how='inner', on='cod_pais'), education, how='inner', on='cod_pais')

# merge_pib_pop_edu.rename(columns={'pais_y': 'Pais'}, inplace=True)
# # Reordenar as colunas
# merge_pib_pop_edu = merge_pib_pop_edu[['Pais_traduzido', 'cod_pais', 'pib_dolar_2012', 'pib_dolar_2016', 'pop_total_2012', 'pop_total_2016','gastos_edu_pib_perc_2012', 'gastos_edu_pib_perc_2016']]
# merge_pib_pop_edu = merge_pib_pop_edu.dropna().reset_index(drop=True)


# #salvando merge 
# merge_pib_pop_edu.to_parquet('merge_pib_pop_edu.parquet')



#%% In[17.0]: df_pontuacao_ano


# # 1) Carrega o DataFrame pré‑processado
# olimpiadas_verao = pd.read_parquet('olimpiadas_verao.parquet')

# # 2) Mapeia corretamente as medalhas (em português) para pontos
# olimpiadas_verao['pontuacao'] = (
#     olimpiadas_verao['Medalha']
#       .map({'Ouro': 3, 'Prata': 2, 'Bronze': 1})
#       .fillna(0)
# )

# # 3) Filtra só 2012 e 2016
# df = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 4) Cria o DataFrame de todos os países‑ano que participaram
# todos_paises_ano = (
#     df[['Pais_traduzido','Ano']]
#     .drop_duplicates()
# )

# # 5) Extrai apenas os “medal events” (cada evento–medalha uma vez)
# df_med_eventos = (
#     df
#     .dropna(subset=['Medalha'])                            # só quem ganhou algo
#     .drop_duplicates(subset=['Pais_traduzido','Ano','Evento','Medalha'])
# )

# # 6) Conta quantas medalhas de cada tipo cada país ganhou
# medal_counts = (
#     df_med_eventos
#       .groupby(['Pais_traduzido','Ano','Medalha'])
#       .size()
#       .reset_index(name='count')
# )

# # 7) Pivot para ter colunas Ouro/Prata/Bronze
# medal_pivot = (
#     medal_counts
#       .pivot_table(
#           index=['Pais_traduzido','Ano'],
#           columns='Medalha',
#           values='count',
#           fill_value=0
#       )
#       .reset_index()
# )

# # 8) Calcula a pontuação total
# medal_pivot['pontuacao_total'] = (
#       medal_pivot.get('Ouro', 0) * 3
#     + medal_pivot.get('Prata', 0) * 2
#     + medal_pivot.get('Bronze', 0) * 1
# )

# # 9) Junta com todos_paises_ano para incluir zeros
# df_pontuacao_ano = (
#     todos_paises_ano
#       .merge(
#          medal_pivot[['Pais_traduzido','Ano','pontuacao_total']],
#          on=['Pais_traduzido','Ano'],
#          how='left'
#       )
#       .fillna({'pontuacao_total': 0})
# )

# # 10) Exibe o resultado e confere Brasil‑2016
# print(df_pontuacao_ano)
# print(df_pontuacao_ano.query("Pais_traduzido == 'Brasil' and Ano == 2016"))



#%% In[18.0]: df_socioeconomico
# Explode wide → long
# df_long = pd.wide_to_long(
#     merge_pib_pop_edu,
#     stubnames=['pib_dolar', 'pop_total', 'gastos_edu_pib_perc'],
#     i=['Pais_traduzido', 'cod_pais'],
#     j='Ano',
#     sep='_',
#     suffix='\\d+'
# ).reset_index()

# # Garante que Ano seja inteiro
# df_long['Ano'] = df_long['Ano'].astype(int)

# # Seleciona só as colunas de interesse
# df_socioeconomico = df_long[
#     ['Pais_traduzido', 'pib_dolar', 'pop_total', 'gastos_edu_pib_perc', 'Ano']
# ]

# print(df_socioeconomico.head())
#%% In[19.0]: df_participantes_unicos


# # 1) Filtra só 2012 e 2016 (se ainda não tiver feito)
# dados_filtrados = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Conta atletas únicos por país e ano
# df_participantes_unicos = (
#     dados_filtrados
#       .groupby(['Pais_traduzido', 'Ano'])['Nome']
#       .nunique()
#       .reset_index(name='num_atletas_unicos')
# )

# print(df_participantes_unicos)



#%% In[20.0]: df_idade_media_atletas

# # 1) Filtra só 2012 e 2016
# df = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Deduplica por atleta‑ano (cada Nome só aparece uma vez)
# df_ath = df.drop_duplicates(subset=['Pais_traduzido','Ano','Nome'])

# # 3) Agora calcula a idade média corretamente
# df_idade_media_atletas = (
#     df_ath
#     .groupby(['Pais_traduzido', 'Ano'])['Idade']
#     .mean()
#     .reset_index(name='idade_media')
# )

# print(df_idade_media_atletas)

#%% In[21.0]: df_eventos_por_pais_ano


# # 1) Já tendo filtrado só 2012 e 2016:
# dados_filtrados = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Conta eventos únicos por país e ano
# df_eventos_por_pais_ano = (
#     dados_filtrados
#       .groupby(['Pais_traduzido', 'Ano'])['Evento']
#       .nunique()
#       .reset_index(name='num_eventos')
# )

# print(df_eventos_por_pais_ano)


#%% In[22.0]: df_porc_feminino

# # 1) Filtrar só 2012 e 2016
# dados_filtrados = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Deduplicar por atleta‑ano (cada Nome só aparece uma vez)
# df_ath = dados_filtrados.drop_duplicates(subset=['Pais_traduzido','Ano','Nome'])

# # 3) Agrupar e calcular % de mulheres
# df_porc_feminino = (
#     df_ath
#     .groupby(['Pais_traduzido','Ano'])['Sexo']
#     .apply(lambda s: (s == 'F').mean() * 100)
#     .reset_index(name='porcentagem_feminino')
# )

# print(df_porc_feminino)



#%% In[23.0]: df_hhi
#hhi_esportes varia de ~0 (muita diversificação) a 1 (total especialização num único esporte).

# # 1) Filtrar só 2012 e 2016 (se ainda não fez)
# dados_filtrados = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Deduplicar por atleta–esporte, para que cada atleta conte apenas uma vez em cada esporte
# df_ath_sport = dados_filtrados.drop_duplicates(
#     subset=['Pais_traduzido', 'Ano', 'Nome', 'Esporte']
# )

# # 3) Função que calcula o HHI sobre uma Series de esportes
# def herfindahl(series):
#     freqs = series.value_counts(normalize=True)  # proporções s_i
#     return (freqs**2).sum()                      # soma(s_i^2)

# # 4) Aplicar por país e ano
# df_hhi = (
#     df_ath_sport
#       .groupby(['Pais_traduzido', 'Ano'])['Esporte']
#       .apply(herfindahl)
#       .reset_index(name='hhi_esportes')
# )

# print(df_hhi)
    
#%% In[24.0]: df_pct_jovens
# # 1) Filtra só 2012 e 2016
# df = olimpiadas_verao[olimpiadas_verao['Ano'].isin([2012, 2016])]

# # 2) Deduplica por atleta‑ano para não pesar múltiplas provas
# df_ath = df.drop_duplicates(subset=['Pais_traduzido','Ano','Nome'])

# # 3) Calcula a % de atletas com menos de 23 anos
# df_pct_jovens = (
#     df_ath
#     .groupby(['Pais_traduzido','Ano'])['Idade']
#     .apply(lambda s: (s < 23).mean() * 100)
#     .reset_index(name='pct_jovens_menor_23')
# )

# print(df_pct_jovens)
    
    
    

#%% In[24.0]: merge de todas as variaveis

# from functools import reduce

# dfs = [
#     df_participantes_unicos,
#     df_idade_media_atletas,
#     df_eventos_por_pais_ano,
#     df_porc_feminino,
#     df_hhi,
#     df_pct_jovens,
#     df_socioeconomico,
#     df_pontuacao_ano
# ]

# # Faz o merge de todos os DataFrames na lista, um a um
# df_agg = reduce(
#     lambda left, right: pd.merge(
#         left, right,
#         on=['Pais_traduzido', 'Ano'],
#         how='inner'   # ou 'outer' se quiser manter todos e preencher NaNs
#     ),
#     dfs
# )

# # Opcional: renomear colunas para padronizar
# df_agg = df_agg.rename(columns={
#     'Nome': 'num_atletas_unicos',    # caso não tenha renomeado antes
#     'Idade': 'idade_media',
#     # já devem estar com nomes corretos: num_eventos, porcentagem_feminino, hhi_esportes, pct_jovens_menor_23
# })

# print(df_agg.head())


# # #salvando merge 
# df_agg.to_parquet('df_agg.parquet')

#%% In[13.0]: Lendo o merge parquet
# df_agg = pd.read_parquet('df_agg.parquet') 