import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from numpy import linalg as LA
from plotly import graph_objects as go
from bs4 import BeautifulSoup
import requests
import matplotlib.ticker as mtick
from scipy.optimize import minimize



lista_acoes_yf = ['ARZZ3.SA','ASAI3.SA','BBSE3.SA','CPFE3.SA','EGIE3.SA','HYPE3.SA','KEPL3.SA','LEVE3.SA','PRIO3.SA','PSSA3.SA','SLCE3.SA','VALE3.SA','VIVT3.SA']
pesos_carteira_eq = [0.05,0.06,0.05,0.10,0.05,0.05,0.08,0.08,0.05,0.08,0.02,0.07,0.05]



class Risco():
    def __init__(self):
        print('Hello world')

    def var_historico(self,periodo_inicial,period_final):


        lista_acoes = lista_acoes_yf
        pesos_carteira = pesos_carteira_eq


        carteira = yf.download(lista_acoes,start=periodo_inicial,end=period_final)['Adj Close']
        retornos = carteira.pct_change()
        retorno_da_carteira = (retornos*pesos_carteira).sum(axis=1)

        self.retorno_portfolio = pd.DataFrame()
        self.retorno_portfolio['Retornos'] = retorno_da_carteira
        var_95 = round((np.nanpercentile(self.retorno_portfolio,5))*100, 2) 
        st.write('O VAR pelo metodo historico da carteira para o período selecionado e :')
        st.warning(var_95)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.retorno_portfolio['Retornos']))
        fig.update_layout(title ='Distribuição dos Retornos Carteira')

        retorno_acumulado = round((((1+self.retorno_portfolio['Retornos']).cumprod())-1)*100,2)
        grafico_de_linha = go.Figure()
        grafico_de_linha.add_trace(go.Scatter(x=self.retorno_portfolio.index,y=retorno_acumulado,mode='lines',name= 'Retorno Acumulado'))
        grafico_de_linha.update_traces(line=dict(color='#ADFF2F'))
        grafico_de_linha.update_layout(title='Retorno Acumulado ao Longo do Tempo')
       
     

        st.plotly_chart(grafico_de_linha)        
       
        return st.plotly_chart(fig)
    

    def var_historico_simulacao(self,periodo_inicial,period_final,acoes,pesos):


        lista_acoes = acoes
        pesos_carteira = pesos


        carteira = yf.download(lista_acoes,start=periodo_inicial,end=period_final)['Adj Close']
        retornos = carteira.pct_change()

        retorno_da_carteira = (retornos*pesos_carteira).sum(axis=1) 

        self.retorno_portfolio = pd.DataFrame()
        self.retorno_portfolio['Retornos'] = retorno_da_carteira

        var_95 = round((np.nanpercentile(self.retorno_portfolio,5))*100, 2) 
        st.write('O VAR pelo metodo historico da carteira para o período selecionado e :')
        st.warning(var_95)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.retorno_portfolio['Retornos']))
        fig.update_layout(title ='Distribuição dos Retornos Carteira')
    
        retorno_acumulado = round(((self.retorno_portfolio['Retornos']+1).cumprod()-1)*100,2)

        grafico_de_linha = go.Figure()
        grafico_de_linha.add_trace(go.Scatter(x=self.retorno_portfolio.index,y=retorno_acumulado,mode='lines',name= 'Retorno Acumulado'))
        grafico_de_linha.update_traces(line=dict(color='#1E90FF'))
        grafico_de_linha.update_layout(title='Retorno Acumulado ao Longo do Tempo')
       
     

        st.plotly_chart(grafico_de_linha)        
       
        return st.plotly_chart(fig)




    '''Monte carlo'''

    def monte_carlo(self,dias):
            
        lista_acoes = lista_acoes_yf


        data_final = dt.datetime.now()
        data_inicial = data_final - dt.timedelta(days=300)

        precos = yf.download(lista_acoes, data_inicial, data_final)['Adj Close']

        #calculando retornos pegando matriz de covariância 

        retornos = precos.pct_change().dropna()
        media_retornos = retornos.mean()
        matriz_covariancia = retornos.cov()
        pesos_carteira = pesos_carteira_eq
        numero_acoes = len(lista_acoes)


        numero_simulacoes = 10000
        dias_projetados = dias 
        capital_inicial = 100000


        retorno_medio = retornos.mean(axis = 0).to_numpy() 
        matriz_retorno_medio = retorno_medio * np.ones(shape = (dias_projetados, numero_acoes))

        L = LA.cholesky(matriz_covariancia)
        


        self.retornos_carteira = np.zeros([dias_projetados, numero_simulacoes]) #cada coluna é uma simulação
        montante_final = np.zeros(numero_simulacoes)

        for s in range(numero_simulacoes):

            Rpdf = np.random.normal(size=(dias_projetados, numero_acoes)) 
            
            retornos_sintéticos = matriz_retorno_medio + np.inner(Rpdf, L) #unica coisa random é o Rpdf
            
            self.retornos_carteira[:, s] = np.cumprod(np.inner(pesos_carteira, 
                                                        retornos_sintéticos) + 1) * capital_inicial
            montante_final[s] = self.retornos_carteira[-1, s]
            
        montante_99 = str(round(np.percentile(montante_final, 1),2))
        montante_95 = str(round(np.percentile(montante_final, 5),2))
        montante_mediano = str(round(np.percentile(montante_final, 50),2))
        cenarios_com_lucro = str((len(montante_final[montante_final > 100000])/
                                        len(montante_final)) * 100) + "%"


        st.text(f"Ao investir R$ 100.000,00 na carteira, podemos esperar esses resultados para os próximo período,")
        st.text("utilizando o método de Monte Carlo com 10 mil simulações:")

        st.caption(f':green[Com 50% de probabilidade, o montante será maior que R$ {montante_mediano}.]') 

        st.caption(f':red[Com 95% de probabilidade, o montante será maior que R$ {montante_95}.]')

        st.caption(f":red[Com 99% de probabilidade, o montante será maior que R$ {montante_99}.]")

        st.caption(f':green[Em {cenarios_com_lucro} dos cenários, foi possível obter lucro no período.]')

        plt.plot(self.retornos_carteira, linewidth=1)
        plt.ylabel('Dinheiro')
        plt.xlabel('Dias')

        st.pyplot()




        config = dict(histtype = "stepfilled", alpha = 0.8, density = False, bins = 150)
        fig, ax = plt.subplots()
        ax.hist(montante_final, **config)
        ax.xaxis.set_major_formatter('R${x:.0f}')
        distribuicao_monte = plt.title('Distribuição montantes finais com simulação MC')
        distribuicao_monte = plt.xlabel('Montante final (R$)')
        distribuicao_monte = plt.ylabel("Frequência")

        return st.pyplot(fig)
    

    def nova_carteira_monte_carlo(self,lista_de_acoes,dias,pesos):
        

        data_final = dt.datetime.now()
        data_inicial = data_final - dt.timedelta(days=300)

        precos = yf.download(lista_de_acoes, data_inicial, data_final)['Adj Close']

        #calculando retornos pegando matriz de covariância 

        retornos = precos.pct_change().dropna()
        media_retornos = retornos.mean()
        matriz_covariancia = retornos.cov()
        pesos_carteira = [pesos]
        numero_acoes = len(lista_de_acoes)


        numero_simulacoes = 10000
        dias_projetados = dias 
        capital_inicial = 100000


        retorno_medio = retornos.mean(axis = 0).to_numpy() 
        matriz_retorno_medio = retorno_medio * np.ones(shape = (dias_projetados, numero_acoes))

        L = LA.cholesky(matriz_covariancia)
        


        self.retornos_carteira = np.zeros([dias_projetados, numero_simulacoes]) #cada coluna é uma simulação
        montante_final = np.zeros(numero_simulacoes)

        for s in range(numero_simulacoes):

            Rpdf = np.random.normal(size=(dias_projetados, numero_acoes)) 
            
            retornos_sintéticos = matriz_retorno_medio + np.inner(Rpdf, L) #unica coisa random é o Rpdf
            
            self.retornos_carteira[:, s] = np.cumprod(np.inner(pesos_carteira, 
                                                        retornos_sintéticos) + 1) * capital_inicial
            montante_final[s] = self.retornos_carteira[-1, s]
            
        montante_99 = str(round(np.percentile(montante_final, 1),2))
        montante_95 = str(round(np.percentile(montante_final, 5),2))
        montante_mediano = str(round(np.percentile(montante_final, 50),2))
        cenarios_com_lucro = str((len(montante_final[montante_final > 100000])/
                                        len(montante_final)) * 100) + "%"


        st.text(f"Ao investir R$ 100.000,00 na carteira, podemos esperar esses resultados para os próximo período,")
        st.text("utilizando o método de Monte Carlo com 10 mil simulações:")

        st.caption(f':green[Com 50% de probabilidade, o montante será maior que R$ {montante_mediano}.]') 

        st.caption(f':red[Com 95% de probabilidade, o montante será maior que R$ {montante_95}.]')

        st.caption(f":red[Com 99% de probabilidade, o montante será maior que R$ {montante_99}.]")

        st.caption(f':green[Em {cenarios_com_lucro} dos cenários, foi possível obter lucro no período.]')

        plt.plot(self.retornos_carteira, linewidth=1)
        plt.ylabel('Dinheiro')
        plt.xlabel('Dias')

        st.pyplot()




        config = dict(histtype = "stepfilled", alpha = 0.8, density = False, bins = 150)
        fig, ax = plt.subplots()
        ax.hist(montante_final, **config)
        ax.xaxis.set_major_formatter('R${x:.0f}')
        distribuicao_monte = plt.title('Distribuição montantes finais com simulação MC')
        distribuicao_monte = plt.xlabel('Montante final (R$)')
        distribuicao_monte = plt.ylabel("Frequência")

        return st.pyplot(fig)


    def markovitz_equities(self):

        data_final = dt.date.today()#.strftime('%Y/%m/%d')
        print(data_final)

        lista_acoes =  lista_acoes_yf
        precos = yf.download(lista_acoes,start='2015-01-01', end=data_final)['Adj Close']
        retornos = precos.pct_change().apply(lambda x: np.log(1+x)).dropna()
        media_retornos = retornos.mean()
        matriz_cov = retornos.cov()
        numero_carteiras = 50000
        vetor_retornos_esperados = np.zeros(numero_carteiras +1)
        vetor_volatilidades_esperadas = np.zeros(numero_carteiras+1)
        vetor_sharpe = np.zeros(numero_carteiras+ 1)
        tabela_pesos = np.zeros((numero_carteiras+1, len(lista_acoes)))

        for k in range(numero_carteiras+1):

            if k < numero_carteiras:
                pesos = np.random.random(len(lista_acoes))
                pesos = pesos/np.sum(pesos)
                tabela_pesos[k, :] = pesos
                
                vetor_retornos_esperados[k] = np.sum(media_retornos * pesos * 252)
                vetor_volatilidades_esperadas[k] = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov*252, pesos)))
                
                vetor_sharpe[k] = vetor_retornos_esperados[k]/vetor_volatilidades_esperadas[k]
            else:
                pesos = pesos_carteira_eq / np.sum(pesos_carteira_eq)  # Use predefined weights for the last iteration
                tabela_pesos[k, :] = pesos
                vetor_retornos_esperados[k] = np.sum(media_retornos * pesos * 252)
                vetor_volatilidades_esperadas[k] = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov * 252, pesos)))
                vetor_sharpe[k] = vetor_retornos_esperados[k] / vetor_volatilidades_esperadas[k]



        posicao_sharpe_maximo = vetor_sharpe.argmax()
        pesos_otimos = tabela_pesos[posicao_sharpe_maximo, :]
        pesos_otimos = [str((peso * 100).round(2)) + "%" for peso in pesos_otimos]

        tabela_pesos_otimos = pd.DataFrame(columns=['Ação','Peso Otimo'])

        for i, acao in enumerate(lista_acoes):

            tabela_pesos_otimos.loc[i] = [acao, pesos_otimos[i]]
        print(tabela_pesos_otimos)    


        vetor_retornos_esperados_arit = np.exp(vetor_retornos_esperados) - 1

        eixo_y_fronteira_eficiente = np.linspace(vetor_retornos_esperados_arit.min(), 
                                                vetor_retornos_esperados_arit.max(), 50)

        def pegando_retorno(peso_teste):
            peso_teste = np.array(peso_teste)
            retorno = np.sum(media_retornos * peso_teste) * 252
            retorno = np.exp(retorno) - 1

            return retorno

        def checando_soma_pesos(peso_teste):

            return np.sum(peso_teste)-1

        def pegando_vol(peso_teste):
            peso_teste = np.array(peso_teste)
            vol = np.sqrt(np.dot(peso_teste.T, np.dot(matriz_cov*252, peso_teste)))
            
            return vol

        peso_inicial = [1/len(lista_acoes)] * len(lista_acoes) 
        limites = tuple([(0, 1) for ativo in lista_acoes])


        eixo_x_fronteira_eficiente = []
        pesos_eficientes = []

        for retorno_possivel in eixo_y_fronteira_eficiente:

            restricoes = ({'type': 'eq', 'fun': checando_soma_pesos},
                        {'type': 'eq', 'fun': lambda w: pegando_retorno(w) - retorno_possivel})

            result = minimize(pegando_vol, peso_inicial, method='SLSQP', bounds=limites, constraints=restricoes)

            eixo_x_fronteira_eficiente.append(result['fun'])     
        
            pesos_eficientes.append(result['x'])

        pesos_eficientes = np.array(pesos_eficientes)

        fig, ax = plt.subplots()
        ax.plot(eixo_x_fronteira_eficiente, eixo_y_fronteira_eficiente, label='Efficient Frontier')

 
        ax.scatter(vetor_volatilidades_esperadas[:-1], vetor_retornos_esperados_arit[:-1], c=vetor_sharpe[:-1], cmap='viridis', label='Random Portfolios')

        ax.scatter(vetor_volatilidades_esperadas[-1], vetor_retornos_esperados_arit[-1], c='red', label='Portfolio Equities')

        scatter_max_sharpe = ax.scatter(vetor_volatilidades_esperadas[posicao_sharpe_maximo], vetor_retornos_esperados_arit[posicao_sharpe_maximo], c='#C71585', label='Portfolio with Best Sharpe Ratio')

        plt.xlabel("Volatilidade esperada")
        plt.ylabel("Retorno esperado")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.legend()
        fig = plt.show()
        st.pyplot(fig)
        st.write('Pesos da carteira com melhor Sharpe Ratio')
        return st.table(tabela_pesos_otimos)
    

    def markovitz_simulando_carteira(self,lista_de_acoes,pesos_sim):

        data_final = dt.date.today()#.strftime('%Y/%m/%d')
        print(data_final)

        lista_acoes =  lista_de_acoes
        precos = yf.download(lista_acoes,start='2015-01-01', end=data_final)['Adj Close']
        retornos = precos.pct_change().apply(lambda x: np.log(1+x)).dropna()
        media_retornos = retornos.mean()
        matriz_cov = retornos.cov()
        numero_carteiras = 50000
        vetor_retornos_esperados = np.zeros(numero_carteiras +1)
        vetor_volatilidades_esperadas = np.zeros(numero_carteiras+1)
        vetor_sharpe = np.zeros(numero_carteiras+ 1)
        tabela_pesos = np.zeros((numero_carteiras+1, len(lista_acoes)))

        for k in range(numero_carteiras+1):

            if k < numero_carteiras:
                pesos = np.random.random(len(lista_acoes))
                pesos = pesos/np.sum(pesos)
                tabela_pesos[k, :] = pesos
                
                vetor_retornos_esperados[k] = np.sum(media_retornos * pesos * 252)
                vetor_volatilidades_esperadas[k] = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov*252, pesos)))
                
                vetor_sharpe[k] = vetor_retornos_esperados[k]/vetor_volatilidades_esperadas[k]
            else:
                pesos = pesos_sim / np.sum(pesos_sim) 
                tabela_pesos[k, :] = pesos
                vetor_retornos_esperados[k] = np.sum(media_retornos * pesos * 252)
                vetor_volatilidades_esperadas[k] = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov * 252, pesos)))
                vetor_sharpe[k] = vetor_retornos_esperados[k] / vetor_volatilidades_esperadas[k]



        posicao_sharpe_maximo = vetor_sharpe.argmax()
        pesos_otimos = tabela_pesos[posicao_sharpe_maximo, :]
        pesos_otimos = [str((peso * 100).round(2)) + "%" for peso in pesos_otimos]

        tabela_pesos_otimos = pd.DataFrame(columns=['Ação','Peso Otimo'])

        for i, acao in enumerate(lista_acoes):

            tabela_pesos_otimos.loc[i] = [acao, pesos_otimos[i]]
        print(tabela_pesos_otimos)    


        vetor_retornos_esperados_arit = np.exp(vetor_retornos_esperados) - 1

        eixo_y_fronteira_eficiente = np.linspace(vetor_retornos_esperados_arit.min(), 
                                                vetor_retornos_esperados_arit.max(), 50)

        def pegando_retorno(peso_teste):
            peso_teste = np.array(peso_teste)
            retorno = np.sum(media_retornos * peso_teste) * 252
            retorno = np.exp(retorno) - 1

            return retorno

        def checando_soma_pesos(peso_teste):

            return np.sum(peso_teste)-1

        def pegando_vol(peso_teste):
            peso_teste = np.array(peso_teste)
            vol = np.sqrt(np.dot(peso_teste.T, np.dot(matriz_cov*252, peso_teste)))
            
            return vol

        peso_inicial = [1/len(lista_acoes)] * len(lista_acoes) 
        limites = tuple([(0, 1) for ativo in lista_acoes])


        eixo_x_fronteira_eficiente = []
        pesos_eficientes = []

        for retorno_possivel in eixo_y_fronteira_eficiente:

            restricoes = ({'type': 'eq', 'fun': checando_soma_pesos},
                        {'type': 'eq', 'fun': lambda w: pegando_retorno(w) - retorno_possivel})

            result = minimize(pegando_vol, peso_inicial, method='SLSQP', bounds=limites, constraints=restricoes)

            eixo_x_fronteira_eficiente.append(result['fun'])     
        
            pesos_eficientes.append(result['x'])

        pesos_eficientes = np.array(pesos_eficientes)

        fig, ax = plt.subplots()
        ax.plot(eixo_x_fronteira_eficiente, eixo_y_fronteira_eficiente, label='Efficient Frontier')

 
        ax.scatter(vetor_volatilidades_esperadas[:-1], vetor_retornos_esperados_arit[:-1], c=vetor_sharpe[:-1], cmap='viridis', label='Random Portfolios')

        ax.scatter(vetor_volatilidades_esperadas[-1], vetor_retornos_esperados_arit[-1], c='red', label='Portfolio Equities')

        scatter_max_sharpe = ax.scatter(vetor_volatilidades_esperadas[posicao_sharpe_maximo], vetor_retornos_esperados_arit[posicao_sharpe_maximo], c='#C71585', label='Portfolio with Best Sharpe Ratio')

        plt.xlabel("Volatilidade esperada")
        plt.ylabel("Retorno esperado")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.legend()
        fig = plt.show()
        st.pyplot(fig)
        st.write('Pesos da carteira com melhor Sharpe Ratio')
        return st.table(tabela_pesos_otimos)