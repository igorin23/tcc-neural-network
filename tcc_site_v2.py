import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import torch

# --- Configuração da Página ---
st.set_page_config(page_title="TCC - Comparação de Modelos", layout="wide")

st.title("📊 Estudo Comparativo: SSD vs YOLO vs Rede Própria")
st.markdown("Projeto de TCC sobre desempenho em reconhecimento de imagens.")

# --- Carregamento dos Modelos (Cache para não recarregar a cada clique) ---
@st.cache_resource
def carregar_modelos():
    # Aqui você carrega seus pesos .pth ou .pt reais
    # Exemplo: modelo_ssd = torch.load('ssd_pesos.pth')
    # Retornando strings apenas para o exemplo funcionar
    return {"SSD": "modelo_ssd_carregado", "YOLO": "modelo_yolo", "Própria": "minha_rede"}

modelos = carregar_modelos()

# --- Navegação ---
abas = st.tabs(["📈 Gráficos de Desempenho", "🖼️ Galeria de Testes", "🚀 Teste ao Vivo (Demo)"])

# --- ABA 1: GRÁFICOS 3D ---
with abas[0]:
    st.header("Análise Multidimensional (3D)")
    st.markdown("Visualização da precisão (Eixo Z) baseada no tamanho do Dataset (X) e Épocas (Y).")

    # 1. Criando dados fictícios no formato correto para 3D
    # Para o seu TCC, você provavelmente carregará isso de um CSV: pd.read_csv('seus_resultados.csv')
    data_3d = {
        'Modelo': [],
        'Qtd_Imagens': [],
        'Epocas': [],
        'Precisao': []
    }

    # Gerando dados de exemplo para os 3 modelos
    # Imaginando cenários: 500, 1000, 2000 e 5000 imagens
    datasets = [500, 1000, 2000, 5000]
    # Épocas checadas: 10, 20, 30, 40, 50
    epocas = [10, 20, 30, 40, 50]

    for modelo_nome in ['SSD', 'YOLO', 'Rede Própria']:
        for d in datasets:
            for e in epocas:
                # Simulando uma precisão baseada no modelo + dados + época (só para o gráfico fazer sentido)
                base = 0.5 if modelo_nome == 'SSD' else (0.6 if modelo_nome == 'YOLO' else 0.4)
                bonus_dados = (d / 10000) # Mais dados ajudam
                bonus_epoca = (e / 200)   # Mais épocas ajudam
                acc = min(0.99, base + bonus_dados + bonus_epoca) # Limita a 99%
                
                data_3d['Modelo'].append(modelo_nome)
                data_3d['Qtd_Imagens'].append(d)
                data_3d['Epocas'].append(e)
                data_3d['Precisao'].append(round(acc, 4))

    # Descomentar quando for subir os dados
    # df_3d = pd.DataFrame(data_3d)
    df_3d = pd.read_csv("results/dados_consolidado.csv")


    # 2. Criando o Gráfico 3D
    fig_3d = px.scatter_3d(

        df_3d, 
        x='Qtd_Imagens', 
        y='Epocas', 
        z='Precisao',
        color='Modelo',          # Cada modelo terá uma cor diferente
        symbol='Modelo',         # Formatos diferentes (bolinha, quadrado, diamante) para acessibilidade
        size='Precisao',         # (Opcional) Bolinhas com maior precisão ficam maiores
        size_max=18,             # Tamanho máximo das bolinhas
        opacity=0.7,             # Transparência para ver pontos sobrepostos
        title="Convergência: Dataset vs Épocas vs Precisão",
        hover_data=['Precisao']  # O que aparece ao passar o mouse
    )

    # Ajuste visual do layout 3D
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Nº Imagens (Dataset)',
            yaxis_title='Nº Épocas',
            zaxis_title='Precisão (mAP)',
        ),
        margin=dict(l=0, r=0, b=0, t=40), # Margens menores para aproveitar espaço
        height=600 # Altura do gráfico
    )

    # Renderizando no Streamlit
    # Nota: Se aparecer o aviso de novo, use apenas st.plotly_chart(fig_3d) sem argumentos extras
    st.plotly_chart(fig_3d, use_container_width=True) 

    st.info("💡 Dica: Você pode clicar e arrastar o gráfico acima para girá-lo e analisar os pontos de concentração de melhor acurácia.")


# --- ABA 2: GALERIA ---
with abas[1]:
    st.header("Resultados Visuais (Estático)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("SSD")
        # st.image("caminho/para/resultado_ssd.jpg") # Descomente e use suas fotos
        st.write("Resultado em Dataset Pequeno")
        
    with col2:
        st.subheader("YOLO")
        # st.video("caminho/para/video_teste.mp4") # Suporta vídeo também
        st.write("Teste em vídeo")
        
    with col3:
        st.subheader("Rede Própria")
        # st.image("caminho/para/resultado_propria.jpg")
        st.write("Destaque para detecção de bordas")

# --- ABA 3: LIVE DEMO (O PLUS) ---
with abas[2]:
    st.header("Teste Real-Time")
    st.write("Faça upload de uma imagem para testar nos modelos agora.")
    
    arquivo = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])
    
    if arquivo is not None:
        imagem = Image.open(arquivo)
        st.image(imagem, caption="Imagem Original", width=400)
        
        if st.button("Processar Imagem"):
            st.write("Processando...")
            
            # Aqui entra sua função de inferência real
            # predicao = fazer_inferencia(modelos['YOLO'], imagem)
            
            # Simulando resultado visual
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.success("SSD: Carro (98%)") # Exemplo
            with col_b:
                st.success("YOLO: Carro (99%)")
            with col_c:
                st.warning("Própria: Carro (75%)")