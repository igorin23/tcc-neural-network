import pandas as pd

yolo_csv = pd.read_csv('results/info_yolo.csv')
rede_neural_csv = pd.read_csv('results/resultados_parciais_tcc.csv')

yolo_pd = pd.DataFrame()

yolo_pd['Qtd_Imagens'] = (yolo_csv['dataset_fracao']*16059).astype('int64')
yolo_pd['Epocas'] = yolo_csv['epoca']
yolo_pd['Precisao'] = yolo_csv['precision']
yolo_pd['Modelo'] = 'YOLO'
yolo_pd = yolo_pd[['Modelo', 'Qtd_Imagens', 'Epocas', 'Precisao']]


rd_neural_pd = pd.DataFrame()

rd_neural_pd['Qtd_Imagens'] = rede_neural_csv['qtd_imagens']
rd_neural_pd['Epocas'] = rede_neural_csv['epoca']
rd_neural_pd['Precisao'] = round(rede_neural_csv['mAP'], 5)
rd_neural_pd['Modelo'] = rede_neural_csv['modelo']
rd_neural_pd = rd_neural_pd[['Modelo', 'Qtd_Imagens', 'Epocas', 'Precisao']]

# print(rd_neural_pd)

df_final = pd.concat([yolo_pd, rd_neural_pd], ignore_index=True)




df_final.to_csv('results/dados_consolidado.csv', index=False)
print('Arquivo Consolidade Gerado com sucesso!')
