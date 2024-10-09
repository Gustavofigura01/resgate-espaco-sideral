# codigo baseado em feed for feedback propagation
import numpy as np
import json
import tensorflow as tf

# funcao de ativacao sigmoid
def sigmoid(x, derivada=False):
    if derivada:
        sigm_x = sigmoid(x)
        return sigm_x * (1.0 - sigm_x)
    else:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class RedeNeural:
    # construtor da classe
    def __init__(self, entrada_base, saida_base, tamanho_entrada, tamanho_oculto, tamanho_saida, peso_salvo1=None, peso_salvo2=None):

        self.TAXA_APRENDIZAGEM = 0.0005

        self.entrada = entrada_base
        self.y = saida_base

        self.tamanho_entrada = tamanho_entrada
        self.tamanho_oculto = tamanho_oculto
        self.tamanho_saida = tamanho_saida

        if peso_salvo1 is None:
            self.pesos_entrada = np.random.rand(tamanho_entrada, tamanho_oculto) - 0.5
        else:
            self.pesos_entrada = peso_salvo1

        if peso_salvo2 is None:
            self.pesos_saida = np.random.rand(tamanho_oculto, tamanho_saida) - 0.5
        else:
            self.pesos_saida = peso_salvo2

    # funcao para treinar a rede neural
    def feedforward(self, entrada=None):
        array_entrada = []

        if entrada is None:
            array_entrada = np.array(self.entrada, ndmin=2)
        else:
            array_entrada = np.array(entrada, ndmin=2)

        self.camada_oculta = sigmoid(tf.matmul(tf.cast(array_entrada, dtype=tf.int32), tf.cast(self.pesos_entrada, dtype=tf.int32)))
        self.saida = sigmoid(tf.matmul(tf.cast(self.camada_oculta, dtype=tf.int32), tf.cast(self.pesos_saida, dtype=tf.int32)))
    
    def retropropagacao(self):
        # aplicação da regra da cadeia para encontrar a derivada da função de perda em relação aos pesos da camada de saída e da camada oculta
        d_pesos_saida = tf.matmul(tf.transpose(self.camada_oculta), ((self.y - self.saida) * sigmoid(self.saida, True)))
        d_pesos_entrada = tf.matmul(self.entrada.T, (tf.matmul((self.y - self.saida) * sigmoid(self.saida, True), tf.transpose(self.pesos_saida)) * sigmoid(self.camada_oculta, True)))

        # atualização dos pesos com a derivada (inclinação) da função de perda
        self.pesos_entrada += d_pesos_entrada * self.TAXA_APRENDIZAGEM
        self.pesos_saida += d_pesos_saida * self.TAXA_APRENDIZAGEM

    def obter_saida(self, entrada):
        self.feedforward(entrada)
        return self.saida

# main para testar a rede neural
if __name__ == "__main__":
    dados = []

    caminho_arquivo = "C:\\Users\\gusta\\Documents\\sistemas inteligentes\\base_dados.json"
    with open(caminho_arquivo) as arquivo:
        dados = json.load(arquivo)


    entrada = []
    saida = []

    for d in dados:
        entrada.append(d[0])
        saida.append(d[1])

    entrada = np.array(entrada)
    saida = np.array(saida)

    rede_neural = RedeNeural(entrada, saida, 2, 20, 1)

    epocas = 10000
    diferenca = 5
    while(diferenca > 0.06 and epocas > 0):
        epocas -= 1
        diferenca_anterior = diferenca
        rede_neural.feedforward()
        diferenca = np.sum((rede_neural.saida - rede_neural.y)**2) / len(rede_neural.entrada)
        print(diferenca)
        rede_neural.retropropagacao()

    peso_salvo1 = rede_neural.pesos_entrada.numpy().tolist()
    peso_salvo2 = rede_neural.pesos_saida.numpy().tolist()

    with open('peso_salvo1', 'w') as out:
        json.dump(peso_salvo1, out)

    with open('peso_salvo2', 'w') as out:
        json.dump(peso_salvo2, out)

    rede_neural.feedforward(np.array([20, 100]))
    print(rede_neural.saida)
    rede_neural.feedforward(np.array([300, 100]))
    print(rede_neural.saida)
    rede_neural.feedforward(np.array([200, 150]))
    print(rede_neural.saida)
