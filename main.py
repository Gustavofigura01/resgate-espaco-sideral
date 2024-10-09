# -- Resgate Espacial: Missão de Sobrevivência
# -- Autor: Gustavo Figura

import collections
import numpy as np
import pygame
import json
import random
import rede_neural
import os
import sys
from pygame.locals import *

# definir posição da janela
x = 1
y = 40
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)

with open('peso_salvo1') as f:
    peso_salvo1 = json.load(f)
with open('peso_salvo2') as f:
    peso_salvo2 = json.load(f)

class JogoResgate(object):

    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(10, 100)
        
        # definir constantes
        self.COR_BRANCA = (255, 255, 255)
        self.LARGURA_JOGO = 600
        self.ALTURA_JOGO = 400
        self.ASTRONAUTA = 10  
        self.LARGURA_NAVE = 35
        self.ALTURA_NAVE = 5
        self.SOLO_JOGO = 350
        self.TETO_JOGO = 10
        self.VELOCIDADE_ASTRONAUTA = 10
        self.VELOCIDADE_NAVE = 20
        self.TAMANHO_FONTE = 30
        self.TENTATIVAS_MAX_JOGO = 3
        self.EVENTO_CUSTOMIZADO = pygame.USEREVENT + 1
        self.fonte = pygame.font.SysFont("Comic Sans MS", self.TAMANHO_FONTE)
        self.imagem_fundo = pygame.image.load("fundo.jpg")
        self.imagem_NAVE = pygame.image.load("NAVE.png")
        self.imagem_ASTRONAUTA = pygame.image.load("astronauta.png")

    # inicializar posições
    def reset(self):
        self.frames = collections.deque(maxlen=4)
        self.fim_jogo = False
        # inicializar posições
        self.NAVE_x = self.LARGURA_JOGO // 2
        self.pontuacao_jogo = 0
        self.recompensa = 0
        self.ASTRONAUTA_x = random.randint(0, self.LARGURA_JOGO)
        self.ASTRONAUTA_y = self.TETO_JOGO
        self.num_tentativas = 0
        # configurar tela, relógio, etc
        self.tela = pygame.display.set_mode(
                (self.LARGURA_JOGO, self.ALTURA_JOGO))
        self.relogio = pygame.time.Clock()
        self.imagem_fundo = pygame.transform.scale(self.imagem_fundo, (self.LARGURA_JOGO, self.ALTURA_JOGO))
    
    # atualizar tela
    def step(self, acao): 
        for evento in pygame.event.get():
            if evento.type == QUIT:
                pygame.quit()
                return

        pygame.event.pump()
        
        # Atualizar posição da NAVE
        if acao == 0:   # mover a NAVE para a esquerda
            self.NAVE_x -= self.VELOCIDADE_NAVE
            if self.NAVE_x < 0:
                self.NAVE_x = 0
        elif acao == 2: # mover a NAVE para a direita
            self.NAVE_x += self.VELOCIDADE_NAVE
            if self.NAVE_x > self.LARGURA_JOGO - self.imagem_NAVE.get_width():
                self.NAVE_x = self.LARGURA_JOGO - self.imagem_NAVE.get_width()
        else:             # não mover a NAVE
            pass

        self.tela.blit(self.imagem_fundo, (0, 0))

        texto_pontuacao = self.fonte.render("Pontuação: {:d}".format(self.pontuacao_jogo), True, self.COR_BRANCA)
        texto_vidas = self.fonte.render("Vidas: {:d}".format(self.TENTATIVAS_MAX_JOGO - self.num_tentativas), True, self.COR_BRANCA)
        self.tela.blit(texto_pontuacao, (10, 10))
        self.tela.blit(texto_vidas, (self.LARGURA_JOGO - texto_vidas.get_width() - 10, 10))
                
        # atualizar posição da ASTRONAUTA
        self.ASTRONAUTA_y += self.VELOCIDADE_ASTRONAUTA
        ASTRONAUTA = self.imagem_ASTRONAUTA.get_rect()
        ASTRONAUTA.centerx = self.ASTRONAUTA_x
        ASTRONAUTA.centery = self.ASTRONAUTA_y
        self.tela.blit(self.imagem_ASTRONAUTA, ASTRONAUTA)

        # atualizar posição da NAVE
        NAVE = self.imagem_NAVE.get_rect()
        NAVE.centerx = self.NAVE_x
        self.NAVE_y = self.SOLO_JOGO - self.imagem_NAVE.get_height()
        NAVE.bottom = self.SOLO_JOGO
        self.tela.blit(self.imagem_NAVE, NAVE)

       # Verificar colisão e atualizar recompensa
        if self.ASTRONAUTA_y >= self.SOLO_JOGO - self.ASTRONAUTA:
            # Calcular a posição do centro da ASTRONAUTA
            ASTRONAUTA_centro_x = self.ASTRONAUTA_x + self.ASTRONAUTA
            ASTRONAUTA_centro_y = self.ASTRONAUTA_y + self.ASTRONAUTA

            # Calcular a posição do centro da NAVE
            NAVE_centro_x = self.NAVE_x + self.LARGURA_NAVE / 2

            # Verificar se houve colisão entre os centros da ASTRONAUTA e da NAVE
            distancia_centros = abs(ASTRONAUTA_centro_x - NAVE_centro_x)
            if distancia_centros <= self.ASTRONAUTA + self.LARGURA_NAVE / 2:
                self.recompensa += 1
                self.pontuacao_jogo += 1
                if self.pontuacao_jogo % 50 == 0: # aumentar velocidade da ASTRONAUTA a cada 50 pontos
                    self.VELOCIDADE_ASTRONAUTA += 0.5
            else:
                self.num_tentativas += 1

            self.ASTRONAUTA_x = random.randint(100, self.LARGURA_JOGO - 100)
            self.ASTRONAUTA_y = self.TETO_JOGO
        
        pygame.display.flip()
            
        quadro = pygame.surfarray.array2d(self.tela)
        
        if self.num_tentativas >= self.TENTATIVAS_MAX_JOGO:
            self.fim_jogo = True
            
        self.relogio.tick(60)
        return quadro, self.recompensa, self.fim_jogo
        
    def obter_frames(self):
        return np.array(list(self.frames))

    def obter_dados_ambiente(self):
        return [self.ASTRONAUTA_x, self.NAVE_x + self.LARGURA_NAVE/2.0]
    
if __name__ == "__main__":
    jogo = JogoResgate()
    rede_neural = rede_neural.RedeNeural([], [], 2, 20, 1, peso_salvo1=peso_salvo1, peso_salvo2=peso_salvo2)
    jogo.reset()
    input_t = jogo.obter_frames()
    fim_jogo = False

    if len(sys.argv) > 1 and sys.argv[1] == 'treinar':
        # Opção: Treinar a IA
        NUM_EPOCAS = 100  # Número de épocas (repetições) desejadas

        for epoca in range(NUM_EPOCAS):
            print(f"Época {epoca + 1}")

            jogo.reset()
            input_t = jogo.obter_frames()
            fim_jogo = False

            while not fim_jogo:
               
                saida = rede_neural.obter_saida(jogo.obter_dados_ambiente())

                if saida > 0.5:
                    acao = 2
                else:
                    acao = 0

                input_tp1, recompensa, fim_jogo = jogo.step(acao)

            # Aqui é onde o jogo terminou
            print("O jogo terminou.")
    else:
        # Opção: Jogar uma partida
        while not fim_jogo:
           
            saida = rede_neural.obter_saida(jogo.obter_dados_ambiente())

            if saida > 0.5:
                acao = 2
            else:
                acao = 0

            input_tp1, recompensa, fim_jogo = jogo.step(acao)

    # Aqui é onde o jogo terminou
    print("O jogo terminou.")

print(acao, recompensa, fim_jogo)
