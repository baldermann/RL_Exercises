# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:31:09 2024

@author: aunko
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pyplot as plt


l1 = 64
l2 = 150
l3 = 100
l4 = 4

model = torch.nn.Sequential(
    
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
    
    )

loss_fn = torch.nn.MSELoss()
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

gamma = 0.9
epsilon = 1.0

action_set = {  # Mapping der Bedeutung des Ausgabevektors des neuronalen Netzes als Aktion in der Gridworld
    0: 'u', # Move Up
    1: 'd', # Move Down
    2: 'l', # Move Left
    3: 'r'} # Move Right

"""
    Q-Learning: Haupt-Trainingsschleife
"""

epochs = 1000
losses = []

for i in range(epochs):  # Haupttrainingsschleife

    game = Gridworld(size = 4, mode = 'random')
    
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0   # Erstellen eines Spiels
                                                                                    # Wir transformieren die 2d-Spielwelt in einen Flatten-Input
                                                                                    # Mit Random.rand() fügen wir etwas rauschen hinzu!
                                                                                    
    # vergleiche state_ mit game.board.render_np().reshape(1, 64) in der Konsolse!

    state1 = torch.from_numpy(state_).float()  # Konvertieren in eine PyTorch-Variable
    
    status = 1 # verwemdet doe Statusvariable des Spiels (ob das Spiel beendet ist oder nicht)
    
    while(status):
        
        qval = model(state1)    # Treibt das Q-Network voran, um seine vorhergesagten Q-Werte für alle Aktionen erhalten
        qval_ = qval.data.numpy()
 
        if random.random() < epsilon:
            
            action_ = np.random.randint(0, 4)   # Wähle eine Aktion!
            
        else:
            
            action_ = np.argmax(qval_)
            
        action = action_set[action_]
        
        game.makeMove(action)   # Make Move!
        
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10 # Flatten der Gridworld
        state2 = torch.from_numpy(state2_).float()
        
        reward = game.reward()
        
        with torch.no_grad():    # .no_grad() sorgt für die Forward-Propagation (Berechnung nächster Zustands-Q-Werte)
            
            newQ = model(state2.reshape(1, 64))
            
        maxQ = torch.max(newQ)    # gibt den maximalen Wert eines tensors aus
        
        if reward == -1:
            
            y = reward + (gamma * maxQ)
            
        else:
            
            y = reward
        
        y = torch.Tensor([y]).detach()
        X = qval.squeeze()[action_]
        
        loss = loss_fn(X, y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        
        state1 = state2
        
        if reward != -1:  # Wenn die Belohnung -1 ist, wurde das Spiel weder gewonnen noch verloren, läuft also noch!
            
            status = 0
            
        if epsilon > 0.1:
            
            epsilon = epsilon - (1 / epochs)
        

plt.plot(range(0, len(losses)), losses)
plt.title("Loss Function")
plt.xlabel('Epochs')
plt.ylabel('PyTorch Loss ')
plt.show()

        
        
def test_model(model, mode = 'static', display = True):
    
    i = 0
    test_game = Gridworld(mode = mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    
    if display:
        
        print("initial State:")
        print(test_game.display())
        
    status = 1
    
    while(status):
        
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)      # <- Auswahl des höchsten Q-Wertes
        action = action_set[action_]
        
        if display:
            
            print('Move #: %s; Taking action: %s' % (i, action))
            
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        
        if display:
            
            print(test_game.display())
            
        reward = test_game.reward()
        
        if reward != -1:
            
            if reward > 0:
                
                status = 2
            
                if display:
                
                    print("Game won! Reward: %s" % (reward, ))
                    
                break
                
            else:
                
                status = 0
              
                if display:
                
                    print("Game lost! Reward: %s" % (reward, ))    
                    
                break
                
                
        i = i + 1
        
        if i > 15:
            
            if display:
                
                print("Game lost; too many moves.")
                
            break
        
    win = True if status == 2 else False
    
    return win
                                                                        




