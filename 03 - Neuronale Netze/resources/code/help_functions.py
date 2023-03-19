
def pruefe_gewichte(nn, gewicht1, gewicht2, gewicht3, gewicht4):
    import torch
    g1 = round(nn.fc1.weight[0][0].item(), 4)
    g2 = round(nn.fc1.bias[4].item(), 4)
    g3 = round(nn.fc2.bias[1].item(), 4)
    g4 = round(nn.fc3.weight[2][3].item(), 4)
    
    wrong = False
    
    result = ""
    if g1 == gewicht1:
      result = "Das erste Gewicht hast du richtig abgelesen!\n"
    else:
      result = "Das erste Gewicht hast du nicht richtig abgelesen!\n"
      wrong = True

    if g2 == gewicht2:
      result += "Das zweite Gewicht hast du richtig abgelesen!\n"
    else:
      result += "Das zweite Gewicht hast du nicht richtig abgelesen!\n"
      wrong = True

    if g3 == gewicht3:
      result += "Das dritte Gewicht hast du richtig abgelesen!\n"
    else:
      result += "Das dritte Gewicht hast du nicht richtig abgelesen!\n"
      wrong = True 

    if g4 == gewicht4:
      result += "Das vierte Gewicht hast du richtig abgelesen!\n"
    else:
      result += "Das vierte Gewicht hast du nicht richtig abgelesen!\n"
      wrong = True

    if wrong:
      return result
    else:
      return "Super! Du hast alle Gewichte richtig abgelesen!"
      
def daten():
    import random
    import math	
    import matplotlib.pyplot as plt
    daten = []
    x_klasse_0 = []
    y_klasse_0 = []
    x_klasse_1 = []
    y_klasse_1 = []
    for _ in range(250):
	    x = random.uniform(0,1) + 0
	    y = random.uniform(0,1) + 2
	    daten.append(((x,y), 0))
	    x_klasse_0.append(x)
	    y_klasse_0.append(y)
	    
    for _ in range(250):
	    x = random.uniform(0,1) + 2
	    y = random.uniform(0,1) + 0
	    daten.append(((x,y), 1))
	    x_klasse_1.append(x)
	    y_klasse_1.append(y)
	    
    plt.scatter(x_klasse_0, y_klasse_0, color='#ec90cc')
    plt.scatter(x_klasse_1, y_klasse_1, color='#4f7087')
	  
    return daten, plt
