
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
    import torch
    from torch.autograd import Variable
    n_data_train = torch.ones(200, 2)
    n_data_test = torch.ones(50, 2)
    x0 = torch.normal(n_data_train + torch.tensor([2.5,5]), 1)      # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(200)               # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(n_data_train + torch.tensor([8,2]), 1)     # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(200)                # class1 y data (tensor), shape=(100, 1)
    x_train = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y_train = torch.cat((y0, y1), ).type(torch.LongTensor)  
    
    n_data_test = torch.ones(50, 2)
    x0 = torch.normal(n_data_test + torch.tensor([2.5,5]), 1)      # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(50)               # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(n_data_test + + torch.tensor([8,2]), 1)     # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(50)                # class1 y data (tensor), shape=(100, 1)
    x_test = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y_test = torch.cat((y0, y1), ).type(torch.LongTensor) 
    
    return Variable(x_train), Variable(y_train), Variable(x_test), Variable(y_test)
  
def datenpunkte_zeichnen(x_data, labels, farben):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    plt.scatter(x_data.data.numpy()[:, 0], x_data.data.numpy()[:, 1], 
            c=labels.data.numpy(), s=50, cmap=colors.ListedColormap(farben))
    plt.show()
    
def daten2():
    import torch
    from torch.autograd import Variable
    n_data_train = torch.ones(200, 2)
    n_data_test = torch.ones(50, 2)
    x0 = torch.normal(n_data_train + torch.tensor([6,5]), 1)      
    y0 = torch.zeros(200)
    x1 = torch.normal(n_data_train + torch.tensor([2,2]), 1)   
    y1 = torch.ones(200)          
    x2 = torch.normal(n_data_train + torch.tensor([10,2]), 1) 
    y2 = 2 * torch.ones(200) 
    x_train = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)
    y_train = torch.cat((y0, y1, y2), ).type(torch.LongTensor)  
    
    n_data_test = torch.ones(50, 2)
    x0 = torch.normal(n_data_test + torch.tensor([6,5]), 1)      
    y0 = torch.zeros(50)
    x1 = torch.normal(n_data_test + torch.tensor([2,2]), 1)   
    y1 = torch.ones(50)          
    x2 = torch.normal(n_data_test + torch.tensor([10,2]), 1) 
    y2 = 2 * torch.ones(50) 
    x_test = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor) 
    y_test = torch.cat((y0, y1, y2), ).type(torch.LongTensor) 
    
    return Variable(x_train), Variable(y_train), Variable(x_test), Variable(y_test)
