
def ei_zeichnen(t):
   import matplotlib.pyplot as plt
   plt.imshow( t.permute(1, 2, 0), cmap="gray" )
