
def ei_zeichnen(t, bildunterschrift):
   import matplotlib.pyplot as plt
   plt.imshow( t.permute(1, 2, 0), cmap="gray" )
