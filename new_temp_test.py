import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

plt.plot([1,3,2], label='something')
plt.plot([.5,.5], [1,3], label='something else')

def update_prop(handle, orig):
    handle.update_from(orig)
    x,y = handle.get_data()
    handle.set_data([np.mean(x)]*2, [0, 2*y[0]])

plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

plt.show()