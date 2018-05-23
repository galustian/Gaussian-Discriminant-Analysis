import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
        
_shift_pressed = False
_datapoints = []
_cmap = mpl.colors.ListedColormap(['blue', 'red'])
_clf = None

def _onkeypress(event):
    global _shift_pressed
    if event.key == 'shift':
        _shift_pressed = True

def _onkeyrelease(event):
    global _shift_pressed
    if event.key == 'shift':
        _shift_pressed = False

def _onclick(event):
    global _shift_pressed
    global _datapoints
    global _cmap
    global _clf

    plt.cla()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Draw _datapoints
    if _shift_pressed == False:
        _datapoints.append(np.array([event.xdata, event.ydata, 0]))
    else:
        _datapoints.append(np.array([event.xdata, event.ydata, 1]))
    
    _datapoints_arr = np.array(_datapoints)
    
    plt.scatter(_datapoints_arr[:, 0], _datapoints_arr[:, 1], c=_datapoints_arr[:, 2], cmap=_cmap)
    fig.canvas.draw()

    x_range = np.linspace(_datapoints_arr[:, 0].min(), _datapoints_arr[:, 0].max(), 50)
    y_range = np.linspace(_datapoints_arr[:, 1].min(), _datapoints_arr[:, 1].max(), 50)
    xx, yy = np.meshgrid(x_range, y_range)

    X = _datapoints_arr[:, :2]
    Y = _datapoints_arr[:, -1]

    if len(np.unique(Y)) > 1:
        _clf.fit(X, Y)
        Y_hat = _clf.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.contourf(xx, yy, Y_hat.reshape(xx.shape), cmap=_cmap, alpha=0.25, antialiased=True)
        fig.canvas.draw()

# ------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
# The FigureCanvas method mpl_connect() returns a connection id which is simply an integer.
# When you want to disconnect the callback, just call:
# fig.canvas.mpl_disconnect(cid)
cid1 = fig.canvas.mpl_connect('button_press_event', _onclick)
cid2 = fig.canvas.mpl_connect('key_press_event', _onkeypress)
cid2 = fig.canvas.mpl_connect('key_release_event', _onkeyrelease)

# pass an instance of the classifier
# must implement fit and predict method
def plot_decision_region(clf):
    global _clf
    _clf = clf
    plt.show()