import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def plot_3D(robot_step : list[float] , obj_step : list[float], sim_env : str) -> None:
    """
    Plot the robot and object steps in 3D
    :param robot_step: (list[float]) the robot steps
    :param obj_step: (list[float]) the object steps
    :param sim_env: (str) the simulation environment
    """

    # plot the robot and object position
    robot_x = []
    robot_y = []
    robot_z = []
    obj_x = []
    obj_y = []
    obj_z = []

    for pos_robot in robot_step:
        robot_x.append(pos_robot[0])
        robot_y.append(pos_robot[1])
        robot_z.append(pos_robot[2])
    for pos_obj in obj_step:
        obj_x.append(pos_obj[0])
        obj_y.append(pos_obj[1])
        obj_z.append(pos_obj[2])


    t = np.arange(len(robot_x))
    # create a color map from middle blue to dark blue
    cm_robot = mpl.cm.Blues(np.linspace(0, 1, 100))
    cm_robot = mpl.colors.ListedColormap(cm_robot[50:, :-1])
    col_robot = [cm_robot(i / len(robot_x)) for i in range(len(robot_x))]

    # create a color map from middle orange to dark orange 'YlOrBr'
    cm_obj = mpl.cm.YlOrBr(np.linspace(0, 1, 100))
    cm_obj = mpl.colors.ListedColormap(cm_obj[40:, :-1])
    col_obj = [cm_obj(i / len(obj_x)) for i in range(len(obj_x))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if "acorn" in sim_env :
        title = "Acorn"
    elif "ball" in sim_env :
        title = "Sand ball"
    elif "cube" in sim_env :
        title = "Sugar cube"
    elif "crumb" in sim_env :
        title = "Bread crumb"

    ax.set_title(title)
    ax.grid(True)
    ax.scatter(robot_x, robot_y, robot_z, col_robot, c=t, cmap=cm_robot, s=20, linewidths=4, label="Robot")
    ax.scatter(obj_x, obj_y, obj_z, col_obj, c=t, cmap=cm_obj, s=20, linewidths=4,  label="Object")

    ax.view_init(elev=30, azim=-80)

    plt.legend(loc=(0.7, 0.9))
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_linewidth(5.0)
    leg.legendHandles[1].set_color('orange')
    leg.legendHandles[1].set_linewidth(5.0)

    plt.show()
    return None