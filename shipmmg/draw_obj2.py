#描写用のクラス
import numpy as np
import matplotlib.pyplot as plt

class DrawObj():
    def __init__(self, ax):
        self.ax = ax
        self.img = []
        self.img2 = []
        self.img.append(ax.plot([], [], color="b"))
        self.img.append(ax.plot([], [], color="y"))
        self.img2.append(ax.plot([], [], color="r"))
        self.img2.append(ax.plot([], [], color="y"))

    def draw_square_with_angle(self, center_x_list, center_y_list, shape_list, angle_list):
        for i in range(len(shape_list)):
            square_x, square_y, angle_x, angle_y = self.square_with_angle(center_x_list[i], center_y_list[i], shape_list[i], angle_list[i])
            self.img[i][0].set_xdata(square_x)
            self.img[i][0].set_ydata(square_y)
        return self.img

    def draw_square_with_angle2(self, center_x_list2, center_y_list2, shape_list2, angle_list2):
        for i in range(len(shape_list2)):
            square_x2, square_y2, angle_2, angle_y2 = self.square_with_angle2(center_x_list2[i], center_y_list2[i], shape_list2[i], angle_list2[i])
            self.img2[i][0].set_xdata(square_x2)
            self.img2[i][0].set_ydata(square_y2)
        return self.img2

    def rotate_pos(self, pos, angle):
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

        return np.dot(pos, rot_mat.T)

    def rotate_pos2(self, pos2, angle2):
        rot_mat2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                            [np.sin(angle2), np.cos(angle2)]])

        return np.dot(pos2, rot_mat2.T)

    def square(self, center_x, center_y, shape, angle):
        """ Create square
        Args:
            center_x (float): the center x position of the square
            center_y (float): the center y position of the square
            shape (tuple): the square's shape(width/2, height/2)
            angle (float): in radians
        Returns: 
            square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
        """
        # start with the up right points
        # create point in counterclockwise, local
        square_xy = np.array([[shape[0], shape[1]],
                              [-shape[0], shape[1]],
                              [-shape[0], -shape[1]],
                              [shape[0], -shape[1]],
                              [shape[0], shape[1]]])

        # translate position to world
        # rotation
        trans_points = self.rotate_pos(square_xy, angle)
        # translation
        trans_points += np.array([center_x, center_y])

        return trans_points[:, 0], trans_points[:, 1]
    
    def square2(self, center_x2, center_y2, shape2, angle2):
        square_xy2 = np.array([[shape2[0], shape2[1]],
                              [-shape2[0], shape2[1]],
                              [-shape2[0], -shape2[1]],
                              [shape2[0], -shape2[1]],
                              [shape2[0], shape2[1]]])

        # translate position to world
        # rotation
        trans_points2 = self.rotate_pos2(square_xy2, angle2)
        # translation
        trans_points2 += np.array([center_x2, center_y2])

        return trans_points2[:, 0], trans_points2[:, 1]
        
    
    def square_with_angle(self,center_x, center_y, shape, angle):
        """ Create square with angle line
        Args:    
            center_x (float): the center x position of the square
            center_y (float): the center y position of the square
            shape (tuple): the square's shape(width/2, height/2)
            angle (float): in radians
        Returns: 
            square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
            square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
            angle_x (numpy.ndarray): x data of square angle
            angle_y (numpy.ndarray): y data of square angle
        """
        square_x, square_y = self.square(center_x, center_y, shape, angle)

        angle_x = np.array([center_x, center_x + np.cos(angle) * shape[0]])
        angle_y = np.array([center_y, center_y + np.sin(angle) * shape[1]])

        return square_x, square_y, angle_x, angle_y
    
    def square_with_angle2(self,center_x2, center_y2, shape2, angle2):
        
        square_x2, square_y2 = self.square2(center_x2, center_y2, shape2, angle2)
        angle_x2 = np.array([center_x2, center_x2 + np.cos(angle2) * shape2[0]])
        angle_y2 = np.array([center_y2, center_y2 + np.sin(angle2) * shape2[1]])

        return square_x2, square_y2, angle_x2, angle_y2
    
def update_obj(i, x_list, y_list, shape_list, ψ_list, x_list2, y_list2, shape_list2, ψ_list2, frate):
    j = int(frate*i)
    plt.title(r'$t$ = ' + '{:.1f}'.format(time_list[j]))
    
    xT = np.array(x_list).T
    _x_list_j = list(xT[j].T)
    yT = np.array(y_list).T
    _y_list_j = list(yT[j].T)
    ψT = np.array(ψ_list).T
    _ψ_list_j = list(ψT[j].T)
    
    xT2 = np.array(x_list2).T
    _x_list_j2 = list(xT2[j].T)
    yT2 = np.array(y_list2).T
    _y_list_j2 = list(yT2[j].T)
    ψT2 = np.array(ψ_list2).T
    _ψ_list_j2 = list(ψT2[j].T)
    
    return drawer.draw_square_with_angle(_x_list_j, _y_list_j,shape_list,_ψ_list_j),drawer.draw_square_with_angle2(_x_list_j2, _y_list_j2,shape_list2,_ψ_list_j2)

#使用例
'''
t=np.linspace(0,n_steps,n_steps)　#(開始時間, 終了時間, データ数)
​
time_list=t
​
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
​
plt.plot(x,y,label='MPC',ls='-')
plt.plot(x_ref,y_ref,label=ref_name,ls='--')
plt.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
​
drawer=DrawObj(ax)
​
frate=1.0
frames=int(t[-1]/frate)
​
ani1=FuncAnimation(fig,update_obj,fargs=([x],[y],[(L_pp/2,B/2)],[psi],[x_ref],[y_ref],[(L_pp/2,B/2)],[psi_ref],frate),interval=100,frames=frames)
​
ani1.save(dirname+'ship_anim.gif',writer='pillow')
'''