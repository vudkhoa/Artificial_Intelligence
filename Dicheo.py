import math
from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import heapq

# Các chi phí di chuyển
cost_regular = 1.0
cost_diagonal = 1.7

COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
    "up left": cost_diagonal,
    "up right": cost_diagonal,
    "down left": cost_diagonal,
    "down right": cost_diagonal,
}

# Bản đồ mê cung
MAP_STRING = """
#########################################
#                                       #
# #### ########### ########### ######  ##
# #   #           #           #      #  #
# #   # ######### # ######### # #### #  #
# #                 #                 # #
# ###### ##### ### # ### ##### ###### # #
#        #   # #   #   # #   #        # #
# ######## # # # ##### # # # ######## # #
# #       # # # #     # # # #       # # #
# # ##### # # ### ### # # ### ##### # # #
# #   #   # #   # #   # #   #   #   # # #
# ### # ##### ### ##### ### ##### # ### #
#     #       #   #   #   #       #     #
####### ### # ##### # ##### # ### #######
#         # #       #       # #         #
# ### ### # ##### # # ##### # ### ### ###
#   #   # #   #   # # #   # #   #   #   #
##### ### ### ### ### ### ### ### ### ###
#       #   #           #   #       #   #
# ####### # ##### ### ##### # ### # ### #
# #       #     # # #   #   #   # #   # #
# ### ##### ### ### ### ### # # ##### # #
#     #   # #   #   #   #   # #     # # #
### ##### # ##### ### ### ### ### # # ###
# #       #       #     #   #   # # #   #
# # # ##### ########### ### ### ### ### #
#   #               #       #           #
#########################################
"""

MAP = [list(row) for row in MAP_STRING.strip().split('\n')]
M, N = len(MAP), len(MAP[0])
W = 25

mau_den = np.zeros((W, W, 3), np.uint8) + (100, 100, 100)
mau_trang = np.zeros((W, W, 3), np.uint8) + (255, 255, 255)


def build_image_from_map(map_data):
    image = np.ones((M * W, N * W, 3), np.uint8) * 255
    for y in range(M):
        for x in range(N):
            cell = map_data[y][x]
            if cell == '#':
                image[y * W:(y + 1) * W, x * W:(x + 1) * W] = mau_den
            elif cell == ' ':
                image[y * W:(y + 1) * W, x * W:(x + 1) * W] = mau_trang
    return image


class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.initial = self.goal = None
        for y in range(len(board)):
            for x in range(len(board[0])):
                if board[y][x] == 'o':
                    self.initial = (x, y)
                elif board[y][x] == 'x':
                    self.goal = (x, y)
        if self.initial is None or self.goal is None:
            raise ValueError("Missing 'o' or 'x' in map")
        super().__init__(initial_state=self.initial)

    def actions(self, state):
        actions = []
        for action in COSTS:
            newx, newy = self.result(state, action)
            if 0 <= newx < N and 0 <= newy < M and self.board[newy][newx] != '#' and action.split().__len__() == 1:
                actions.append(action)
            elif 0 <= newx < N and 0 <= newy < M and self.board[newy][newx] != '#' and action.split().__len__() == 2:
                # print(action)
                if (action == 'down right' and self.board[newy][newx - 1] != '#' and self.board[newy - 1][newx] != '#'):
                    actions.append(action)
                elif (action == 'down left' and self.board[newy][newx + 1] != '#' and self.board[newy - 1][newx] != '#'):
                    actions.append(action)
                elif (action == 'up left' and self.board[newy][newx + 1] != '#' and self.board[newy + 1][newx] != '#'):
                    actions.append(action)
                elif (action == 'up right' and self.board[newy][newx - 1] != '#' and self.board[newy + 1][newx] != '#'):
                    actions.append(action)
        return actions

    def result(self, state, action):
        x, y = state
        if 'up' in action:
            y -= 1
        if 'down' in action:
            y += 1
        if 'left' in action:
            x -= 1
        if 'right' in action:
            x += 1
        return (x, y)

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y) # Công thức tính khoảng cách trong không gian 2 chiều.


def a_star(problem):
    start = problem.initial
    goal = problem.goal

    open_list = []
    heapq.heappush(open_list, (0 + problem.heuristic(start), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: problem.heuristic(start)}

    while open_list:
        _, g, current, parent = heapq.heappop(open_list)

        if problem.is_goal(current):
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
            return path

        for action in problem.actions(current):
            neighbor = problem.result(current, action)
            tentative_g_score = g + problem.cost(current, action, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + problem.heuristic(neighbor)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor, current))
                came_from[neighbor] = current

    return None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tìm đường trong mê cung")
        self.dem = 0
        self.old_start_x = self.old_start_y = -1
        self.old_goal_x = self.old_goal_y = -1

        # Copy bản đồ gốc để reset
        self.original_map = [row.copy() for row in MAP]
        self.map_data = [row.copy() for row in MAP]

        self.image = build_image_from_map(self.map_data)
        self.pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.image_tk = ImageTk.PhotoImage(self.pil_image)

        self.cvs_me_cung = tk.Canvas(self, width=N * W, height=M * W,
                                     relief=tk.SUNKEN, border=2)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.cvs_me_cung.bind("<Button-1>", self.xu_ly_mouse)

        lbl_frm_menu = tk.LabelFrame(self)
        btn_reset = tk.Button(lbl_frm_menu, text='Reset', width=7,
                              command=self.btn_reset_click, bg='pink')
        btn_reset.grid(row=1, column=0, padx=5, pady=5, sticky=tk.N)

        self.cvs_me_cung.grid(row=0, column=0, padx=5, pady=5)
        lbl_frm_menu.grid(row=0, column=1, padx=5, pady=7, sticky=tk.NW)

        messagebox.showinfo('Thông báo', 'Click chọn điểm bắt đầu (o), sau đó chọn điểm kết thúc (x)')

    def xu_ly_mouse(self, event):
        x, y = event.x // W, event.y // W
        if not (0 <= x < N and 0 <= y < M) or self.map_data[y][x] == '#':
            return

        if self.dem == 0:
            if self.old_start_x != -1:
                self.map_data[self.old_start_y][self.old_start_x] = ' '
                self.vẽ_ô(self.old_start_x, self.old_start_y, '#FFFFFF')
            self.map_data[y][x] = 'o'
            self.vẽ_ô(x, y, '#BF8725', hình_tròn=True)
            self.old_start_x, self.old_start_y = x, y
            self.dem = 1

        elif self.dem == 1:
            if self.old_goal_x != -1:
                self.map_data[self.old_goal_y][self.old_goal_x] = ' '
                self.vẽ_ô(self.old_goal_x, self.old_goal_y, '#FFFFFF')
            self.map_data[y][x] = 'x'
            self.vẽ_ô(x, y, '#BF8725', hình_tròn=False)
            self.old_goal_x, self.old_goal_y = x, y
            self.dem = 2

            try:
                problem = MazeSolver(self.map_data)
                result = astar(problem, graph_search=True)
                path = [x[1] for x in result.path()]
                for (x, y) in path:
                    if (x, y) not in [problem.initial, problem.goal]:
                        self.vẽ_ô(x, y, '#817821')
                        self.update()
                        time.sleep(0.05)
                messagebox.showinfo('Kết quả', 'Tìm đường thành công bằng thư viện!')
            except Exception as e:
                messagebox.showerror('Lỗi', str(e))


            try:
                problem = MazeSolver(self.map_data)
                path = a_star(problem)
                for (x, y) in path:
                    if (x, y) not in [problem.initial, problem.goal]:
                        self.vẽ_ô(x, y, '#ADD8E6')
                        self.update()
                        time.sleep(0.05)
                messagebox.showinfo('Kết quả', 'Tìm đường thành công bằng code tự xây!')
            except Exception as e:
                messagebox.showerror('Lỗi', str(e))

    def vẽ_ô(self, x, y, màu, hình_tròn=False):
        if hình_tròn:
            self.cvs_me_cung.create_oval(x * W + 2, y * W + 2, (x + 1) * W - 2, (y + 1) * W - 2,
                                         outline=màu, fill=màu)
        else:
            self.cvs_me_cung.create_rectangle(x * W + 2, y * W + 2, (x + 1) * W - 2, (y + 1) * W - 2,
                                              outline=màu, fill=màu)

    def btn_reset_click(self):
        self.dem = 0
        self.old_start_x = self.old_start_y = -1
        self.old_goal_x = self.old_goal_y = -1

        self.map_data = [row.copy() for row in self.original_map]
        self.image = build_image_from_map(self.map_data)
        self.pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.image_tk = ImageTk.PhotoImage(self.pil_image)

        self.cvs_me_cung.delete(tk.ALL)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)


if __name__ == "__main__":
    app = App()
    app.mainloop()
