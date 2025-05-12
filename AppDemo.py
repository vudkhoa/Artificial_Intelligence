import math
from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import heapq

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.7

COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
}

MAP = """
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

# Convert map to a list
MAP = [list(x) for x in MAP.split("\n") if x]
M = len(MAP)
N = len(MAP[0])
W = 25

# Draw maze background
mau_den     = np.zeros((W, W, 3), np.uint8) + (100, 100, 100)
mau_trang   = np.zeros((W, W, 3), np.uint8) + (255, 255, 255)
mau_do      = np.zeros((W, W, 3), np.uint8) + (255, 0, 0)
image       = np.ones((M*W, N*W, 3), np.uint8) * 255

for x in range(M):
    for y in range(N):
        if MAP[x][y] == '#':
            image[x*W:(x+1)*W, y*W:(y+1)*W] = mau_den
        elif MAP[x][y] == ' ':
            image[x*W:(x+1)*W, y*W:(y+1)*W] = mau_trang

color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_coverted)

class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.goal = None
        self.initial = None

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        if self.initial is None or self.goal is None:
            raise ValueError("Missing start point ('o') or goal point ('x') in the map.")

        super(MazeSolver, self).__init__(initial_state=self.initial)

    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if 0 <= newx < len(self.board[0]) and 0 <= newy < len(self.board):
                if self.board[newy][newx] != "#":
                    actions.append(action)
        return actions

    def result(self, state, action):
        x, y = state
        if "up" in action:
            y -= 1
        if "down" in action:
            y += 1
        if "left" in action:
            x -= 1
        if "right" in action:
            x += 1
        return (x, y)

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.hypot(x - gx, y - gy)
    
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
        self.dem = 0
        self.title('Tìm đường trong mê cung')

        # relief=tk.SUNKEN, border=2: Hiệu ứng viền. 
        self.cvs_me_cung = tk.Canvas(self, width=N*W, height=M*W, relief=tk.SUNKEN, border=2)
        # hiện thị hình lên canvas
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.cvs_me_cung.bind("<Button-1>", self.xu_ly_mouse)

        lbl_frm_menu = tk.LabelFrame(self)
        btn_reset = tk.Button(lbl_frm_menu, text='Reset', width=7,
                              command=self.btn_reset_click, bg='pink')
        btn_reset.grid(row=1, column=0, padx=5, pady=5, sticky=tk.N)

        self.cvs_me_cung.grid(row=0, column=0, padx=5, pady=5)
        lbl_frm_menu.grid(row=0, column=1, padx=5, pady=7, sticky=tk.NW)

        self.old_start_x = self.old_start_y = -1
        self.old_goal_x = self.old_goal_y = -1

        messagebox.showinfo('announcement', 'Chọn điểm bắt đầu (start) rồi chọn điểm đích (goal) trên mê cung.')

    def xu_ly_mouse(self, event):
        px, py = event.x, event.y
        x, y = px // W, py // W  # x là cột, y là hàng

        if y >= len(MAP) or x >= len(MAP[0]) or MAP[y][x] == '#':
            return

        if self.dem == 0:
            # Đặt điểm bắt đầu
            if self.old_start_x != -1:
                # Xóa điểm cũ
                MAP[self.old_start_y][self.old_start_x] = ' '
            MAP[y][x] = 'o'
            self.old_start_x, self.old_start_y = x, y
            # Vẽ điểm mới
            self.cvs_me_cung.create_oval(x*W+2, y*W+2, (x+1)*W-2, (y+1)*W-2, outline='#BF8725', fill='#BF8725')
            self.dem += 1

        elif self.dem == 1:
            # Kiểm tra trùng với điểm bắt đầu
            if x == self.old_start_x and y == self.old_start_y:
                messagebox.showerror('Lỗi', 'Điểm đích không được trùng với điểm bắt đầu.')
                return
            # Đặt điểm kết thúc
            if self.old_goal_x != -1:
                # Xóa điểm cũ
                MAP[self.old_goal_y][self.old_goal_x] = ' '
            MAP[y][x] = 'x'
            self.old_goal_x, self.old_goal_y = x, y
            # Vẽ điểm mới
            self.cvs_me_cung.create_rectangle(x*W+2, y*W+2, (x+1)*W-2, (y+1)*W-2, outline='#BF8725', fill='#BF8725')
            self.dem += 1

            # Astar code library
            try:
                problem = MazeSolver(MAP)
                result = astar(problem, graph_search=True)
                if result is None:
                    messagebox.showwarning('Kết quả', 'Không tìm thấy đường đi từ điểm bắt đầu đến điểm đích.')
                else:
                    path = [state[1] for state in result.path()]
                    for (px, py) in path:
                        self.cvs_me_cung.create_rectangle(px*W+2, py*W+2, (px+1)*W-2, (py+1)*W-2, outline='#817821', fill='#817821')
                        time.sleep(0.05)
                        self.cvs_me_cung.update()
                    messagebox.showinfo('Result', 'Đã tìm thấy đường đi bằng thư viện!')
            except ValueError as e:
                messagebox.showerror('Error', str(e))

            # Astar code tay.
            try:
                problem = MazeSolver(MAP)
                path = a_star(problem)
                if path is None:
                    messagebox.showwarning('Kết quả', 'Không tìm thấy đường đi từ điểm bắt đầu đến điểm đích.')
                else:
                    for (px, py) in path:
                        self.cvs_me_cung.create_rectangle(px*W+2, py*W+2, (px+1)*W-2, (py+1)*W-2, outline = '#1E90FF', fill = '#ADD8E6')
                        time.sleep(0.05)
                        self.cvs_me_cung.update()
                    messagebox.showinfo('Result', 'Đã tìm thấy đường đi bằng code tự xây!')
            except ValueError as e:
                messagebox.showerror('Error', str(e))

            except Exception as e:
                messagebox.showerror('Error', f"Lỗi không xác định: {e}")

    def btn_reset_click(self):
        self.cvs_me_cung.delete(tk.ALL)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.dem = 0
        for x in range(M):
            for y in range(N):
                if MAP[x][y] in ('o', 'x'):
                    MAP[x][y] = ' '

if __name__ == "__main__":
    app = App()
    app.mainloop()
