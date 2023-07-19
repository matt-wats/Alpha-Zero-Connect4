import Game
import pygame
import numpy as np

import Search
import Model
import torch

model = torch.load("./Trained_Model/model.pt")
model.load_state_dict(torch.load("./Trained_Model/model_state.pt"))
evaluator = Model.Evaluator(model)


# Constants
BOARD_ROWS = 6
BOARD_COLS = 7
SQUARE_SIZE = 100
RADIUS = (SQUARE_SIZE // 2) - 5

WINDOW_WIDTH = BOARD_COLS*SQUARE_SIZE
WINDOW_HEIGHT = (BOARD_ROWS+1)*SQUARE_SIZE

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Connect 4")

# Draw the Connect 4 board
def draw_board(board):
    window.fill((255, 255, 255))
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            pygame.draw.rect(window, BLUE, (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2)
            if board[row][col] == 0:
                pygame.draw.circle(window, BLACK, center, RADIUS)
            if board[row][col] == 1:
                pygame.draw.circle(window, RED, center, RADIUS)
            elif board[row][col] == 2:
                pygame.draw.circle(window, YELLOW, center, RADIUS)

# Get the column index based on the mouse position
def get_column_index(pos):
    x = pos[0] // SQUARE_SIZE
    return x

# Main game loop
def game_loop(b: Game.Connect4):
    running = True
    while running:
        if b.flipped:
            try:
                root = Search.Node(b.copy())
                mcts = Search.APV_MCTS(root)
                mcts.search(evaluator, 300)
                computer_move, policies = mcts.select_play_action()
                computer_move = np.argmax(policies)
                b.do_move(computer_move)
            except:
                print("Computer couldn't find a move")
                running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Get the column index where the mouse was clicked
                    column = get_column_index(pygame.mouse.get_pos())
                    try:
                        b.do_move(column)
                    except:
                        print("Can't perform action")
                        pass
        

        # Draw board
        tensor_board = (b.board[0] + 2*b.board[1]) if not b.flipped else (b.board[1] + 2*b.board[0])
        draw_board(tensor_board)
        pygame.display.update()

    pygame.quit()



# Run the game
b = Game.Connect4()
game_loop(b)
if b.is_terminal:
    if b.value == 1:
        winner = 1 + (not b.flipped)
        print(f"Congrats to player {winner} for winning")
    elif b.value == 0:
        print("It was a draw !")
