from Game import Connect4
from Search import APV_MCTS, Node


def simulate_game(evaluator, search_iters: int = 100) -> tuple:
    game = Connect4()

    root = Node(game.copy())
    mcts = APV_MCTS(root)

    boards = [game.board.copy()]
    masks = [game.get_impossible_actions()]
    probs = []

    num_moves = 0
    while not game.get_is_terminal():
            
        mcts.search(evaluator, iters=search_iters)
        move, action_probs = mcts.select_play_action()

        probs.append(action_probs)

        game.do_move(move)

        boards.append(game.board.copy())
        masks.append(game.get_impossible_actions())
        num_moves += 1
    probs.append([0]*7)

    value = game.value
    values = [value*(-1)**i for i in range(1+num_moves)][::-1]
    return game, mcts, boards, masks, probs, values

