import numpy as np
from Game import Connect4

class Node():
    def __init__(self, state: Connect4, parent = None, prob: float = 1.0, move=None) -> None:
        self.visit_count = 0.0            #N
        self.total_action_value = 0.0     #W
        self.mean_action_value = 0.0      #Q
        self.prior_probability = prob   #P

        self.move = move

        self.state = state
        self.parent = parent
        self.children = []
        self.action_probabilities = None

    def select_child(self, c_puct: float = 1.5):
        assert len(self.children) > 0, "Can't Select Action if Possible Actions Don't Exist"
        
        best_score = float("-inf")
        best_child = None
        sqrt_visit_count = np.sqrt(self.visit_count-1)
        for child in self.children:
            Q = child.mean_action_value
            U = c_puct * child.prior_probability * sqrt_visit_count / (1 + child.visit_count)
            score = Q+U
            if score > best_score:
                best_score = score
                best_child = child

        return best_child
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def evaluate_expand(self, evaluator) -> float:
        p, v = evaluator(self.state)
        states, moves = self.state.get_legal_states()
        for state, move in zip(states, moves):
            self.children.append(Node(state, parent=self, prob=p[move], move=move))
        return v



class APV_MCTS():
    def __init__(self, root: Node) -> None:
        self.root = root
        self.c_puct = 1.5
        self.temperature = 1.0 # higher temperature -> more exploration

    def search(self, evaluator, iters: int = 1):
        while self.root.visit_count < iters:
            node = self.root
            while not node.is_leaf():
                selected_child = node.select_child(c_puct=self.c_puct)
                node = selected_child

            v = node.evaluate_expand(evaluator)
            self.backup(node, v)

    def backup(self, node: Node, v: float) -> None:
        node.visit_count += 1
        node.total_action_value += v
        node.mean_action_value = node.total_action_value / node.visit_count
        if not node.parent is None:
            self.backup(node.parent, -v)

    def select_play_action(self):
        children_probabilities = [child.visit_count**self.temperature for child in self.root.children]
        sum_children_probabilities = sum(children_probabilities)
        children_probabilities = [action_probability / sum_children_probabilities for action_probability in children_probabilities]
        chosen_child_idx = np.random.choice(len(children_probabilities), p=children_probabilities)

        action_probabilities = [0]*7
        for idx,child in enumerate(self.root.children):
            action_probabilities[child.move] = children_probabilities[idx]

        # advance root
        self.root = self.root.children[chosen_child_idx]
        self.root.parent = None

        return self.root.move, action_probabilities