import pygame
import neat
import os
import time
import pickle
from pong import Game  # Assuming pong module contains the Game class


class PongGame:
    def __init__(self, window, width, height):
        """
        Initialize PongGame object.

        Parameters:
        - window: Pygame window
        - width: Width of the game window
        - height: Height of the game window
        """
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

    def test_ai(self, neural_net):
        """
        Test the AI against a human player by passing a NEAT neural network.

        Parameters:
        - neural_net: NEAT neural network
        """
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            output = neural_net.activate((self.right_paddle.y, abs(
                self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            # Update AI paddle based on neural network decision
            if decision == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:
                self.game.move_paddle(left=False, up=False)

            # Update human player paddle based on key input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # Draw the game and update display
            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome1, genome2, config, draw=False):
        """
        Train the AI by passing two NEAT neural networks and the NEAT config object.
        These AIs will play against each other to determine their fitness.

        Parameters:
        - genome1: NEAT genome for the first AI
        - genome2: NEAT genome for the second AI
        - config: NEAT configuration object
        - draw: Flag indicating whether to draw the game (default: False)

        Returns:
        - Boolean indicating if a forced quit occurred
        """
        running = True
        start_time = time.time()

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        self.genome1 = genome1
        self.genome2 = genome2

        max_hits = 50

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()

            # Move AI paddles based on neural network decisions
            self.move_ai_paddles(net1, net2)

            # Draw the game with optional features
            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            # Check for game end conditions and calculate fitness
            if game_info.left_score == 1 or game_info.right_score == 1 or game_info.left_hits >= max_hits:
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, neural_net1, neural_net2):
        """
        Determine where to move the left and the right paddle based on the two 
        neural networks that control them. 

        Parameters:
        - neural_net1: Neural network for the left paddle
        - neural_net2: Neural network for the right paddle
        """
        players = [(self.genome1, neural_net1, self.left_paddle, True),
                   (self.genome2, neural_net2, self.right_paddle, False)]
        for (genome, neural_net, paddle, left) in players:
            output = neural_net.activate(
                (paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:
                genome.fitness -= 0.01  # Discourage not moving
            elif decision == 1:
                valid = self.game.move_paddle(left=left, up=True)
            else:
                valid = self.game.move_paddle(left=left, up=False)

            # Punish AI if the movement makes the paddle go off the screen
            if not valid:
                genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        """
        Calculate fitness for the given game information and duration.

        Parameters:
        - game_info: Information about the game state
        - duration: Time duration of the game
        """
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration


def evaluate_genomes(genomes, config):
    """
    Run each genome against each other one time to determine the fitness.

    Parameters:
    - genomes: List of NEAT genomes
    - config: NEAT configuration object
    """
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i+1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            pong = PongGame(win, width, height)

            # Train AI and handle forced quit event
            force_quit = pong.train_ai(genome1, genome2, config, draw=True)
            if force_quit:
                quit()


def run_neat_algorithm(config):
    """
    Run the NEAT algorithm to evolve the AI.

    Parameters:
    - config: NEAT configuration object
    """
    # Load configuration and set up NEAT population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))

    # Run NEAT to evolve the AI
    winner = population.run(evaluate_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_best_neural_network(config):
    """
    Test the best-performing neural network against a human player.

    Parameters:
    - config: NEAT configuration object
    """
    # Load the best-performing AI and test against a human player
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_neural_net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, width, height)
    pong.test_ai(winner_neural_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              config_path)

    run_neat_algorithm(neat_config)
    test_best_neural_network(neat_config)
