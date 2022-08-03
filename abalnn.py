import numpy as np
import os
import bitstring
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from creature_lib import *

parser = argparse.ArgumentParser(description='Agent-Based Artificial Life with Neural Networks by amerotz.')
parser.add_argument('-s', '--seed', type=int, default=0, help='The seed for the simulation. WARNING: the seed is set only at the start, meaning that two consecutive runs of 500 generations with a checkpoint yield different results from one single 1000 generations run.')
parser.add_argument('--height', type=int, default=100, help='Height of the world map.')
parser.add_argument('--width', type=int, default=100, help='Width of the world map.')

parser.add_argument('-hn', '--hidden', type=int, default=4, help='Number of hidden neurons for creatures.')
parser.add_argument('-m', '--mutation', type=float, default=0.01, help='Chance of a creature mutating when reproducing/repopulating.')
parser.add_argument('--genes', type=int, default=4, help='Number of genes in the creatures\' genomes.')

parser.add_argument('--energy', type=int, default=20, help='Starting energy and energy given by a single piece of food.')
parser.add_argument('--food', type=float, default=0.5, help='Percentage of map covered in food.')
parser.add_argument('--eyes', type=int, default=1, help='Visual radius for creatures. 1 means a 3 by 3 grid, 2 means 5 by 5, etc.')
parser.add_argument('--reproduce', action='store_true', help='Enable reproduction.')
parser.add_argument('--kill', action='store_true', help='Enable creatures killing each other.')
parser.add_argument('-p', '--population', type=int, default=100, help='Initial population size.')
parser.add_argument('-g', '--generations', type=int, default=100, help='Number of generations to run the simulation for.')
parser.add_argument('-st', '--steps', type=int, default=100, help='Number of step in a generation.')
parser.add_argument('--continuous', action='store_true', help='Enable continuous simulation: repopulation only happens when all creatures die. Makes sense if reproduction is enabled.')
parser.add_argument('--border', type=str, default='none', help='Add uncrossable borders. One of "none" or "box".')

parser.add_argument('--save', action='store_true', help='Save a checkpoint at the end of this run.')
parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Load a checkpoint.')

parser.add_argument('-v', '--verbose', action='store_true', help='Print logs while running.')
parser.add_argument('--movie', action='store_true', help='Create a movie of every step for every generation. Considerably slows down performance.')
parser.add_argument('--plot_every', type=int, default=10, help='Number of steps between recorded plot data.')
parser.add_argument('--plot', action='store_true', help='Create and save population and genomes plots.')
parser.add_argument('-e', '--every', type=int, default=1, help='How often step logs should be printed.')

args = parser.parse_args()

assert args.border in ['none', 'box']


np.random.seed(args.seed)

sensory_inputs = ['EN', 'AGE']

# food inputs
for i in range((2*args.eyes+1)**2):
    sensory_inputs.append(f'F{i}')

# border inputs
for i in range((2*args.eyes+1)**2):
    sensory_inputs.append(f'B{i}')

# creature inputs
for i in range((2*args.eyes+1)**2):
    sensory_inputs.append(f'C{i}')

hidden_num = args.hidden
hidden_neurons = [f'H{i}' for i in range(hidden_num)]

action_outputs = ['EAT',
                  'x-y-', 'y-', 'x+y-',
                  'x-', 'STOP', 'x+',
                  'x-y+', 'y+', 'x+y+']

if args.reproduce:
    action_outputs.append('REP')

if args.kill:
    action_outputs.append('KILL')

inputs = []
inputs.extend(sensory_inputs)
inputs.extend(hidden_neurons)
in_len = len(inputs)

outputs = []
outputs.extend(hidden_neurons)
outputs.extend(action_outputs)
out_len = len(outputs)


mutation_rate = args.mutation
initial_connections = args.genes

if args.movie:
    os.system('rm movie -fr')
    os.mkdir('movie')

# map size
H = args.height
W = args.width


initial_energy = args.energy
initial_pop = args.population
max_generations = args.generations
generation_steps = args.steps

genealogy = {}


if args.checkpoint != None:
    creatures = pickle.load(open(args.checkpoint, 'rb'))

    # repopulate from last creatures
    parents = np.random.choice(
        np.unique([c.genome for c in creatures]),
        size=initial_pop)

    creatures, pos_map = populate(H, W, initial_pop, initial_energy,
                                  in_len, out_len, hidden_num,
                                  initial_connections, outputs,
                                  mutation_rate, genealogy, parents)

else:
    # populate randomly
    creatures, pos_map = populate(H, W, initial_pop, initial_energy,
                                  in_len, out_len, hidden_num,
                                  initial_connections, outputs,
                                  mutation_rate, genealogy, None)

def log_creatures(plot_data, creatures, generation):
    genes = {}
    for c in creatures:
        g = c.genome
        if not g in genes:
            genes[g] = 0
        genes[g] += 1

    for g in plot_data:
        if not g in genes:
            plot_data[g].append(0)

    for g in genes:
        if not g in plot_data:
            plot_data[g] = [0 for _ in range(generation)]
        plot_data[g].append(genes[g])

def doStep(animals):

    children = []

    # simulate each creature
    for c in animals:

        # cell neighbourhood

        food_cells = []
        border_cells = []
        creature_cells = []
        for i in range(-args.eyes, args.eyes+1):
            py = (c.y + i + H)%H
            for j in range(-args.eyes, args.eyes+1):
                px = (c.x + j + W)%W
                food_cells.append(food_map[py, px])
                border_cells.append(border_map[py, px])
                if py == c.y and px == c.x:
                    creature_cells.append(0)
                else:
                    creature_cells.append(pos_map[py, px])

        food_cells = np.array(food_cells)
        border_cells = np.array(border_cells)
        creature_cells = np.array(creature_cells)

        cells = np.concatenate((food_cells, border_cells, creature_cells))

        actions = c.think(cells)

        # update energy
        c.consumeEnergy(1)
        if c.dead:
            pos_map[c.y, c.x] = 0
            food_map[c.y, c.x] = food
            continue

        if actions == {}:
            continue

        probs = np.array(list(actions.values()))
        probs -= np.min(probs)
        psum = np.sum(probs)
        if psum != 0:
            probs /= psum
        else:
            continue

        action = np.random.choice(list(actions.keys()), p=probs)

        if action == 'EAT':
            if food_map[c.y, c.x] == food and border_map[c.y, c.x] != border:
                c.eat(initial_energy)
                food_map[c.y, c.x] = ground

        elif action == 'REP':
            px = 0
            py = 0
            found = False

            i = -1
            while i < 2 and not found:
                y = (c.y + i + H)%H
                j = -1
                while j < 2 and not found:
                    x = (c.x + j + W)%W
                    if pos_map[y, x] == 0 and border_map[y, x] != border:
                        pos_map[y, x] = 1
                        px = x
                        py = y
                        found = True
                    j += 1
                i += 1

            if found:

                c.consumeEnergy(1)

                child = c.reproduce(px, py, mutation_rate)
                children.append(child)
                updateGenealogy(genealogy, c.genome, child.genome)

        elif action == 'KILL':

            victims = []

            for i in range(-1, 2):
                y = (c.y + i + H)%H
                for j in range(-1, 2):
                    x = (c.x + j + W)%W
                    if c.x == x and c.y == y:
                        continue
                    if pos_map[y, x] != 0:
                        victims.append(f'{x}_{y}')

            if len(victims) != 0:

                killed = np.random.choice(victims).split('_')

                px = int(killed[0])
                py = int(killed[1])

                found = False
                a = 0
                while a  < len(animals) and not found:
                    v = animals[a]
                    if v.y == py and v.x == px:
                        v.die()
                        pos_map[py, px] = 0
                        food_map[py, px] = food
                        found = True
                    a += 1

                c.consumeEnergy(1)

        else:

            # movement
            mx = 0
            my = 0

            if 'x-' in action:
                mx = -1
            elif 'x+' in action:
                mx = 1

            if 'y-' in action:
                my = -1
            elif 'y+' in action:
                my = 1


            px = (c.x + mx + W)%W
            py = (c.y + my + H)%H


            if border_map[py, px] != border and pos_map[py, px] == 0:
                pos_map[c.y, c.x] = 0
                c.move(px, py)
                pos_map[py, px] = 1


        c.older()

    # remove dead
    new_animals = list(filter(lambda x : not x.dead, animals))
    new_animals.extend(children)

    return new_animals


plot_data = {}

if args.plot:
    log_creatures(plot_data, creatures, 0)

repop = False
movie_index = 0
food_map = createFoodMap(H, W, args.food)
border_map = createBorderMap(H, W, args.border)

max_c = 1000

for generation in range(max_generations):


    if args.verbose:
        print(f'GENERATION {generation}')


    print(f'GEN {generation}, STEP 0/{generation_steps}, POP {len(creatures)}')

    repop = False

    # each generation is divided in steps
    for time_step in range(generation_steps):

        if args.movie:
            plt.axis('off')
            plt.imshow(np.maximum(food_map,2*border_map),
                       vmin = 0, vmax = 3, cmap='viridis_r')

            un = list(np.unique([c.genome for c in creatures]))
            un.sort()
            pal = sns.color_palette('magma', len(un))
            cols = [pal[un.index(c.genome)] for c in creatures]
            plt.scatter(x=[c.x for c in creatures],
                        y=[c.y for c in creatures],
                        c=cols, s=1000/min(H, W))
            num = '{0:04d}'.format(movie_index)
            plt.savefig(f'movie/{num}.jpg',
                        dpi=100, bbox_inches='tight')
            movie_index += 1
            plt.cla()


        new_creatures = doStep(creatures)

        # skip to new generation
        if len(new_creatures) == 0:
            if args.verbose:
                print(f'No creatures left after {time_step} steps.')
                repop = True
                break
        else:
            start = max(len(new_creatures) - max_c, 0)
            creatures = new_creatures[start:]

        # logging
        if args.verbose:
            if (1+time_step) % args.every == 0:
                print(f'GEN {generation}, STEP {1+time_step}/{generation_steps}, POP {len(creatures)}')

    # plot data
    if args.plot and generation % args.plot_every == 0:
        log_creatures(plot_data, creatures, 1+(generation//args.plot_every))

    if not args.continuous or repop:

        # create food
        food_map = createFoodMap(H, W, args.food)

        # repopulate for next generation
        parents = np.random.choice(
            np.unique([c.genome for c in creatures]),
            size=initial_pop)

        creatures, pos_map = populate(H, W, initial_pop, initial_energy,
                                      in_len, out_len, hidden_num,
                                      initial_connections, outputs,
                                      mutation_rate, genealogy, parents)


if args.save:
    pickle.dump(creatures, open('checkpoint.pkl', 'wb'))

if args.plot:

    # plot brains
    survivors = [x for x in plot_data if plot_data[x][-1] != 0]
    print(survivors)

    tot = len(survivors)

    if tot != 0:
        cols = int(np.sqrt(tot))
        rows = tot // cols
        rows += tot % cols

        pos = range(1, tot+1)

        fig = plt.figure(0, figsize=[cols*4, rows*3])
        fig.clf()
        for i, s in enumerate(survivors):
            b = Creature(0, 0, 0,
                         in_len, out_len, hidden_num,
                         initial_connections, outputs, s).brain

            ax = fig.add_subplot(rows, cols, pos[i])

            ax.imshow(b, vmin=0, vmax=1, cmap='Greens')
            ax.set_xticks(range(in_len))
            ax.set_xticklabels(inputs, rotation=60, fontsize=5)
            ax.set_yticks(range(out_len))
            ax.set_yticklabels(outputs, fontsize=5)
        plt.tight_layout()
        plt.savefig('genomes.png', dpi=200, bbox_inches='tight')

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    stacked = []
    for g in plot_data:
        plot_data[g] = np.array(plot_data[g])
        stacked.append(100*plot_data[g]/initial_pop)

    col = sns.color_palette('coolwarm', len(stacked))
    random.shuffle(col)
    stacked = np.stack(stacked)
    ax.stackplot(range(stacked.shape[1]), stacked, colors=col)
    ax.set_xticks(range(stacked.shape[1]))
    ax.set_xticklabels(args.plot_every*np.array(range(stacked.shape[1])))
    if not args.continuous:
        ax.set_yticks(range(0, 110, 10))
    ax.grid(True)

    plt.savefig('pop.png', dpi=200, bbox_inches='tight')

if args.movie:
    os.system('ffmpeg -framerate 25 -pattern_type glob -i "movie/*.jpg" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p out.mp4')
    os.system('rm movie -fr')

with open('genealogy.txt', 'w') as f:
    f.write(str(genealogy))
