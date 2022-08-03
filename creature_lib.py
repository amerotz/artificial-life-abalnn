import numpy as np

occupied = 1
border = 1
food = 1
ground = 0


def createFoodMap(H, W, food_p):

    # random food & ground
    grid_map = np.random.choice([food, ground], size=(H, W), p=[food_p, 1-food_p])

    return grid_map

def createPosMap(H, W, creatures):

    pos_map = np.zeros((H, W), dtype=int)

    for c in creatures:
        pos_map[c.y, c.x] = 1

    return pos_map

def createBorderMap(H, W, b_type):

    grid_map = np.zeros((H, W))

    if b_type == 'box':
        grid_map[0, :] = border
        grid_map[:, 0] = border
        grid_map[H-1, :] = border
        grid_map[:, W-1] = border

    return grid_map

def populate(H, W, pop, initial_energy,
             in_len, out_len, hidden_num,
             initial_connections, outputs,
             mutation_rate, genealogy, parents):

    pos_map = np.zeros((H, W), dtype=int)
    creatures = []

    for i in range(pop):

        # generate position
        x = 0
        y = 0
        occupied = 1
        while occupied == 1:
            x = 1+np.random.randint(W-2)
            y = 1+np.random.randint(H-2)
            occupied = pos_map[y, x]

        pos_map[y, x] = 1

        genome = ''
        if parents is not None:
            par = parents[i]
            genome = Creature.mutate(par, initial_connections,
                                     in_len, out_len, mutation_rate)

            if genealogy is not None:
                updateGenealogy(genealogy, par, genome)

        creatures.append(
            Creature(x, y,
                     initial_energy,
                     in_len, out_len, hidden_num,
                     initial_connections, outputs, genome)
        )

    return creatures, pos_map

def updateGenealogy(genealogy, parent, child):
    if parent == child:
        return
    if not parent in genealogy:
        genealogy[parent] = []
    if not child in genealogy[parent]:
        genealogy[parent].append(child)
    if not child in genealogy:
        genealogy[child] = []


def checkSeen(el, array):
    for a in array:
        diff = el - a
        norm = np.linalg.norm(diff, 2)
        if norm < 0.001:
            return True
        return False

class Creature:

    def __init__(self, x, y, energy,
                 in_len, out_len, hidden_num,
                 initial_connections, outputs,
                 genome=''):

        self.in_len = in_len
        self.out_len = out_len
        self.hidden_num = hidden_num
        self.initial_connections = initial_connections
        self.outputs = outputs

        self.x = x
        self.y = y
        self.energy = energy
        self.age = 0
        self.brain = np.zeros((self.out_len, self.in_len))
        self.dead = False

        if genome == '':
            self.genome = self.random_genome()
        else:
            self.genome = genome

        for gene in self.genome.split(' '):
            s = int(gene[:2], 16)
            d = int(gene[2:4], 16)
            v = int(gene[4:], 16)

            if d < self.out_len and s < self.in_len:
                self.brain[d][s] = v/65536

    def older(self):
        self.age += 1

    # create a random genome
    def random_genome(self):

        src = np.random.randint(low=0,
                                high=self.in_len,
                                size=self.initial_connections)
        dst = np.random.randint(low=0,
                                high=self.out_len,
                                size=self.initial_connections)

        vals = np.random.randint(low=0,
                                 high=np.iinfo(np.uint16).max,
                                 size=self.initial_connections)

        genome = [
            f"{'{0:02x}'.format(s)}{'{0:02x}'.format(d)}{'{0:04x}'.format(v)}"
            for s, d, v in zip(src, dst, vals)
        ]

        return ' '.join(genome)

    # elanorate inputs
    def think(self, sensors):

        h_state = np.zeros(self.hidden_num)
        sensory_ins = np.concatenate(([self.energy, self.age], sensors, h_state))

        res = np.zeros(self.out_len-self.hidden_num)

        seen = [np.zeros(self.in_len)]

        in_size = self.in_len-self.hidden_num

        already_seen = checkSeen(sensory_ins, seen)

        count = 0
        stop = False
        while not stop:
            seen.append(sensory_ins)

            outs = np.tanh(self.brain.dot(sensory_ins))

            res += outs[self.hidden_num:]

            sensory_ins = np.zeros(self.in_len)
            sensory_ins[in_size:] = outs[:self.hidden_num]

            already_seen = checkSeen(sensory_ins, seen)
            count += 1

            stop = count == self.initial_connections or already_seen

        res = np.tanh(res)

        actions = {
            self.outputs[i]:res[i-self.hidden_num]
            for i in range(self.hidden_num, self.out_len) if res[i-self.hidden_num] != 0
        }

        return actions


    def consumeEnergy(self, energy):
        self.energy -= energy
        if self.energy <= 0:
            self.die()


    # change position
    def move(self, x, y):
        self.x = x
        self.y = y

    # add energy
    def eat(self, food_en):
        self.energy += food_en

    # create a child with a mutated genome
    def reproduce(self, x, y, mutation_rate):
        new_genome = self.mutate(self.genome, self.initial_connections,
                                 self.in_len, self.out_len, mutation_rate)


        child = Creature(x, y, self.energy/2,
                         self.in_len, self.out_len, self.hidden_num,
                         self.initial_connections, self.outputs,
                         new_genome)

        return child

    # mutate 1 bit of the genome
    @staticmethod
    def mutate(genome, initial_connections, in_len, out_len, mutation_rate):

        tmp = np.random.rand()

        if tmp < mutation_rate:

            genome = ''.join(genome.split(' '))
            genome = np.binary_repr(
                int(genome, 16), width=32*initial_connections
            )

            genome = [c for c in genome]
            index = np.random.randint(len(genome))
            while not (index % 32 >= 16 or
                       (index % 16 >= 7 - np.log2(in_len) and index < 8) or
                       (index % 16 >= 15 - np.log2(out_len) and index < 16)):
                index = np.random.randint(len(genome))

            genome[index] = str(1-int(genome[index]))

            genome = ''.join(genome)
            genome = [genome[i*32:(i+1)*32] for i in range(initial_connections)]

            new_genome = []
            for gene in genome:
                s = int(gene[:8], 2)
                d = int(gene[8:16], 2)
                v = int(gene[16:], 2)

                new_genome.append(
                    f"{'{0:02x}'.format(s)}{'{0:02x}'.format(d)}{'{0:04x}'.format(v)}")

            genome = ' '.join(new_genome)

        return genome

    def die(self):
        self.dead = True


