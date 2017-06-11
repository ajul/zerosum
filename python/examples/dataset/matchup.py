import numpy

class MatchupDataset():
    def __init__(self, data, names):
        self.data = self.canonicalize(data)
        self.names = names

    # normalizes data to have a mean of 0.5
    def canonicalize(self, data):
        data = numpy.array(data, dtype='float64')
        if numpy.min(data) >= 0.0:
            data *= 0.5 / numpy.mean(data)
        return data
        
    def sorted_by_sum(self):
        idx = numpy.argsort(self.data.sum(axis = 0))
        newData = self.data[idx, :][:, idx]
        newNames = list(self.names[i] for i in idx)
        return MatchupDataset(newData, newNames)
        
    def to_csv(self):
        result = ',' + ','.join(self.names) + '\n'
        for i in range(self.data.shape[0]):
            result += self.names[i] + ','
            for j in range(self.data.shape[1]):
                result += ('%f' % self.data[i, j]) + ','
            result += '\n'
        return result

# Super Smash Bros. 64. Source: https://www.ssbwiki.com/Character_matchup_(SSB)
ssb64 = MatchupDataset(
    [
        [50, 60, 55, 65, 65, 65, 60, 70, 65, 70, 75, 70, ],
        [40, 50, 60, 60, 60, 60, 50, 60, 60, 60, 60, 60, ],
        [45, 40, 50, 55, 60, 50, 50, 60, 60, 60, 60, 60, ],
        [35, 40, 45, 50, 55, 70, 50, 70, 60, 65, 70, 60, ],
        [35, 40, 40, 45, 50, 50, 50, 60, 60, 50, 60, 50, ],
        [35, 40, 50, 30, 50, 50, 40, 50, 40, 50, 50, 40, ],
        [40, 50, 50, 50, 50, 60, 50, 60, 60, 60, 60, 60, ],
        [30, 40, 40, 30, 40, 50, 40, 50, 40, 40, 40, 40, ],
        [35, 40, 40, 40, 40, 60, 40, 60, 50, 60, 50, 50, ],
        [30, 40, 40, 35, 50, 50, 40, 60, 40, 50, 50, 60, ],
        [25, 40, 40, 30, 40, 50, 40, 60, 50, 50, 50, 60, ],
        [30, 40, 40, 40, 50, 60, 40, 60, 50, 40, 40, 50, ],
    ],
    ['Pikachu', 'Kirby', 'Captain Falcon', 'Fox',
     'Yoshi', 'Jigglypuff', 'Mario', 'Samus',
     'Donkey Kong', 'Ness', 'Link', 'Luigi'])

# Super Street Fighter 2 Turbo. Source: http://curryallergy.blogspot.com/2008/11/super-turbo-new-arcadia-diagram.html
ssf2t = MatchupDataset(
    [
        [5.0, 6.5, 8.0, 6.0, 7.5, 4.5, 6.5, 3.0, 6.0, 7.5, 6.5, 7.0, 4.0, 4.0, 6.5, 6.5],
        [3.5, 5.0, 7.0, 3.5, 5.0, 4.0, 3.5, 2.5, 5.5, 5.0, 4.5, 5.0, 3.5, 4.0, 5.0, 6.0],
        [2.0, 3.0, 5.0, 2.5, 7.5, 8.5, 2.0, 3.5, 6.5, 8.0, 7.0, 1.5, 3.5, 4.0, 3.5, 6.5],
        [4.0, 6.5, 7.5, 5.0, 7.0, 7.5, 6.0, 5.0, 5.5, 6.0, 4.0, 6.5, 5.5, 4.0, 6.5, 7.0],
        [2.5, 5.0, 2.5, 3.0, 5.0, 6.0, 4.0, 2.0, 7.0, 4.0, 3.5, 2.5, 2.5, 3.5, 2.0, 6.5],
        [5.5, 6.0, 1.5, 2.5, 4.0, 5.0, 2.0, 2.5, 5.5, 3.5, 3.0, 5.5, 4.5, 3.5, 4.0, 5.5],
        [3.5, 6.5, 8.0, 4.0, 6.0, 8.0, 5.0, 1.5, 6.5, 8.0, 7.5, 5.5, 4.0, 3.0, 6.5, 5.5],
        [7.0, 7.5, 6.5, 5.0, 8.0, 7.5, 8.5, 5.0, 6.5, 6.5, 6.5, 8.0, 5.5, 3.5, 9.0, 6.5],
        [4.0, 4.5, 3.5, 4.5, 3.0, 4.5, 3.5, 3.5, 5.0, 1.5, 3.0, 3.0, 4.5, 3.0, 3.5, 3.0],
        [2.5, 5.0, 2.0, 4.0, 6.0, 6.5, 2.0, 3.5, 8.5, 5.0, 3.0, 2.0, 2.5, 4.0, 3.0, 2.0],
        [3.5, 5.5, 3.0, 6.0, 6.5, 7.0, 2.5, 3.5, 7.0, 7.0, 5.0, 2.5, 3.0, 3.5, 3.5, 3.5],
        [3.0, 5.0, 8.5, 3.5, 7.5, 4.5, 4.5, 2.0, 7.0, 8.0, 7.5, 5.0, 3.0, 4.0, 5.5, 6.5],
        [6.0, 6.5, 6.5, 4.5, 7.5, 5.5, 6.0, 4.5, 5.5, 7.5, 7.0, 7.0, 5.0, 5.5, 8.0, 6.5],
        [6.0, 6.0, 6.0, 6.0, 6.5, 6.5, 7.0, 6.5, 7.0, 6.0, 6.5, 6.0, 4.5, 5.0, 7.5, 5.0],
        [3.5, 5.0, 6.5, 3.5, 8.0, 6.0, 3.5, 1.0, 6.5, 7.0, 6.5, 4.5, 2.0, 2.5, 5.0, 3.5],
        [3.5, 4.0, 3.5, 3.0, 3.5, 4.5, 4.5, 3.5, 7.0, 8.0, 6.5, 3.5, 3.5, 5.0, 6.5, 5.0],
    ],
    ['Ryu', 'Ken', 'E. Honda', 'Chun-Li',
     'Blanka', 'Zangief', 'Guile', 'Dhalsim',
     'T. Hawk', 'Cammy', 'Fei-Long', 'Dee Jay',
     'Boxer', 'Claw', 'Sagat', 'Dictator'])
