# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import collections
import itertools
import math
import random


def physical_index(r, s, c, columns):
    return (r // 2) * 8 * columns + (r & 1) + 2 * (s * columns + c)


class SFPU:
    """Four value/index registers, each split into four subvectors."""

    def __init__(self, values, columns=8):
        self.columns = columns
        # Physical input order is LReg15 + {0, 1, 64, 65}.
        self.reg = [
            [[(values[physical_index(r, s, c, columns)],
               physical_index(r, s, c, columns))
              for c in range(columns)] for s in range(4)] for r in range(4)
        ]
        self.ops = collections.Counter()
        self.sequence = []

    def swap(self, a, b, max_subvectors=(0, 1, 2, 3)):
        """SFPSWAP, with register a getting max in the listed subvectors."""
        self.ops['SFPSWAP'] += 1
        self.sequence.append(('swap', a, b, tuple(max_subvectors)))
        for s in range(4):
            for c in range(self.columns):
                av, bv = self.reg[a][s][c], self.reg[b][s][c]
                # Reproduce SFPSWAP's positive/negative equal-value asymmetry.
                # VC=a, VD=b. Modes 1/2/3 put min in b in the chosen subvectors.
                smaller = av[0] < bv[0] or (av[0] == bv[0] and math.copysign(1.0, av[0]) < 0)
                exchange = smaller if s in max_subvectors else not smaller
                if exchange:
                    self.reg[a][s][c], self.reg[b][s][c] = bv, av

    def transpose(self):
        self.ops['SFPTRANSP'] += 1
        self.sequence.append(('transpose',))
        self.reg = [[self.reg[s][r] for s in range(4)] for r in range(4)]

    def lists(self):
        return [[self.reg[r][0][c] for r in range(4)] for c in range(self.columns)]


def sort4(m, direction):
    # Five-comparator sorting network. Direction can vary by subvector.
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
        m.swap(a, b, direction)


def bitonic4(m, direction):
    for a, b in ((0, 2), (1, 3), (0, 1), (2, 3)):
        m.swap(a, b, direction)


def top4_lists(values, columns=8, sorted_output=True):
    m = SFPU(values, columns)
    sort4(m, (0, 2))             # Four-element lists: descending/ascending/D/A.
    m.transpose()
    m.swap(0, 1)                 # Retain top four of the first eight.
    m.swap(2, 3)                 # Retain top four of the second eight.
    m.transpose()
    bitonic4(m, (0, 1))          # Retained s0 descending, retained s2 ascending.
    m.transpose()
    m.swap(0, 2)                 # Retain top four of sixteen.
    m.transpose()
    if sorted_output:
        bitonic4(m, (0, 1, 2, 3))
    return m


def full_sort16(values, columns=8):
    m = SFPU(values, columns)
    # Standard bitonic network, q = r + 4*s.
    m.swap(0, 1)
    m.swap(2, 3, ())
    bitonic4(m, (0, 2))
    m.transpose()
    m.swap(0, 1)
    m.swap(2, 3, ())
    m.transpose()
    bitonic4(m, (0, 1))
    m.transpose()
    for a, b in ((0, 2), (1, 3), (0, 1), (2, 3)):
        m.swap(a, b)
    m.transpose()
    bitonic4(m, (0, 1, 2, 3))
    return m


def top4_32(values, garbage=None):
    assert len(values) == 32
    m = SFPU(values + (garbage if garbage is not None else [100000] * 96))
    m.swap(0, 1, (0, 2))
    m.transpose()
    m.swap(0, 1)
    m.transpose()
    m.swap(0, 1)
    lists = [[m.reg[r][s][c] for s in range(2) for r in range(2)] for c in range(8)]
    return m, lists


def check32(values, garbage=None):
    m, lists = top4_32(values, garbage)
    for c, result in enumerate(lists):
        expected = sorted([values[physical_index(r, s, c, 8)]
                           for s in range(2) for r in range(2)], reverse=True)
        assert [v for v, _ in result] == expected
        assert all(i < 32 and values[i] == v for v, i in result)
    candidates = sorted((x for row in lists for x in row), reverse=True)
    assert [v for v, _ in candidates[:4]] == sorted(values, reverse=True)[:4]


def check(values, columns, exhaustive=False):
    for sorted_output in (False, True):
        m = top4_lists(values, columns, sorted_output)
        lists = m.lists()
        for c, result in enumerate(lists):
            source = [values[physical_index(r, s, c, columns)]
                      for r in range(4) for s in range(4)]
            expected = sorted(source, reverse=True)[:4]
            actual = [v for v, _ in result]
            assert (actual if sorted_output else sorted(actual, reverse=True)) == expected
            assert len({i for _, i in result}) == 4
            assert all(values[i] == v for v, i in result)
        if sorted_output:
            heads = [0] * columns
            winners = []
            for _ in range(4):
                c = max(range(columns), key=lambda c: lists[c][heads[c]][0]
                        if heads[c] < 4 else -float('inf'))
                winners.append(lists[c][heads[c]])
                heads[c] += 1
            assert [v for v, _ in winners] == sorted(values, reverse=True)[:4]
            assert len({i for _, i in winners}) == 4
    if exhaustive:
        m = full_sort16(values, columns)
        for c in range(columns):
            actual = [m.reg[r][s][c][0] for s in range(4) for r in range(4)]
            source = [values[physical_index(r, s, c, columns)]
                      for r in range(4) for s in range(4)]
            assert actual == sorted(source, reverse=True)


def main():
    for bits in itertools.product((0, 1), repeat=16):
        check(bits, 1, True)
    print('PASS: all 65,536 binary 16-element groups, sorted/pruned/full-sort networks')
    rng = random.Random(20260913)
    for _ in range(2000):
        values = list(range(-64, 64))
        rng.shuffle(values)
        check(values, 8)
        check([rng.randrange(-4, 5) for _ in range(128)], 8)
    print('PASS: 4,000 signed/tied 128-element cases, index integrity and scalar eight-list merge')
    for bits in itertools.product((0, 1), repeat=4):
        values = [bits[r + 2 * s] for s in range(2) for c in range(8) for r in range(2)]
        check32(values)
    for _ in range(4000):
        values = [rng.randrange(-64, 64) for _ in range(32)]
        garbage = [rng.randrange(-1000000, 1000000) for _ in range(96)]
        check32(values, garbage)
    print('PASS: 32-element network: all binary four-element groups and 4,000 random cases with hostile unused lanes')
    for _ in range(100):
        values = [rng.choice((-float('inf'), -1.0, -0.0, 0.0, 1.0, float('inf'))) for _ in range(128)]
        check(values, 8)
        check32(values[:32], values[32:])
    print('PASS: signed-zero/infinity and duplicate-value permutation checks (NaN policy not specified)')
    for name, m in (
        ('full sort16', full_sort16(list(range(128)))),
        ('top4 lists', top4_lists(list(range(128)))),
        ('top4 unordered', top4_lists(list(range(128)), sorted_output=False)),
        ('32 -> eight sorted lists', top4_32(list(range(32)))[0]),
    ):
        cycles = 2 * m.ops['SFPSWAP'] + m.ops['SFPTRANSP']
        print(f'{name}: {dict(m.ops)}, core SFPU cycles={cycles}, '
              f'without explicit swap NOPs={cycles + m.ops["SFPSWAP"]}')
    prune_check()


def prune_check():
    # Bit-sliced execution of every binary 16-element input simultaneously.
    inputs = []
    for i in range(16):
        if i < 3:
            pattern = bytes((0xAA, 0xCC, 0xF0)[i:i + 1])
        else:
            n = 1 << (i - 3)
            pattern = bytes(n) + bytes([255]) * n
        inputs.append(int.from_bytes(pattern * (8192 // len(pattern)), 'little'))
    expected = [(1 << 65536) - 1, 0, 0, 0, 0]
    for x in inputs:
        for k in range(4, 0, -1):
            expected[k] |= expected[k - 1] & x

    def valid(sequence):
        reg = [[inputs[physical_index(r, s, 0, 1)] for s in range(4)] for r in range(4)]
        for op in sequence:
            if op[0] == 'transpose':
                reg = [list(x) for x in zip(*reg)]
            else:
                _, a, b, direction = op
                for s in range(4):
                    high, low = reg[a][s] | reg[b][s], reg[a][s] & reg[b][s]
                    reg[a][s], reg[b][s] = (high, low) if s in direction else (low, high)
        return all(reg[r][0] == expected[r + 1] for r in range(4))

    seq = top4_lists(list(range(16)), 1).sequence
    assert valid(seq)
    removable = [i for i in range(len(seq)) if valid(seq[:i] + seq[i + 1:])]
    print(f'Bit-sliced exhaustive pruning: individually removable operations = {removable}')
    assert not removable


if __name__ == '__main__':
    main()
