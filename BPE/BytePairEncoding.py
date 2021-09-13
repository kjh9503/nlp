import collections ,re

class BPE:
    def __init__(self, dictionary, iter):
        self.dictionary = dictionary
        self.iter = iter
        self.bpe_codes = {}
        self.bpe_codes_reverse = {}

    def get_stats(self):
        pairs = collections.defaultdict(int)
        for word, freq in self.dictionary.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_dictionary(self, pair):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in self.dictionary:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.dictionary[word]
        self.dictionary = v_out

    def merging(self):
        for i in range(self.iter):
            print("### Iteration : {}".format(i))
            pairs = self.get_stats()
            best = max(pairs, key=pairs.get)
            self.merge_dictionary(best)
            self.bpe_codes[best] = i
            self.bpe_codes_reverse[best[0] + best[1]] = best
            print("new merge : ", best)
            print("dictionary : ", self.dictionary)

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pair = (prev_char, char)
            pairs.add(pair)
            prev_char = char
        return pairs

    def encode(self, orig):
        word = tuple(orig) + ('</w>',)
        # word = ('l', 'o', 'k', 'i', '</w>')

        print("__word split into characters : {}".format(word))

        pairs = self.get_pairs(word)

        if not pairs:
            print('not pairs')
            return orig

        iter = 0
        while True:
            iter += 1
            print("__Iteration {} : __".format(iter))

            print("bigrams in the word : {}".format(pairs))

            bigram = min(pairs, key=lambda pair: self.bpe_codes.get(pair, float('inf')))
            """
            bpe_codes : {('e', 's'): 0, ('es', 't'): 1, ('est', '</w>'): 2, ('l', 'o'): 3, ('lo', 'w'): 4, ('n', 'e'): 5, ('ne', 'w'): 6,
            ('new', 'est</w>'): 7, ('low', '</w>'): 8, ('w', 'i'): 9}
            pairs가 위와 같고
            pairs 가 {('/', 'w'), ('w', '>'), ('o', 'k'), ('k', 'i'), ('i', '<'), ('<', '/'), ('l', 'o')}일 때
            pairs 중 최소값을 고르되, bpe_codes에 key가 있으면 bpe codes의 value값, bpe_codes에 key가 없으면 inf 값을 갖게함
            즉 pairs = {('/', 'w') : inf, ('w', '>') : inf, ('o', 'k') : inf, ('k', 'i') : inf, ('i', '<') : inf,\
                        ('<', '/') : inf, ('l', 'o') : 3}
            이 되서 bigram 은 ('l' 'o')가 됨
            """
            print("candidate for merging : {}".format(bigram))

            if bigram not in self.bpe_codes:
                print("__Candidate not in BPE merges, algorithm stops.__")
                break

            first, second = bigram
            # first : 'l', second : 'o'
            new_word = []
            i = 0
            while i < len(word):
                # word = ('l', 'o', 'k', 'i', '</w')
                try:
                    j = word.index(first, i)  # i 번째부터 찾아서 first가 있는 index 반환
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            print(word)
            print("word after merging : {}".format(word))

            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>', ''),)

        return word