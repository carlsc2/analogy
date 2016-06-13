from analogy import AIMind
from pprint import pprint

a1 = AIMind("data files/large_unrestricted_techdata.xml")
##pprint(a1.find_best_analogy("Google",a1))

#a1 = AIMind("data files/plang_small.xml")

#a1 = AIMind("data files/techdata.xml")
#a2 = AIMind("data files/big_music.xml")
a2 = AIMind("data files/music_small.xml")
#pprint(a1.find_best_analogy("C (programming language)",a2))

#pprint(a2.find_best_analogy("Rock music",a1))


#a1.find_optimal_matchups("Google",a2)

a1.find_optimal_matchups(a2)

#tmp = [a1.find_best_analogy(f,a2) for f in a1.features]
#pprint(sorted(tmp,key=lambda x:[0]))
