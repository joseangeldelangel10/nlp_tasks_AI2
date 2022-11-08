f = open("es-en/europarl-v7.es-en.en", "r")

max_iter = 10
i = 1
for line in f:
    print(line)
    if i == max_iter:
        break
    i += 1