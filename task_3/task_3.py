f = open("es-en/europarl-v7.es-en.en", "r")
maximum_iter = 10
i = 1
for line in f:
    print(line)
    if i == maximum_iter:
        break
    i += 1