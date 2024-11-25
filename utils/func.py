def to_scores(inputs):
    p = inputs
    for i0 in range(p.shape[0]):
        for i1 in range(p.shape[1]):

            integer = int(p[i0][i1])
            decimal = p[i0][i1] % 1

            if decimal >= 0.00 and decimal < 0.25:
                decimal = 0

            elif decimal >= 0.25 and decimal < 0.75:
                decimal = 0.5

            elif decimal >= 0.75 and decimal < 1.00:
                decimal = 1

            p[i0][i1] = integer+decimal

    return p
