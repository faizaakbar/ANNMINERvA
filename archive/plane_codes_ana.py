import plane_codes

vpd = plane_codes.get_views()
ric = plane_codes.build_reversed_indexed_codes()

vpd_t = [tuple([it, vpd[it]]) for it in vpd]
ric_t = [tuple([it, ric[it]]) for it in ric]
ric_t = ric_t[1:]

for i in range(len(ric_t)):
    if ric_t[i][1][-1] != 0:
        vpd_t.insert(i, 'Target {}'.format(ric_t[i][1][-1]))

compiled = list(zip(vpd_t[:50], ric_t[:50]))

n_x = 0
targs = [1, 2, 3, 'water', 4, 5, None]
targ_idx = 0
targ = targs[targ_idx]
for x in compiled:
    c = x[0][1]
    if c.islower():
        print('{} xs before target {}'.format(n_x, targ))
        targ_idx += 1
        targ = targs[targ_idx]
        # n_x = 0
    elif c == 'X':
        n_x += 1
