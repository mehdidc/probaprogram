# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('agg')  # NOQA
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from scipy.special import binom

Point = namedtuple('Point', ['x', 'y'])


class Sampler(object):

    def __init__(self, attach_parts=True, nbparts=(1, 5), nbpoints=(1, 5), random_state=42):
        self.nbparts = nbparts
        self.nbpoints = nbpoints
        self.attach_parts = attach_parts
        self.rng = np.random.RandomState(random_state)

    def sample(self):
        nbparts = self.sample_nbparts()
        parts = []
        for i in range(nbparts):
            part = self.sample_part(parts)
            parts.append(part)
        return parts

    def sample_nbparts(self):
        low, high = self.nbparts
        return self.rng.randint(low, high)

    def sample_part(self, parts):
        if len(parts) == 0 or self.attach_parts is False:
            starting_point = self.sample_point([])
        else:
            k = self.rng.randint(0, len(parts))
            starting_point = parts[k][-1]

        nb_points = self.sample_nb_points()

        point = starting_point
        points = [point]
        for i in range(nb_points - 1):
            point = self.sample_point(points)
            points.append(point)
        return points

    def sample_nb_points(self):
        low, high = self.nbpoints
        return self.rng.randint(low, high)

    def sample_point(self, points):
        x, y = self.rng.uniform(size=2)
        return Point(x=x, y=y)


def render(obj, **kwargs):
    curves = []
    for part in obj:
        curve = Bezier(part, **kwargs)
        curves.append(curve)
    if len(curves):
        return np.concatenate(curves, axis=0)
    else:
        return []

# Source : https://gist.github.com/Juanlu001/7284462
def Bernstein(n, k):
    """Bernstein polynomial.
    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, num=400, sx=10, sy=10):
    """
    Build Bézier curve from points.
    """
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(t),
                          np.array([points[ii].x * sx, points[ii].y * sy]))
    return curve


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def linearize(obj, max_parts=5, max_subparts=5):
    v = []
    for part in obj:
        for point in part:
            v.append(point.x)
            v.append(point.y)
        nb = max_subparts - len(part)
        v.extend([-1] * 2 * nb)

    nb = max_parts - len(obj)
    v.extend([-1] * max_subparts * nb * 2)
    return v


def delinearize(v, max_parts=5, max_subparts=5):
    parts = []
    ptr = 0
    for i in range(max_parts):
        if v[ptr] < 0:
            break
        part = []
        for j in range(max_subparts):
            x = v[ptr]
            ptr += 1
            y = v[ptr]
            ptr += 1
            if x < 0 or y < 0:
                ptr += (max_subparts - 1 - j) * 2
                break
            else:
                point = Point(x=x, y=y)
                part.append(point)
        parts.append(part)
    return parts


def optimize(image):
    import cma
    max_parts = 5
    max_subparts = 5
    nb = max_parts * max_subparts * 2

    import optunity
    solvers = optunity.available_solvers()
    print('Available solvers: ' + ', '.join(solvers))

    def encode(params):
        x = [None] * len(params)
        for k, v in params.items():
            x[int(k)] = v
        return x

    def f(x):
        obj = delinearize(x, max_parts=max_parts, max_subparts=max_subparts)
        curve = render(obj)
        predicted_image = to_img(curve)
        fitness = ((image - predicted_image) ** 2).mean()
        # fitness = np.abs(image - predicted_image).sum()
        return fitness

    def g(**params):
        x = encode(params)
        return f(x)

    #params = {str(i): [-1, 1] for i in range(nb)}
    #pars, details, p = optunity.minimize(f, num_evals=1000, solver_name="particle swarm", **params)

    xinit = np.random.uniform(-1, 1, size=nb)
    res = cma.fmin(f, xinit, 0.8, {'maxfevals': 10000})[0]
    obj = delinearize(res, max_parts=max_parts, max_subparts=max_subparts)
    return obj


def genetic_optimize(image, rng=np.random, nb_iter=10, **kw):

    size = 100
    nb = 10
    nb_children = size / 2

    mutation_proba = [
        ('random', 0.1),
        ('add_part', 0.1),
        ('remove_part', 0.01),
        ('add_point', 0.1),
        ('remove_point', 0.01)
    ]

    population = [sample(rng=rng) for i in range(size)]

    def f(x):
        return ((to_img(render(x), **kw) - image) ** 2).sum()

    for i in range(nb_iter):
        print(np.min(map(f, population)))
        population = sorted(population, key=f)
        best = population[0:nb]
        new_population = []
        for k in range(nb_children):
            ia, ib = rng.choice(range(len(best)), size=2, replace=False)
            a = best[ia]
            b = best[ib]
            size_children = (len(a) + len(b)) / 2
            S = size_children / 2
            parts_a = [a[c] for c in rng.choice(range(len(a)), size=S)]
            parts_b = [b[c] for c in rng.choice(range(len(b)), size=size_children - S)]
            children = parts_a + parts_b
            new_population.append(children)
        mutated_new_population = []
        for p in new_population:
            r = rng.uniform()
            proba_cum = 0.
            for name, proba in mutation_proba:
                if proba_cum <= r:
                    if name == "random":
                        p[:] = sample(rng=rng)
                    elif name == "add_part":
                        x = sample(rng=rng)
                        part = x[np.random.choice(len(x))]
                        p.append(part)
                    elif name == "remove_part":
                        if len(p):
                            del p[np.random.choice(len(p))]
                    elif name == "add_point":
                        k = np.random.choice(range(len(p)))
                        point = sample_point([], rng=rng)
                        p[k].append(point)
                    elif name == "remove_point":
                        if len(p):
                            k = np.random.choice(range(len(p)))
                            if len(p[k]):
                                kp = np.random.choice(len(p[k]))
                                del p[k][kp]
                proba_cum += proba
            if len(p):
                mutated_new_population.append(p)
        population = best + mutated_new_population
    return min(population, key=f)


def to_img(curve, w=1, h=1, dpi=28):
    if len(curve):
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        plt.scatter(curve[:, 0], curve[:, 1], c="black")
        plt.axis('off')
        X = fig2data(fig)
        plt.close(fig)
        X = X[:, :, 0]
        X = X.astype(np.float32) / X.max()
        X = 1 - X
        return X
    else:
        X = np.zeros((w*dpi, h*dpi))
        return X

def to_img2(curve, w=1, h=1, thickness=2):
    img = np.zeros((w, h))
    if len(curve) == 0:
        return img
    x_min = curve[:, 1].min()
    x_max = curve[:, 1].max()
    x_max += (x_min==x_max)

    y_min = curve[:, 0].min()
    y_max = curve[:, 0].max()
    y_max += (y_min==y_max)
    for c in curve:
        y, x  = c
        y, x = int(h * (y - y_min) / y_max), int(w * (x - x_min) / x_max)
        img[y-thickness:y+thickness, x-thickness:x+thickness] = 1
    return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nbl = 10
    nbc = 10
    w = 64
    h = 64
    canvas = np.empty((w * nbl, h * nbc))
    sampler = Sampler(attach_parts=True, nbpoints=(2, 5), nbparts=(1, 5))
    for l in range(nbl):
        for c in range(nbc):
            o = sampler.sample()
            img = to_img2(render(o, num=500, sx=64, sy=64), w=w, h=h)
            x = l * w
            y = c * h
            canvas[x:x+w, y:y+h] = img
    plt.imshow(canvas, cmap="gray", interpolation='none')
    plt.savefig('out.png')
