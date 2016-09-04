import numpy as np
import sklearn.preprocessing
import numpy

def data_discretization(X, n_bins):
    from sklearn.cluster import KMeans
    clus = KMeans(n_clusters=n_bins).fit(X.flatten()[:, np.newaxis])
    def inverse_transform(x):
        x = np.array(x)
        nb = x.shape[0]
        x = x.flatten()
        c = clus.cluster_centers_[x]
        c = c.reshape((nb, -1))
        return c
    return clus.predict(X.flatten()[:, np.newaxis]).reshape(X.shape), inverse_transform

def generate_text(pred_func, vectorizer, cur=None, nb=1, max_length=10, random_state=None,  way='argmax', temperature=1):
    """
    cur        : cur text to condition on (seed), otherwise initialized by the begin character, shape = (N, T)
    pred_func  : function which predicts the next character based on a set of characters, it takes (N, T) as input and returns (N, D) as output
                 where T is the number of time steps, N size of mini-batch and D size of vocabulary. for each example we thus return
                 a score for each word in the vocabulary, note that this score should not be normalized using softmax (it its the pre-softmax score)
    max_length : nb of characters to generate
    nb : nb of samples to generate (from the same seed)
    """
    assert way in ('proba', 'argmax')
    rng = np.random.RandomState(random_state)
    nb_words = len(vectorizer._word2int)
    if cur is None:
        # initialize the 'seed' with random words
        gen = rng.randint(0, vectorizer._nb_words,
                          size=(nb, vectorizer.length + max_length))
        start = vectorizer.length
    else:
        # initialize the seed by cur
        gen = np.ones((len(cur), cur.shape[1] + max_length))
        start = cur.shape[1]
        gen[:, 0:start] = cur
    gen = intX(gen)
    for i in range(start, start + max_length):
        pr = pred_func(gen, i)
        pr = softmax(pr * temperature)
        next_gen = []
        for word_pr in pr:
            if way == 'argmax':
                word_idx = word_pr.argmax()  # only take argmax
            elif way == 'proba':
                word_idx = rng.choice(np.arange(len(word_pr)), p=word_pr)
            next_gen.append(word_idx)
        gen[:, i] = next_gen
    return vectorizer.inverse_transform(gen[:, start:])

def generate_text_deterministic(pred_func, vectorizer, cur=None, nb=1, max_length=10):
    nb_words = len(vectorizer._word2int)
    if cur is None:
        # initialize the 'seed' with random words
        gen = rng.randint(0, vectorizer._nb_words,
                          size=(nb, vectorizer.length + max_length))
        start = vectorizer.length
    else:
        # initialize the seed by cur
        gen = np.ones((len(cur), cur.shape[1] + max_length))
        start = cur.shape[1]
        gen[:, 0:start] = cur
    gen = categ(gen, D=nb_words)
    gen = floatX(gen)
    for i in range(start, start + max_length):
        gen[:, i] = pred_func(gen, i)
    return vectorizer.inverse_transform(gen[:, start:])

ZERO_CHARACTER = 0
BEGIN_CHARACTER = 1

class DocumentVectorizer(object):

    def __init__(self, length=None, begin_letter=True, pad=True):
        self.length = length
        self.begin_letter = begin_letter
        self.pad = pad

    def fit(self, docs):
        all_words = set(word for doc in docs for word in doc)
        all_words = set(all_words)
        all_words.add(ZERO_CHARACTER)
        all_words.add(BEGIN_CHARACTER)
        self._nb_words = len(all_words)
        self._word2int = {w: i for i, w in enumerate(all_words)}
        self._int2word = {i: w for i, w in enumerate(all_words)}
        return self

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _doc_transform(self, doc):
        doc = map(self._word_transform, doc)
        if self.length:
            if len(doc) >= self.length:
                return doc[0:self.length]
            else:
                doc_new = []
                if self.begin_letter:
                    doc_new.append(self._word_transform(1))
                doc_new.extend(doc)
                if self.pad:
                    remaining = self.length - len(doc_new)
                    doc_new.extend(map(self._word_transform, [0] * remaining))
                return doc_new
        else:
            return doc

    def _word_transform(self, word):
        return self._word2int[word]

    def transform(self, docs):
       docs = map(self._doc_transform, docs)
       if self.length:
           docs = np.array(docs)
       return docs

    def inverse_transform(self, X):
        docs = []
        for s in X:
            docs.append([self._int2word[w] for w in s])
        return docs

def intX(x):
    return np.int32(x)

def floatX(x):
    return np.float32(x)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape = None):
    """ Display an array of rgb images.
    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = numpy.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()
    for i in range(M.shape[0]):
        M[i] -= M[i].flatten().min()
        M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    if shape is None:
        n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]

    im = numpy.array(bordercolor)*numpy.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = numpy.concatenate((
                  numpy.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*numpy.ones((height,border,3),dtype=float)), 1),
                  bordercolor*numpy.ones((border,width+border,3),dtype=float)
                  ), 0)
    return im

def categ(X, D=10):
    nb = np.prod(X.shape)
    x = X.flatten()
    m = np.zeros((nb, D))
    m[np.arange(nb), x] = 1.
    m = m.reshape(X.shape + (D,))
    m = floatX(m)
    return m

