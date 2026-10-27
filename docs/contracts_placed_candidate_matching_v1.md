# Placed candidate acceptance and bottleneck matching

Source: S-2009 journal pp106-107 (PDF12-13), following Theorem7;
S-2002 pp261-262. The input is an **already placed** pair of images.
No orientation, translation, reflection, similarity or affine search is done.

`candidate_acceptance_exact(source,target,radius_squared)` builds the complete
bipartite graph of exact squared distances <= the given rational threshold.
Ordinary acceptance requires a neighbor for every vertex on both sides; unequal
cardinalities are allowed. Narrow acceptance requires a complete bijection.
Positive certificates contain a perfect matching; negative certificates contain
an exact Hall-deficient subset with its COMPLETE neighbor set, or the unequal
cardinalities. Negative narrow acceptance does not deny ordinary acceptance or
the existence of a better placement. The source p106 names B' instead of A once
in the Q_b definition; p107 explicitly identifies A, agreeing with the symmetric
neighborhood relation used here. This was checked visually in both PDF pages.

`placed_bottleneck_exact` minimizes the largest paired squared distance over
all bijections for equal-sized inputs at this placement. It searches the finite
sorted set of all edge lengths using augmenting-path matching. Its certificate
has a perfect matching at the optimal threshold and a Hall obstruction at the
immediately preceding distinct threshold. If the optimum is the first edge
length, the minimum edge itself is the lower bound. The independent verifier
recomputes all distances and these bounds without calling any matching search.
This solver/dual certificate is the project's implementation of the author's
finite acceptance condition, not an assertion that the paper specifies this
particular matching algorithm. Neither result is the full image distance R(A,B).

Inputs: complete nonempty labelled dictionaries, 1..32 distinct rational 2D
points per side; int64 labels; int/Fraction coordinates and threshold up to
256 bits, bool/float rejected. Reuse of the fixed-orientation module's parser
and cooperative budget does not invoke its optimization. Certificates bind
all inputs, exact edges, threshold, predicates and witness. Stored numbers have
a 16384-bit bound and containers/depth are bounded. Work and time are shared
across construction and internal verification (default2M/30s). Budget exhaustion
is UNKNOWN with no success certificate; invalid construction inputs are
UNSUPPORTED; invalid certificates are INVALID. Exact JSON uses the existing
lossless Fraction codec.
