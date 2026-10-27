# Primary futlyar v1: one given cover, rational-center strict interior

## Source and scope

Local source `.sources/kozlov2025_three_approximations.txt`, `===== page 10 =====` (journal p155), defines the primary skeleton as centers of independent pieces of a complete image. **R is the least distance between distinct centers, not half that distance.** Points of each piece must be closer to its own center than to every other center; overlap is handled by the resulting truncated circles. The source rejects a cover when R is less than half its size. `docs/open_questions.md` Q4 records that the half-size screen alone is insufficient and that m=1, coincident centers and the primary circle boundary remain unresolved.

This implementation decides one supplied cover of 1..7 labelled rational points. It performs no cover search, general futlyar construction, generalized-radius selection, fits optimization, granularity, complete-group or semantic-image decision. TRUE establishes the supported strict-interior primary conditions for that exact cover.

## API and results

`assess_primary_futlyar(points, parts, *, max_work=2000000, seconds=30)` returns schema `primary-futlyar-v1` and one of:

- `TRUE`: all input points are covered by independent nonempty parts, every block has an independently verified complete-input optimal ellipse/point/segment center certificate, and all primary strict-interior conditions hold.
- `FALSE`: a definite condition fails for this supplied cover. This does not disprove another cover or a futlyar of the image.
- `UNKNOWN`: shared work/time, intermediate bits, or exact-center construction/verification is incomplete.
- `UNSUPPORTED`: m=1, coincident centers, primary circle equality, or a block center/size expression is outside the rational normalization domain.
- `INVALID_INPUT`: malformed/incomplete input, prohibited scalar types or an invalid supplied identity.

Coordinates accept exactly int/Fraction, excluding bool and float; initial coordinate bits are bounded by existing independent-cover normalization. The rational intermediate bound is 16384 bits. Points must be distinct, labels unique integers, and parts nonempty, disjoint and complete. The complete image contains at most seven points. Parts retain their supplied order and labels within each part are sorted.

`assess_primary_futlyar_certificate(points, certificate, *, max_work=2000000, seconds=30)` independently returns `VERIFIED`, `INVALID` or `UNKNOWN`, with work_used. `verify_primary_futlyar_certificate` is its boolean convenience wrapper. The verifier calls the independent cover and exact-center certificate verifiers, never cover enumeration, a center solver, a hull constructor or an ellipse constructor.

## Exact arithmetic and criterion

Every center comes from `solve_exact_center` followed by `assess_exact_center_result`. Point/segment and triangle/contact3/contact5 rational profiles are supported. A contact4 algebraic profile is supported only when AN normalization proves every center coordinate and both length trace/discriminant expressions constant rational values. A nonconstant normalized expression is explicitly UNSUPPORTED; no conversion to float occurs. This conservative gate does not claim that every mathematically rational expression must normalize to a constant.

Let centers be c_i and let r2 = min_{i<j} ||c_i-c_j||². Require r2 > 0 and m >= 2. Each label p in part i must satisfy ||p-c_i||² < ||p-c_j||² for all j != i, and ||p-c_i||² < r2. The first condition checks truncated-circle membership directly without a geometric clipping approximation. Equality in the latter condition is UNSUPPORTED unless another definite condition already proves FALSE.

Every piece size must satisfy R >= size_i/2; this is equivalent to the cover maximum-size screen. For a segment/point, compare 4*r2 >= length². For an ellipse with full major-axis length sqrt(2*(trace+sqrt(discriminant))), define u=2*r2-trace and compare u>=0 and u²>=discriminant. This is an exact Fraction comparison; it handles half-size equality without an epsilon. This screen does not replace independence, complete coverage, nearest-center or circle inclusion.

## Certificate and trust boundary

Successful `certificate` has exactly fields schema, status, parts, declared_labels, source_coordinate_sha256, cover_certificate, blocks, centers, radius_squared, size_descriptors, radius_screens, point_checks. Schema is `primary-futlyar-certificate-v1`, status TRUE. `blocks` is `[{'labels': [...], 'result': exact_center_result}, ...]`. Each result contains its full original optimum certificate and number context. Centers are tuples of Fraction, radius_squared is Fraction. Point checks record label, owner index, all center distances squared, uniquely_nearest and strictly_inside. These are diagnostics, not trusted cached judgments.

The independent verifier checks full source binding, every cover/block certificate, derives centers and size formulas, recalculates R², half-size screens and every complete-input distance, and compares all stored derived fields. Numeric certificate fields reject bool/float pollution; diagnostic boolean fields require bool. Extra primary certificate fields are rejected. In-memory certificates contain Fraction and tuples; callers needing JSON can use the existing exact_center encode_exact/decode_exact identity-preserving tags.

## Shared bounded work

One parent Budget tracks work/time. Children receive only remaining work and seconds; their reported work is charged back to the parent. Root/AN work receives that same Budget. Fixed charges before bounded legacy cover identity checks are cooperative work accounting, not CPU-instruction counts. Verifier UNKNOWN is preserved as UNKNOWN by the producer. Exhaustion never becomes FALSE, an empty root list or an accepted partial certificate.
