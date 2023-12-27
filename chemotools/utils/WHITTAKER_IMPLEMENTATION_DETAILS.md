# 🧮 Implementation of the Whittaker Henderson smoothing algorithm

The internal linear system to be solved for the Whittaker Henderson smoothing algorithm
is given by the following set of equations:<br>

$\mathbf{z}=(\mathbf{W}+\tau\cdot\mathbf{D}_{m}^{T}\mathbf{D}_{m})^{-1}\mathbf{W}\mathbf{y}$

where $\mathbf{z}$ is the vector of smoothed values, $\mathbf{W}$ is the diagonal matrix
of weights, $\tau$ is the smoothing parameter, $\mathbf{D}_{m}$ is the ${m}$-th order
forward finite difference matrix, and $\mathbf{y}$ is the vector of observed values.
The matrix $P=\mathbf{D}_{m}^{T}\mathbf{D}_{m}$ is often referred to as the
*penalty matrix*.<br><br>
The matrix to invert is symmetric, sparse, banded with ${2\cdot m + 1}$ non-zero
diagonals, and positive definite, i.e., all its eigenvalues are strictly positive
($>0$). From an algorithmic point of view, this means that the inversion can be performed
in ${\mathcal{O}\left(n\right)}$ time, where ${n}$ is the number of observations, by
using a banded Cholesky decomposition:<br>

$\mathbf{W}+\tau\cdot\mathbf{P}=\mathbf{L}\mathbf{L}^T$

where $\mathbf{L}$ is a lower triangular matrix which shares the same sparsity pattern
as $\mathbf{W}+\tau\cdot\mathbf{P}$. Inversion of a lower triangular matrix is trivial
when backward and forward substitution are used.

## ⚠️ Problem

However, all this is only true from a mathematical point of view. In practice,
floating point arithmetics introduce numerical errors which can lead to an indefinite
matrix. In this case, the Cholesky decomposition fails and the algorithm cannot be used.
This happens for relatively small $n$ already when $m$ exceeds 4, but in spectroscopy
$n > 1000$ is not uncommon and $m = 6$ has been shown helpful in deriving an additional
set of weights fro $\tau$ to make the smooth spatially adaptive.<br>
Besides, there is another problem. The penalty matrix
$\mathbf{P}$ alone is only positive semi-definite since it possesses $m$ zero
eigenvalues by design. From a mathematical perspective, this would not pose a problem
since $\mathbf{W}+\tau\cdot\mathbf{P}$ would still be positive definite as $\tau$ tends
to $+\infty$. Numerically, this is by far not the case because $\tau$-values that are an
order of $10^{16}$ greater than the order of the weights are already sufficient to make
the matrix positive semi-definite or even indefinite if some of the small eigenvalues go
negative in the calculations (for 64-bit float precision).<br>
On the other hand, as $\tau$ tends to $0$, the matrix can also become ill-conditioned
as well if some weights are numerically zero due to zero division.<br><br>
All in all, the banded Cholesky decomposition is not a robust algorithm for solving the
system of linear equations and even pivoted LU-decomposition suffers from the same
problems - even though it can withstand a few more orders of magnitude in $\tau$.

## 💡 Solution

One way out of this dilemma is to make the matrix positive definite by adding a small
positive constant to the main diagonal:

$\mathbf{W}+\tau\cdot\mathbf{P}+\epsilon\cdot\mathbf{I}$

where $\epsilon$ is a small positive constant and $\mathbf{I}$ is the identity matrix.
Despite its simplicity, this approach requires that $\epsilon$ is determined at runtime
which can be costly because it depends on the eigenvalues of $\mathbf{W}$, the
eigenvalues of $\mathbf{P}$, as well as $\tau$.<br>
Therefore, both $\mathbf{W}$ and $\mathbf{P}$ are made positive
definite by adding a small positive constant to their main diagonal before the
decomposition is performed:

$\mathbf{W}+\epsilon_{w}\cdot\mathbf{I}+\tau\cdot\left(\mathbf{P}+\epsilon_{p}\cdot\mathbf{I}\right)=\mathbf{L}\mathbf{L}^T$

Now, the only thing that remains to be done is to determine $\epsilon_{w}$ and
$\epsilon_{p}$ at runtime in an efficient manner that does not require the calculation
of any of the eigenvalues because this would be too costly. On top of that, if
approximations are used, they need to be as close as possible to the actual values,
because too large values of $\epsilon_{w}$ and $\epsilon_{p}$ can obscure the smoothing
effect while too small values can lead to numerical instabilities. A typically applied
way of scaling looks like

$\epsilon_{a}=\varepsilon\cdot n\cdot\lambda_{max}\left(\mathbf{A}\right)$

where $\varepsilon$ is the floating point machine imprecision, $n$ is the number of
observations, and $\lambda_{max}\left(\mathbf{A}\right)$ is the largest eigenvalue of
the matrix $\mathbf{A}$ in question.<br>
This scaling is used, e.g., in ``numpy.linalg.lstsq`` where singular values that are
numerically zero need to be removed (it is used as a threshold there).

### 🏋️ Determination of the weight $\epsilon_{w}$

The largest eigenvalue of $\mathbf{W}$ is given by the largest weight since it is a
diagonal matrix. Therefore, $\epsilon_{w}$ can be determined by

$\epsilon_{w}=\varepsilon\cdot n\cdot\max\left(diag\left(\mathbf{W}\right)\right)$

where $diag\left(\mathbf{W}\right)$ is the vector of diagonal elements of $\mathbf{W}$
and $max$ extracts the maximum value. This is trivial and efficient to calculate.

### ☄️ Determination of the penalty $\epsilon_{p}$

Finding the largest eigenvalue of $\mathbf{D}_{m}^{T}\mathbf{D}_{m}$ is more
complicated. However, some simulations have shown that the limit value of the largest
eigenvalue is given by

$\lim_{n \to \infty} \lambda_{max}\left(\mathbf{D}_{m}^{T}\mathbf{D}_{m}\right)=4^{m}$

which appears to be a strict upper limit and thus perfectly suited for the scaling
factor

$\epsilon_{p}=\varepsilon\cdot n\cdot 4^{m}$

Such an approximation is also cheap to compute, thereby making the algorithm both
efficient and robust.<br>
❗❗❗<br>
Due to the power of $m$, this approximation scales badly with increasing $m$ and
$n$. It is therefore recommended not to use $m > 6$. Probably also $n$ needs to be
limited in the future by running multiple smooths on subsets of the data and then
combining the results.<br>
❗❗❗

### 🧑‍💻 Final Implementation

The updated weights matrix is then given by

$\mathbf{W^{+}}=\mathbf{W}+\epsilon_{1}\cdot\mathbf{I}=\mathbf{W}+\varepsilon\cdot max\left(n, 10\right)\cdot\max\left(diag\left(\mathbf{W}\right)\right)\cdot\mathbf{I}$

where the $max$-operator was included to prevent $\epsilon_{1}$ from becoming too small.
Lifting the weights is not a problem because weights that need to be lifted will still
be negligible compared to the other weights afterwards.<br>
Analogously, the updated penalty matrix is given by

$\mathbf{P}^{+}=\mathbf{P}+\epsilon_{2}\cdot\mathbf{I}=\mathbf{P}+\varepsilon\cdot max\left(n, 10\right)\cdot 4^{m}\cdot\mathbf{I}$

From a mathematical point of view, this approach introduces a second penalty term which
is the classical Tikhonov regularization term. Yet, this term is very small and
therefore virtually negligible compared to the actual derivative penalty term.<br>
Nevertheless, the Tikhonov regularization term will penalize large absolute values of
the smoothed values $\mathbf{z}$ which is not desirable since this will pull
$\mathbf{z}$ towards zero. To resolve this, the weighted average of the original values
$\mathbf{y}$ is subtracted before the smoothing is performed and added again afterwards:

$\bar{y}=\frac{\sum_{i=1}^{n}w_{i}\cdot y_{i}}{\sum_{i=1}^{n}w_{i}}$<br>
$\mathbf{z}=\bar{y}+\left(\mathbf{W^{+}}+\tau\cdot\mathbf{P^{+}}\right)^{-1}\mathbf{W^{+}}\left(\mathbf{y}-\bar{y}\right)$

Consequently, $\mathbf{z}$ is pulled towards the weighted average of the original values
$\mathbf{y}$ instead of zero as $\tau$ tends to $+\infty$ which is way more desirable
(note that as $\tau$ tends to $+\infty$ $\mathbf{z}$ becomes a flat line anyway and
making it become the weighted average of $\mathbf{y}$ is mathematically sound).

## 🏄 Extensions

To make the smoothing spatially adaptive, the smoothing parameter $\tau$ is replaced by
a individual smoothing parameters $\tau_{i}$ for each observation $y_{i}$.
Mathematically, this is equivalent to

$\mathbf{z}=\left(\mathbf{W}+\tau\cdot\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}\right)^{-1}\mathbf{W}\mathbf{y}$

where $\mathbf{M}$ is a diagonal matrix of smoothing parameter weights.<br>
Now, the determination of $\epsilon_{p}$ becomes more complicated because the
eigenvalues of $\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}$ are not known. However, they
can be estimated via the spectral norm which is defined as

$\left\Vert\mathbf{A}\right\Vert _{2}=\sqrt{\lambda_{max}\left(\mathbf{A}^{T}\mathbf{A}\right)}$

where $\mathbf{A}$ is a matrix and $\lambda_{max}\left(\mathbf{A}^{T}\mathbf{A}\right)$
is the largest eigenvalue of the matrix product $\mathbf{A}^{T}\mathbf{A}$.<br>
This norm is sub-multiplicative, i.e.,

$\left\Vert \mathbf{A}\mathbf{B}\right\Vert _{2}\leq\left\Vert\mathbf{A}\right\Vert _{2}\cdot\left\Vert\mathbf{B}\right\Vert _{2}$

which means that an upper bound for the maximum eigenvalue of $\mathbf{A}\mathbf{B}$ can
be estimated when the spectral norms of $\mathbf{A}$ and $\mathbf{B}$ are known. This is
the case for $\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}$ since $\mathbf{M}$ is again a
diagonal matrix and the spectral norm of $\mathbf{D}_{m}$ has almost been calculated
above as the maximum eigenvalue of $\mathbf{D}_{m}^{T}\mathbf{D}_{m}$.<br>
It follows that

$\left\Vert\mathbf{M}\right\Vert _{2}=\sqrt{\lambda_{max}\left(\mathbf{M}^{T}\mathbf{M}\right)}=\sqrt{max\left(\mathbf{M}^{T}\mathbf{M}\right)}=\sqrt{max\left(diag\left(\mathbf{M}\right)\right)^{2}}=max\left(abs\left(diag\left(\mathbf{M}\right)\right)\right)$

where $abs\left(diag\left(\mathbf{M}\right)\right)$ is the vector of absolute values of
the diagonal elements of $\mathbf{M}$.<br>
For $\mathbf{D}_{m}^{T}$ the spectral norm is given by

$\left\Vert\mathbf{D}_{m}^{T}\right\Vert _{2}=\sqrt{\lambda_{max}\left(\mathbf{D}_{m}^{T}\mathbf{D}_{m}\right)}=\sqrt{4^{m}}=2^{m}$

Finally, the upper bound for the maximum eigenvalue of
$\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}$ is given by

$\left\Vert\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}\right\Vert _{2}=\sqrt{\lambda_{max}\left(\mathbf{D}_{m}^{T}\mathbf{M}^T\mathbf{D}_{m}\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}\right)}=\lambda_{max}\left(\mathbf{D}_{m}^{T}\mathbf{M}\mathbf{D}_{m}\right)\leq\left\Vert\mathbf{D}_{m}^{T}\right\Vert _{2}\cdot\left\Vert\mathbf{M}\right\Vert _{2}\cdot\left\Vert\mathbf{D}_{m}\right\Vert _{2}=2^{m}\cdot max\left(abs\left(diag\left(\mathbf{M}\right)\right)\right)\cdot 2^{m}=4^{m}\cdot max\left(abs\left(diag\left(\mathbf{M}\right)\right)\right)$

Combining all this $\mathbf{P}^{+}$ can be determined by

$\epsilon_{p}=\varepsilon\cdot max\left(n, 10\right)\cdot 4^{m}\cdot max\left(abs\left(diag\left(\mathbf{M}\right)\right)\right)$<br>
$\mathbf{P}^{+}=\mathbf{P}+\epsilon_{p}\cdot\mathbf{I}=\mathbf{P}+\varepsilon\cdot max\left(n, 10\right)\cdot 4^{m}\cdot max\left(abs\left(diag\left(\mathbf{M}\right)\right)\right)\cdot\mathbf{I}$

This is again a cheap and robust approximation that does not require the calculation of
any eigenvalues.<br>
❗❗❗<br>
Due to the power of $m$, this approximation scales badly with increasing $m$ and
$n$. It is therefore recommended not to use $m > 6$. Probably also $n$ needs to be
limited in the future by running multiple smooths on subsets of the data and then
combining the results.<br>
❗❗❗<br>
Such an approach will be useful for a spatially adaptive smoothing algorithm like the
one provided in

*A. Corbas, S.J. Choquette: "Automated Spectral Smoothing with Spatially Adaptive
Penalized Least Squares", Applied Spectroscopy Volume 65, Issue 6, pp.665-677, 2011
[DOI](https://doi.org/10.1366/10-05971)*
